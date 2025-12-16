import os
from dotenv import load_dotenv
from backend import GraphRAGBackend

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

load_dotenv()

# ============================
# Initialize backend
# ============================
backend = GraphRAGBackend(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USER"),
    os.getenv("NEO4J_PASSWORD")
)

# ============================
# Merge + Clean Results
# ============================
def merge_results(baseline, embeddings):
    """
    Merge without losing ANY information.
    If both baseline and embedding results have extra fields,
    the merged dict keeps ALL of them.
    """
    merged_dict = {}

    def make_key(item):
        if "hotel" in item: return f"hotel:{item.get('hotel')}"
        if "Hotel" in item: return f"hotel:{item.get('Hotel')}"
        if "from_country" in item and "to_country" in item:
            return f"visa:{item['from_country']}->{item['to_country']}"
        if "city" in item: return f"city:{item.get('city')}"
        if "country" in item: return f"country:{item.get('country')}"
        return str(item)

    for source in (baseline, embeddings):
        for res in source:
            key = make_key(res)

            # If first time seeing this result → store copy
            if key not in merged_dict:
                merged_dict[key] = res.copy()
                continue

            # If exists → merge without overwriting
            for k, v in res.items():
                if k not in merged_dict[key]:
                    merged_dict[key][k] = v
                else:
                    # If values differ → keep both
                    if merged_dict[key][k] != v:
                        merged_dict[key][k] = {
                            "baseline_value": merged_dict[key][k],
                            "embedding_value": v
                        }

    merged = list(merged_dict.values())
    return merged


def format_for_context(res):
    """
    Convert ANY result (hotel, visa, city, attraction...) into full readable context
    without losing ANY fields.
    """

    lines = []

    # Add type label
    if "Hotel" in res or "hotel" in res:
        hotel_name = res.get("Hotel") or res.get("hotel")
        lines.append(f"=== HOTEL: {hotel_name} ===")
    elif "from_country" in res and "to_country" in res:
        lines.append("=== VISA RULE ===")
    elif "City" in res or "city" in res:
        lines.append("=== CITY ===")
    else:
        lines.append("=== RESULT ===")

    # Add ALL keys and values
    for k, v in res.items():
        lines.append(f"{k}: {v}")

    return "\n".join(lines)


# ============================
# Query Validation
# ============================
def is_hotel_related_query(query: str) -> bool:
    """
    Check if the query is related to hotels, travel, or visa information.
    Reject code writing, general knowledge, or off-topic requests.
    """
    query_lower = query.lower()
    
    # Keywords that indicate hotel/travel queries
    hotel_keywords = [
        'hotel', 'room', 'booking', 'stay', 'accommodation',
        'resort', 'lodge', 'inn', 'motel', 'guest house',
        'check-in', 'check-out', 'reservation',
        'visa', 'travel', 'trip', 'destination',
        'city', 'country', 'rating', 'review',
        'amenities', 'price', 'location'
    ]
    
    # Check for hotel-related keywords
    for keyword in hotel_keywords:
        if keyword in query_lower:
            return True
    
    # If no hotel keywords found, reject
    return False


# ============================
# Prompt (Persona + Context + Task)
# ============================
llm_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a specialized hotel and travel assistant. You ONLY answer questions about:"
     "- Hotel recommendations and information"
     "- Visa requirements for travel"
     "- Travel destinations and accommodations"
     "\n\n"
     "You MUST NOT:"
     "- Write code or programming scripts"
     "- Answer general knowledge questions unrelated to hotels/travel"
     "- Provide information outside the hotel and travel domain"
     "\n\n"
     "Use the provided context to answer hotel and travel queries accurately. "
     "When hotels are mentioned, provide their details including name, location, rating, and relevant information."
    ),

    ("user",
     "USER QUERY: {task}\n\n"
     "AVAILABLE INFORMATION:\n{context}\n\n"
     "Please provide a helpful response based on the information above. "
     "If multiple hotels are listed, present them in a clear, organized format."
    )
])

output_parser = StrOutputParser()


# ============================
# Query Function
# ============================
def answer_query(user_query):
    """
    Thin wrapper for quick testing.
    Uses the SAME pipeline as UI & evaluation.
    """
    # Validate query before processing
    if not is_hotel_related_query(user_query):
        return "I'm sorry, I can only answer questions about hotels, accommodations, and travel visa information. Please ask a hotel or travel-related question."
    
    result = run_rag(
        user_query=user_query,
        model_name="gpt-4.1-mini",
        retrieval_mode="hybrid"
    )
    return result["llm_answer"]


def is_hf_model(model_name):
    return (
        model_name.startswith("google/") or
        model_name.startswith("meta-") or
        model_name.startswith("mistral")
    )


def get_model(model_name):
    if model_name in ["gpt-4.1-mini", "gpt-5-mini"]:
        return ChatOpenAI(model=model_name, temperature=0)

    # HuggingFace models
    if is_hf_model(model_name):
        pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",   # ✔ let accelerate decide
            torch_dtype="auto"   # ✔ fp16/bf16 depending on GPU
        )
        return HuggingFacePipeline(pipeline=pipe)

    raise ValueError(f"Unknown model: {model_name}")


def build_hf_prompt(task, context):
    return (
        "You are a hotel and travel assistant.\n"
        "Use ONLY the context below.\n"
        "If the answer is not in the context, say:\n"
        "'The available information does not contain this detail.'\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{task}\n\n"
        "ANSWER:"
    )


def run_rag(
    user_query: str,
    model_name: str = "gpt-4.1-mini",
    retrieval_mode: str = "hybrid"
):
    """
    Main entry point for UI + evaluation (with token tracking).
    """

    # 0. Validate query
    if not is_hotel_related_query(user_query):
        return {
            "baseline_results": [],
            "embedding_results": [],
            "merged_context": "",
            "llm_answer": "I'm sorry, I can only answer questions about hotels, accommodations, and travel visa information. Please ask a hotel or travel-related question.",
            "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }

    # 1. Run backend
    resp = backend.process_query(user_query)

    baseline_results = resp.get("baseline_results", [])
    embedding_results = resp.get("embedding_results", [])
    
    intent = resp.get("intent", "UNKNOWN")
    entities = resp.get("entities", {})

    if retrieval_mode == "baseline":
        embedding_results = []
    elif retrieval_mode == "embeddings":
        baseline_results = []

    # 2. Merge
    merged = merge_results(baseline_results, embedding_results)

    # 3. Build context
    context_text = "\n".join(format_for_context(r) for r in merged)

    # 4. Select model
    model = get_model(model_name)
    local_chain = llm_prompt | model | output_parser

    # -------------------------------
    # 5. Run model + track tokens
    # -------------------------------
    prompt_variables = {
        "task": user_query,
        "context": context_text
    }

    # ---- OpenAI models (support .usage_metadata) ----
    if model_name in ["gpt-4.1-mini", "gpt-5-mini"]:
        raw_response = (llm_prompt | model).invoke(prompt_variables)
        llm_answer = raw_response.content

        token_usage = raw_response.usage_metadata

    # ---- HuggingFace models ----
    else:
        prompt_text = build_hf_prompt(user_query, context_text)
        
        raw_output = model.invoke(prompt_text)

        # Take only text after "ANSWER:"
        if "ANSWER:" in raw_output:
            llm_answer = raw_output.split("ANSWER:")[-1].strip()
        else:
            llm_answer = raw_output.strip()

        token_usage = {
            "input_tokens": len(prompt_text) // 4,
            "output_tokens": len(llm_answer) // 4,
            "total_tokens": (len(prompt_text) + len(llm_answer)) // 4
        }

    # 6. Return everything
    return {
        "intent": intent,
        "entities": entities,
        "baseline_results": baseline_results,
        "embedding_results": embedding_results,
        "merged_context": context_text,
        "llm_answer": llm_answer,
        "tokens": token_usage
    }


# ============================
# Test run
# ============================
if __name__ == "__main__":
    print(answer_query("Do I need a visa from United States to United Kingdom?"))
