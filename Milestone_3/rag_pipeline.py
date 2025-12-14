import os
from dotenv import load_dotenv
from backend import GraphRAGBackend

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

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
    Universal merge function that works for ANY entity type:
    hotels, visa rules, flights, cities, attractions, etc.
    """
    print("Baseline results:\n", baseline)
    merged = []
    seen = set()

    def make_key(item):
        """
        Create a unique key to detect duplicates.
        """
        if "hotel" in item: return f"hotel:{item.get('hotel')}"
        if "Hotel" in item: return f"hotel:{item.get('Hotel')}"
        if "from_country" in item and "to_country" in item:
            return f"visa:{item['from_country']}->{item['to_country']}"
        if "city" in item: return f"city:{item.get('city')}"
        if "country" in item: return f"country:{item.get('country')}"
        return str(item)

    # Add both baseline + embeddings
    for source in (baseline, embeddings):
        for res in source:
            key = make_key(res)
            if key not in seen:
                merged.append(res)
                seen.add(key)

    print("Merged results:\n", merged)

    return merged


def format_for_context(res):
    """
    Convert ANY result into readable context.
    Ensures no important fields are lost.
    """

    # ---- HOTEL NAMES ----
    hotel = res.get("Hotel") or res.get("hotel")
    if hotel:

        # pick rating-like keys intelligently
        rating = (
            res.get("Avg_Rating_By_Group") or
            res.get("average_reviews_score") or
            res.get("Value_Score") or
            res.get("Safety_Location_Score") or
            res.get("Facility_Score") or
            res.get("cleanliness_score") or
            res.get("score") or
            res.get("Rating") or
            "N/A"
        )

        return f"Hotel: {hotel} | Score: {rating}"

    # ---- VISA RULES ----
    if "from_country" in res and "to_country" in res:
        return (
            f"Visa rule: From {res['from_country']} to {res['to_country']} | "
            f"Visa Required: {res.get('visa_required', 'Unknown')} | "
            f"Visa Type: {res.get('visa_type', 'N/A')}"
        )

    # ---- CITY INFO ----
    if "City" in res or "city" in res:
        return f"City: {res.get('city')} | Country: {res.get('country', 'N/A')}"

    # ---- FALLBACK ----
    return ", ".join(f"{k}: {v}" for k, v in res.items())


# ============================
# Prompt (context + persona + task)
# ============================
llm_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a grounded travel assistant. "
     "You must ONLY use the provided context. "
     "If the context does not contain the answer, say: "
     "'I do not have enough information to answer.' "
     "Do not guess or hallucinate."
    ),

    ("user",
     "PERSONA: You are a travel assistant helping with hotels & visa.\n"
     "USER QUERY:\n{task}\n\n"
     "CONTEXT:\n{context}\n\n"
     "TASK:\nAnswer the user strictly using the context above."
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
    result = run_rag(
        user_query=user_query,
        model_name="gpt-4.1-mini",
        retrieval_mode="hybrid"
    )
    return result["llm_answer"]

def get_model(model_name):
    if model_name in ["gpt-4.1-mini", "gpt-5-mini"]:
        return ChatOpenAI(
            model=model_name,
            temperature=0
        )

    if model_name == "google/gemma-2-2b-it":
        pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto"
        )
        return ChatHuggingFace(pipeline=pipe)
    raise ValueError(f"Unknown model: {model_name}")


def run_rag(
    user_query: str,
    model_name: str = "gpt-4.1-mini",
    retrieval_mode: str = "hybrid"
):
    """
    Main entry point for UI + evaluation (with token tracking).
    """

    # 1. Run backend
    resp = backend.process_query(user_query)

    baseline_results = resp.get("baseline_results", [])
    embedding_results = resp.get("embedding_results", [])

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

    # ---- HuggingFace models (NO token metadata) ----
    else:
        llm_answer = local_chain.invoke(prompt_variables)

        # Approximate token count
        prompt_text = llm_prompt.format(**prompt_variables)
        approx_tokens = int(len(prompt_text) / 4)

        token_usage = {
            "input_tokens": approx_tokens,
            "output_tokens": int(len(llm_answer) / 4),
            "total_tokens": approx_tokens + int(len(llm_answer) / 4)
        }

    # 6. Return everything
    return {
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
