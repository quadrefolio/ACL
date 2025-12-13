import os
from dotenv import load_dotenv
from backend import GraphRAGBackend

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
# Model
# ============================
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

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

# FIXED: chain must be llm_prompt, not undefined var
chain = llm_prompt | model | output_parser


# ============================
# Query Function
# ============================
def answer_query(user_query):
    resp = backend.process_query(user_query)

    merged_context = merge_results(
        resp["baseline_results"],
        resp["embedding_results"]
    )

    context_text = "\n\n".join(format_for_context(r) for r in merged_context)

    print("\n======= MERGED RESULTS =======")
    for i, r in enumerate(merged_context, 1):
        print(f"{i}. {r}")
    print("==============================\n")

    answer = chain.invoke({
        "task": user_query,
        "context": context_text
    })

    return answer


def run_rag(
    user_query: str,
    model_name: str = "gpt-4.1-mini",
    retrieval_mode: str = "hybrid"  # baseline | embeddings | hybrid
):
    """
    Main entry point for UI.
    Returns all intermediate + final outputs for transparency.
    """

    # Run backend
    resp = backend.process_query(user_query)

    baseline_results = resp.get("baseline_results", [])
    embedding_results = resp.get("embedding_results", [])

    # Allow UI to choose retrieval mode
    if retrieval_mode == "baseline":
        embedding_results = []
    elif retrieval_mode == "embeddings":
        baseline_results = []

    # Merge
    merged = merge_results(baseline_results, embedding_results)

    # Format context
    context_text = "\n".join(format_for_context(r) for r in merged)

    # Select model dynamically (for UI dropdown)
    model = ChatOpenAI(
        model=model_name,
        temperature=0
    )

    local_chain = llm_prompt | model | output_parser

    llm_answer = local_chain.invoke({
        "task": user_query,
        "context": context_text
    })

    return {
        "baseline_results": baseline_results,
        "embedding_results": embedding_results,
        "merged_context": context_text,
        "llm_answer": llm_answer
    }


# ============================
# Test run
# ============================
if __name__ == "__main__":
    print(answer_query("Hotels in Cairo with good cleanliness"))
