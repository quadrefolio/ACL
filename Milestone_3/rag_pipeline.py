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
    merged = []
    seen = set()

    # Baseline results
    for res in baseline:
        name = res.get("Hotel")
        if name and name not in seen:
            merged.append({
                "Hotel": name,
                "Rating": res.get("Avg_Rating_By_Group") or res.get("Rating") or "N/A"
            })
            seen.add(name)

    # Embedding results
    for res in embeddings:
        name = res.get("Hotel") or res.get("hotel")
        if name and name not in seen:
            merged.append({
                "Hotel": name,
                "Rating": res.get("score", "N/A")
            })
            seen.add(name)

    return merged


def format_for_context(res):
    hotel = res.get("Hotel")
    if hotel:
        rating = res.get("Rating", "N/A")
        return f"Hotel: {hotel} | Rating: {rating}"
    return str(res)


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

    # Merge baseline + embeddings
    merged_context = merge_results(
        resp["baseline_results"],
        resp["embedding_results"]
    )

    # Format for LLM
    context_text = "\n".join(format_for_context(r) for r in merged_context)

    print("\n======= MERGED RESULTS =======")
    for i, r in enumerate(merged_context, 1):
        print(f"{i}. {r}")
    print("==============================\n")

    # Run LLM
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
    print(answer_query("Best hotels for families"))
