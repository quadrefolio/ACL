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
# Merge & format results
# ============================
def merge_results(baseline, embeddings):
    merged = []
    seen = set()
    
    # Normalize baseline results
    for res in baseline:
        hotel_name = res.get("Hotel")
        if hotel_name and hotel_name not in seen:
            merged.append({
                "Hotel": hotel_name,
                "Rating": res.get("Avg_Rating_By_Group")  # Use Avg_Rating if exists
            })
            seen.add(hotel_name)
    
    # Normalize embedding results
    for res in embeddings:
        hotel_name = res.get("Hotel") or res.get("hotel")
        if hotel_name and hotel_name not in seen:
            merged.append({
                "Hotel": hotel_name,
                "Rating": res.get("score")  # fallback to embedding score
            })
            seen.add(hotel_name)
    
    return merged


def format_for_context(res):
    if "Hotel" in res:
        hotel = res["Hotel"]
        rating = res.get("Rating", "N/A")
        return f"Hotel: {hotel} - Rating: {rating}"
    elif "Destination" in res:
        destination = res["Destination"]
        visa_type = res.get("Visa_Type", "Unknown")
        return f"Visa Info: {destination} - {visa_type}"
    else:
        return str(res)


# ============================
# Initialize model
# ============================
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# ============================
# Create prompt template
# ============================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("user",
     "Context:\n{context}\n\n"
     "Persona:\nYou are a helpful travel assistant.\n\n"
     "Task:\nAnswer the user's question using only the context. "
     "Do not mention the list, scores, or ratings explicitly; just provide a natural answer.")
])

# ============================
# Output parser
# ============================
output_parser = StrOutputParser()

# ============================
# Full chain
# ============================
chain = prompt | model | output_parser

# ============================
# Query function
# ============================
def answer_query(user_query):
    # Step 1: Process query through backend
    resp = backend.process_query(user_query)

    # Step 2: Merge baseline and embedding results
    merged_context = merge_results(resp["baseline_results"], resp["embedding_results"])
    
    # Step 3: Format merged context for LLM
    context_text = "\n".join([format_for_context(r) for r in merged_context])
    
    # Optional: print merged results for debugging
    print("\n================= MERGED RESULTS =================")
    for i, r in enumerate(merged_context, 1):
        print(f"{i}. {r}")
    print("=================================================\n")
    
    # Step 4: Invoke LLM chain
    result = chain.invoke({
        "context": context_text,
        "task": user_query
    })
    return result

# ============================
# Test
# ============================
if __name__ == "__main__":
    q = "Best hotels for families"
    print(answer_query(q))