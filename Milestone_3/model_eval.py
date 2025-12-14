import time
import csv
from rag_pipeline import run_rag
from langchain_community.chat_models import ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
os.makedirs("Results", exist_ok=True)

# ----------------------------------
# Models to compare
# ----------------------------------
MODELS = [
    "gpt-4.1-mini",
    "gpt-5-mini",
    "google/gemma-2-2b-it"
]

# ----------------------------------
# Fixed test queries
# ----------------------------------
TEST_QUERIES = [
    "Do I need a visa from United States to United Kingdom?",
    "Recommend hotels for families in Paris",
    "Find hotels with high cleanliness",
    "Show reviews for Hilton London"
]

# ----------------------------------
# Cost per 1K tokens (approximate)
# ----------------------------------
MODEL_COST = {
    "gpt-4.1-mini": 0.0005,
    "gpt-5-mini": 0.0003,
    "google/gemma-2-2b-it": 0.0
}

# ----------------------------------
# Evaluation runner
# ----------------------------------
def evaluate_models():
    results = []

    for query in TEST_QUERIES:
        for model in MODELS:

            try:
                start = time.time()

                output = run_rag(
                    user_query=query,
                    model_name=model,
                    retrieval_mode="hybrid"
                )

                elapsed = round(time.time() - start, 2)

                tokens = output.get("tokens", {})
                total_tokens = tokens.get("total_tokens", 0)
                cost = round((total_tokens / 1000) * MODEL_COST[model], 6)

                results.append({
                    "query": query,
                    "model": model,
                    "response_time_sec": elapsed,
                    "total_tokens": total_tokens,
                    "estimated_cost": cost,
                    "answer": output["llm_answer"]
                })

            except Exception as e:
                # ⛔ If gemma or any model fails, log the error instead of crashing
                results.append({
                    "query": query,
                    "model": model,
                    "response_time_sec": None,
                    "total_tokens": None,
                    "estimated_cost": None,
                    "answer": f"ERROR: {str(e)}"
                })
                continue

    return results

# ----------------------------------
# Save results
# ----------------------------------
def save_results(results, filename="Results/model_comparison.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved to {filename}")

# ----------------------------------
# Run evaluation
# ----------------------------------
if __name__ == "__main__":
    results = evaluate_models()
    save_results(results)