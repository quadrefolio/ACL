from typing import Dict, List, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

# Load environment variables
load_dotenv()

# Import modules
from intent import classify_intent
from entities import extract_hotel_entities
from cypher_queries.baseline import resolve_cypher_query
from embeddings.embedding import embedding_based_search


class GraphRAGBackend:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Backend initialized.\n")

    def close(self):
        self.driver.close()

    def classify_intent(self, query: str) -> str:
        intent = classify_intent(query)
        print(f"[Intent] → {intent}")
        return intent

    def extract_entities(self, query: str) -> Dict[str, Any]:
        entities = extract_hotel_entities(query)
        print(f"[Entities] → {entities}")
        return entities

    def execute_cypher(self, query: str, params: Dict) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    def run_baseline_query(self, intent: str, entities: Dict[str, Any]) -> List[Dict]:
        query, params = resolve_cypher_query(intent, entities)

        if not query:
            print("[Baseline] No query matched.")
            return []

        print("[Baseline] Executing query...")
        print(f"           Params: {params}")
        results = self.execute_cypher(query, params)
        print(f"           Found {len(results)} results.")
        return results

    def run_embedding_search(self, intent: str, query: str, top_k: int):
        print("[Embeddings] Running semantic search...")
        response = embedding_based_search(intent, query, top_k)
        return response.get("results", []), response

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def process_query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        print(f"\n🔍 Processing Query: '{user_query}'\n")

        intent = self.classify_intent(user_query)
        entities = self.extract_entities(user_query)

        baseline_results = self.run_baseline_query(intent, entities)

        embedding_results, embedding_meta = self.run_embedding_search(
            intent, user_query, top_k
        )

        return {
            "query": user_query,
            "intent": intent,
            "entities": entities,
            "baseline_results": baseline_results,
            "embedding_results": embedding_results,
            "embedding_metadata": {
                "routed_to": "visa_embeddings" if intent == "VISA_INFO" else "hotel_embeddings",
                "result_count": embedding_meta.get("count", 0)
            }
        }

    # =========================================================================
    # DISPLAY HELPERS
    # =========================================================================

    def format_baseline_result(self, res: Dict) -> str:
        hotel = res.get("Hotel") or res.get("hotel_name") or "Unknown"
        city = res.get("City") or res.get("city") or "N/A"

        if "Visa_Type" in res or "Destination" in res:
            destination = res.get("Destination", "Unknown")
            vtype = res.get("Visa_Type", "Unknown")
            return f"{destination}: {vtype}"


        stars = res.get("Stars") or res.get("stars") or "N/A"
        rating = res.get("Rating") or res.get("rating") or "N/A"

        return f"{hotel} - {stars}★, Rating: {rating}"

    def format_embedding_result(self, result: Dict, intent: str) -> str:
        score = result.get("score", 0.0)

        # VISA result formatting
        if intent == "VISA_INFO":
            origin = result.get("origin")
            dest = result.get("destination")
            rule = result.get("rule", "").strip()

            if origin and dest:
                return f"{origin} → {dest} - {score:.3f}"

            return f"Visa Rule - {score:.3f}"

        # HOTEL result formatting
        hotel = result.get("hotel") or result.get("hotel_name") or "Unknown"
        return f"{hotel} - Similarity: {score:.3f}"



    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================

    def display_results(self, response: Dict[str, Any]):
        print("\n" + "=" * 80)
        print("📊 RESULTS SUMMARY")
        print("=" * 80)

        print(f"Query: {response['query']}")
        print(f"Intent: {response['intent']}\n")

        # ---- Baseline -------------------------------------------------------
        baseline = response["baseline_results"]
        print(f"🗄️ Baseline Results ({len(baseline)})")
        print("-" * 80)

        if baseline:
            for i, res in enumerate(baseline[:5], 1):
                print(f"  {i}. {self.format_baseline_result(res)}")
        else:
            print("  (No results)")
        print()

        # ---- Embeddings -----------------------------------------------------
        embedding = response["embedding_results"]
        meta = response["embedding_metadata"]

        print(f"🎯 Embedding Results ({len(embedding)})")
        print(f"  Routed to: {meta.get('routed_to')}")
        print("-" * 80)

        if embedding:
            for i, res in enumerate(embedding[:5], 1):
                print(f"  {i}. {self.format_embedding_result(res, response['intent'])}")
        else:
            print("  (No results)")

        print("\n" + "=" * 80 + "\n")


# ============================================================================
# TEST RUNNER
# ============================================================================

def test_backend():
    backend = GraphRAGBackend(
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD"),
    )

    test_queries = [
        "Find me luxury hotels in Paris",
        "Do I need a visa from United States to United Kingdom?",
        "Show me hotels with excellent facilities in Tokyo",
        "Best hotels for families",
        "Hotels in Cairo with good cleanliness",
    ]

    for q in test_queries:
        resp = backend.process_query(q)
        backend.display_results(resp)
        input("Press Enter to continue...")

    backend.close()


if __name__ == "__main__":
    test_backend()
