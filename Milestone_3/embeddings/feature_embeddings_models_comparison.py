from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import time

# --- CONFIGURATION ---
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678" # <--- UPDATE THIS

# --- 1. LOAD BOTH MODELS ---
print("⏳ Loading Model A (MiniLM - Fast)...")
model_a = SentenceTransformer('all-MiniLM-L6-v2')

print("⏳ Loading Model B (MPNet - Smart)...")
model_b = SentenceTransformer('all-mpnet-base-v2')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def test_query(user_text):
    print(f"\n🔎 TEST QUERY: '{user_text}'")
    print("="*60)
    
    with driver.session() as session:
        # --- TEST MODEL A (MiniLM) ---
        start_a = time.time()
        vector_a = model_a.encode(user_text).tolist()
        
        result_a = session.run("""
            CALL db.index.vector.queryNodes('hotel_minilm', 3, $vec)
            YIELD node, score
            RETURN node.name as hotel, score, node.text_representation as desc
        """, vec=vector_a).data()
        time_a = time.time() - start_a
        
        print(f"🔹 MODEL A (MiniLM) | Time: {round(time_a, 4)}s")
        for row in result_a:
            print(f"   [{round(row['score'], 3)}] {row['hotel']}")

        print("-" * 60)

        # --- TEST MODEL B (MPNet) ---
        start_b = time.time()
        vector_b = model_b.encode(user_text).tolist()
        
        result_b = session.run("""
            CALL db.index.vector.queryNodes('hotel_mpnet', 3, $vec)
            YIELD node, score
            RETURN node.name as hotel, score, node.text_representation as desc
        """, vec=vector_b).data()
        time_b = time.time() - start_b
        
        print(f"🔸 MODEL B (MPNet)  | Time: {round(time_b, 4)}s")
        for row in result_b:
            print(f"   [{round(row['score'], 3)}] {row['hotel']}")
            
        print("="*60)

if __name__ == "__main__":
    # Test Case 1: Specific Feature (Both should get this right)
    test_query("Hotel in Paris with good facilities")
    
    # Test Case 2: Vague "Vibe" (Model B usually wins here)
    test_query("A romantic and quiet place for a couple")
    
    # Test Case 3: Complex/Conflicting (Model B should handle better)
    test_query("Cheap hotel but very clean and safe")
    
    driver.close()