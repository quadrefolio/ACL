
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678" 

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
model = SentenceTransformer('all-MiniLM-L6-v2')

def test_visa_vector_search(query):
    print(f"\n🔎 SEARCH QUERY: '{query}'")
    print("="*80)
    
    # Generate Vector
    vector = model.encode(query).tolist()
    
    with driver.session() as session:
        # QUERY THE VISA INDEX, NOT THE HOTEL INDEX
        result = session.run("""
            CALL db.index.vector.queryNodes('visa_embeddings', 3, $vec)
            YIELD node, score
            RETURN node.text as rule, score
        """, vec=vector).data()
        
        print("📊 VISA VECTOR RESULTS (Should be Rules, not Hotels):")
        for row in result:
            print(f"   [{round(row['score'], 4)}] {row['rule']}")

if __name__ == "__main__":
    test_visa_vector_search("Do I need a visa from United States to United Kingdom?")