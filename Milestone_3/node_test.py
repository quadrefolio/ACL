from neo4j import GraphDatabase

# --- CONFIGURATION ---
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # <--- UPDATE THIS

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_similarity_test(model_name, property_name, index_name):
    print(f"\n🧪 TESTING MODEL: {model_name}")
    print("="*60)
    
    with driver.session() as session:
        # 1. Pick a Target Hotel
        target_hotel = "The Azure Tower" # or any hotel name you know exists
        
        # 2. Get its vector
        result = session.run(f"""
            MATCH (h:Hotel {{name: $name}})
            RETURN h.{property_name} as vector
        """, name=target_hotel).single()
        
        if not result or not result["vector"]:
            print(f"❌ Error: Could not find vector for '{target_hotel}'.")
            return

        target_vector = result["vector"]
        print(f"✅ Found Target Vector (Size: {len(target_vector)})")

        # 3. Search for Neighbors using the Vector Index
        # This confirms the Index is built and working.
        search_query = f"""
            CALL db.index.vector.queryNodes('{index_name}', 5, $vec)
            YIELD node, score
            RETURN node.name as neighbor, score
        """
        
        neighbors = session.run(search_query, vec=target_vector).data()
        
        print(f"🔎 Nearest Neighbors to '{target_hotel}':")
        for n in neighbors:
            # Skip the hotel itself (score ~ 1.0)
            if n['neighbor'] == target_hotel: continue
            print(f"   - {n['neighbor']} (Score: {round(n['score'], 4)})")

if __name__ == "__main__":
    try:
        # Test 1: Node2Vec (Pure Structure)
        run_similarity_test("Node2Vec", "embedding_node2vec", "hotel_node2vec")
        
        # Test 2: GraphSAGE (Structure + Features)
        run_similarity_test("GraphSAGE", "embedding_graphsage", "hotel_graphsage")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        driver.close()