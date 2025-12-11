from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import os

from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print("⏳ Loading Text Model for Feature Search...")
model = SentenceTransformer('all-MiniLM-L6-v2')
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def compare_all_embeddings(target_description, top_k=5):
    """
    Compare all three embedding approaches for the same query:
    1. Feature Embedding (MiniLM) - Content-based matching
    2. Node2Vec - Pure structure/topology-based matching
    3. GraphSAGE - Hybrid (structure + features) matching
    """
    print(f"\n{'='*80}")
    print(f"🔎 SEARCH QUERY: '{target_description}'")
    print(f"{'='*80}\n")
    
    # 1. Generate Vector for the Text Query
    text_vector = model.encode(target_description).tolist()
    
    with driver.session() as session:
        
        # ========== APPROACH 1: FEATURE EMBEDDING (MiniLM) ==========
        print(f"📊 APPROACH 1: FEATURE EMBEDDING (Content-Based)")
        print(f"   Using: MiniLM text embeddings (semantic similarity)")
        print(f"   Index: hotel_minilm")
        print("-" * 80)
        
        feature_results = session.run("""
            CALL db.index.vector.queryNodes('hotel_minilm', $k, $vec)
            YIELD node, score
            RETURN node.name as hotel, score
            ORDER BY score DESC
        """, vec=text_vector, k=top_k).data()
        
        if feature_results:
            for i, row in enumerate(feature_results, 1):
                print(f"   {i}. [{round(row['score'], 4)}] {row['hotel']}")
            top_hotel_feature = feature_results[0]['hotel']
        else:
            print("   ❌ No results found")
            return
        
        print()
        
        # ========== APPROACH 2: NODE2VEC (Structure-Based) ==========
        print(f"📊 APPROACH 2: NODE2VEC (Structure-Based)")
        print(f"   Using: Graph topology embeddings (structural similarity)")
        print(f"   Index: hotel_node2vec")
        print(f"   Strategy: Use top feature result as seed, find structural neighbors")
        print("-" * 80)
        
        # Get the Node2Vec vector of the top feature-based result
        seed_vector_res = session.run("""
            MATCH (h:Hotel {name: $name})
            RETURN h.embedding_node2vec as vector
        """, name=top_hotel_feature).single()
        
        if seed_vector_res and seed_vector_res['vector']:
            node2vec_vector = seed_vector_res['vector']
            
            node2vec_results = session.run("""
                CALL db.index.vector.queryNodes('hotel_node2vec', $k, $vec)
                YIELD node, score
                RETURN node.name as hotel, score
                ORDER BY score DESC
            """, vec=node2vec_vector, k=top_k + 1).data()
            
            # Filter out the seed hotel itself
            node2vec_results = [r for r in node2vec_results if r['hotel'] != top_hotel_feature][:top_k]
            
            for i, row in enumerate(node2vec_results, 1):
                print(f"   {i}. [{round(row['score'], 4)}] {row['hotel']}")
        else:
            print("   ❌ Could not find Node2Vec vector for seed hotel")
        
        print()
        
        # ========== APPROACH 3: GRAPHSAGE (Hybrid) ==========
        print(f"� APPROACH 3: GRAPHSAGE (Hybrid: Structure + Features)")
        print(f"   Using: Combined graph topology + node features")
        print(f"   Index: hotel_graphsage")
        print(f"   Strategy: Use top feature result as seed, find hybrid neighbors")
        print("-" * 80)
        
        # Get the GraphSAGE vector of the top feature-based result
        sage_vector_res = session.run("""
            MATCH (h:Hotel {name: $name})
            RETURN h.embedding_graphsage as vector
        """, name=top_hotel_feature).single()
        
        if sage_vector_res and sage_vector_res['vector']:
            graphsage_vector = sage_vector_res['vector']
            
            graphsage_results = session.run("""
                CALL db.index.vector.queryNodes('hotel_graphsage', $k, $vec)
                YIELD node, score
                RETURN node.name as hotel, score
                ORDER BY score DESC
            """, vec=graphsage_vector, k=top_k + 1).data()
            
            # Filter out the seed hotel itself
            graphsage_results = [r for r in graphsage_results if r['hotel'] != top_hotel_feature][:top_k]
            
            for i, row in enumerate(graphsage_results, 1):
                print(f"   {i}. [{round(row['score'], 4)}] {row['hotel']}")
        else:
            print("   ❌ Could not find GraphSAGE vector for seed hotel")
        
        print()
        
        # ========== SUMMARY ==========
        print(f"{'='*80}")
        print(f"📋 SUMMARY")
        print(f"{'='*80}")
        print(f"Seed Hotel (from Feature Search): '{top_hotel_feature}'")
        print(f"\n💡 Interpretation:")
        print(f"   • Feature Embedding: Finds hotels with similar descriptions/content")
        print(f"   • Node2Vec: Finds hotels with similar review patterns (who reviewed them)")
        print(f"   • GraphSAGE: Finds hotels combining both content and review patterns")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    try:
        # Test 1: Specific Location (Feature should excel)
        compare_all_embeddings("Do I need a visa from United States to United Kingdom", top_k=5)
        
        BOOKING_QUERIES = [
            # 1. Direct Feature Search
            "I want a hotel in Paris with excellent cleanliness standards",
            
            # 2. Amenities (Synonyms: "Workout" should match "Gym")
            "Looking for a place in London with a workout facility and spa",
            
            # 3. Location Context (Descriptive)
            "Hotels located in the heart of New York near main attractions",
            
            # 4. Negative Constraint (Hard for Vectors, good for testing MPNet)
            "A cheap hotel that is not dirty", 
            
            # 5. Combined Criteria
            "5-star luxury stay in Dubai with a swimming pool",
            
            # 6. Business Needs
            "Quiet hotel in Tokyo suitable for business meetings with good wifi",
            
            # 7. Visa Context (Testing if it finds the destination country's hotels)
            "Planning a trip to France, do I need travel documents?",
            
            # 8. Vague Location
            "Accommodation in the capital of Italy",
            
            # 9. Specific Hotel Name (fuzzy search)
            "The Rits Carlton or something similar",
            
            # 10. Value Focus
            "Best value for money hotel in Berlin"
            ]

        # --- TASK 2: HOTEL RECOMMENDER SYSTEM ---
        # Goal: Test "Vibe", "Demographics", and "Sentiment".
        # These rely heavily on the Review Embeddings.

        RECOMMENDER_QUERIES = [
            # 1. Demographic: Couples (Vibe: Romantic)
            "A romantic getaway for a couple with beautiful views",
            
            # 2. Demographic: Family (Vibe: Safe, Spacious, Activities)
            "Family friendly resort with activities for kids",
            
            # 3. Demographic: Solo Female (Vibe: Safety, Location)
            "Safe and secure hotel for a solo female traveler",
            
            # 4. Sentiment: Peace/Relaxation
            "I need a very quiet and peaceful place to relax",
            
            # 5. Sentiment: Service Quality
            "Hotels with incredibly helpful and polite staff",
            
            # 6. Activity Based
            "Good base for sightseeing and walking tours",
            
            # 7. Food/Dining
            "Hotel with an amazing breakfast buffet",
            
            # 8. Structural Similarity (Implicit)
            "Something modern and stylish with high-tech amenities",
            
            # 9. "Hidden Gem" (Implied by high review score text)
            "Underrated place that exceeds expectations",
            
            # 10. Negative Filter (Testing if it retrieves complaints - useful for 'What to avoid')
            "Noisy rooms with bad service" 
        ]
            
        for query in BOOKING_QUERIES:
            compare_all_embeddings(query)
            
        for query in RECOMMENDER_QUERIES:
            compare_all_embeddings(query)
        # # Test 2: Vague Vibe (Topology might reveal interesting patterns)
        # compare_all_embeddings("Relaxing resort for families", top_k=5)
        
        # # Test 3: Specific Amenity
        # compare_all_embeddings("best hotel for solo travelers", top_k=5)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()