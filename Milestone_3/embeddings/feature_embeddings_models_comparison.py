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
    test_query(query)
    
for query in RECOMMENDER_QUERIES:
    test_query(query)
    
    driver.close()