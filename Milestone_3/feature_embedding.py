import pandas as pd
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import time
import os

# --- CONFIGURATION ---
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # <--- UPDATE THIS

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# --- LOAD MODELS ---
print("⏳ Loading Model A (Fast: all-MiniLM-L6-v2)...")
model_a = SentenceTransformer('all-MiniLM-L6-v2') # 384 dimensions

print("⏳ Loading Model B (Accurate: all-mpnet-base-v2)...")
model_b = SentenceTransformer('all-mpnet-base-v2') # 768 dimensions

def create_indexes(session):
    print("⚙️ Creating 4 Vector Indexes (2 Models x 2 Entities)...")
    
    # --- MODEL A INDEXES (MiniLM - 384 dim) ---
    session.run("""
        CREATE VECTOR INDEX hotel_minilm IF NOT EXISTS
        FOR (h:Hotel) ON (h.embedding_minilm)
        OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
    """)
    session.run("""
        CREATE VECTOR INDEX review_minilm IF NOT EXISTS
        FOR (r:Review) ON (r.embedding_minilm)
        OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
    """)

    # --- MODEL B INDEXES (MPNet - 768 dim) ---
    session.run("""
        CREATE VECTOR INDEX hotel_mpnet IF NOT EXISTS
        FOR (h:Hotel) ON (h.embedding_mpnet)
        OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """)
    session.run("""
        CREATE VECTOR INDEX review_mpnet IF NOT EXISTS
        FOR (r:Review) ON (r.embedding_mpnet)
        OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """)
    print("✅ Indexes Created.")

def enrich_hotels():
    # Read CSV
    if not os.path.exists('../Milestone_2/hotels.csv'):
        print("❌ Error: ../Milestone_2/hotels.csv not found.")
        return
        
    df = pd.read_csv('../Milestone_2/hotels.csv')
    print(f"🏨 Processing {len(df)} Hotels...")
    
    with driver.session() as session:
        for _, row in df.iterrows():
            # Create Feature Vector String
            # Combining: Name, Location, Rating, Facilities Score, Cleanliness Score
            description = (
                f"Hotel: {row['hotel_name']}. "
                f"Location: {row['city']}, {row['country']}. "
                f"Rating: {row['star_rating']} stars. "
                f"Cleanliness: {row['cleanliness_base']}. "
                f"Facilities: {row['facilities_base']}."
            )
            
            # Generate Embeddings
            vec_a = model_a.encode(description).tolist()
            vec_b = model_b.encode(description).tolist()
            
            # Update Node
            session.run("""
                MATCH (h:Hotel {hotel_id: $hid})
                SET h.embedding_minilm = $vec_a,
                    h.embedding_mpnet = $vec_b,
                    h.text_representation = $desc
            """, hid=int(row['hotel_id']), vec_a=vec_a, vec_b=vec_b, desc=description)
            
    print("✅ Hotels Enriched.")

def enrich_reviews():
    if not os.path.exists('../Milestone_2/reviews.csv'):
        print("❌ Error: ../Milestone_2/reviews.csv not found.")
        return

    # Optimization: Read only first 2000 for testing speed (Remove .head() for full run)
    df = pd.read_csv('../Milestone_2/reviews.csv')
    print(f"✍️ Processing {len(df)} Reviews...")
    
    count = 0
    start_time = time.time()
    
    with driver.session() as session:
        for _, row in df.iterrows():
            # Inject Metadata because raw text is synthetic
            enriched_text = (
                f"Overall Score: {row['score_overall']}/10. "
                f"Cleanliness: {row['score_cleanliness']}. "
                f"Comfort: {row['score_comfort']}. "
                f"Staff: {row['score_staff']}. "
                f"Review Text: {str(row['review_text'])}"
            )
            
            # Generate Embeddings
            vec_a = model_a.encode(enriched_text).tolist()
            vec_b = model_b.encode(enriched_text).tolist()
            
            # Update Node
            session.run("""
                MATCH (r:Review {review_id: $rid})
                SET r.embedding_minilm = $vec_a,
                    r.embedding_mpnet = $vec_b,
                    r.text_representation = $text
            """, rid=int(row['review_id']), vec_a=vec_a, vec_b=vec_b, text=enriched_text)
            
            count += 1
            if count % 100 == 0:
                print(f"   ... processed {count} reviews")

    print(f"✅ Reviews Enriched in {round(time.time() - start_time, 2)}s.")

if __name__ == "__main__":
    try:
        with driver.session() as session:
            create_indexes(session)
            enrich_hotels()
            enrich_reviews()
        print("\n🎉 SUCCESS! Graph is ready for Milestone 3.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    finally:
        driver.close()