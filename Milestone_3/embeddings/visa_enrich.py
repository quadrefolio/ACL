
import pandas as pd
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

print("⏳ Loading Model...")
model = SentenceTransformer('all-MiniLM-L6-v2') 

def enrich_visa_rules():
    print("⚙️ Processing Visa Rules...")
    

    df = pd.read_csv('archive/visa.csv')
    
    with driver.session() as session:
        # 1. Create Constraint (to avoid duplicates)
        try:
            session.run("CREATE CONSTRAINT FOR (v:VisaRule) REQUIRE v.id IS UNIQUE")
        except: pass # Ignore if exists

        # 2. Create Vector Index for Visa Rules
        print("   - Creating Visa Vector Index...")
        session.run("""
            CREATE VECTOR INDEX visa_embeddings IF NOT EXISTS
            FOR (v:VisaRule) ON (v.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
        """)

        # 3. Process Each Rule
        count = 0
        for index, row in df.iterrows():
            origin = row['from']
            dest = row['to']
            req = row['requires_visa']
            v_type = row['visa_type']
            
            # --- THE MAGIC STEP: Create a Natural Language Sentence ---
            # This allows the AI to "find" this rule using semantic search.
            
            rule_text = (
                f"Visa Requirement Rule: For travelers from {origin} going to {dest}. "
                f"Is a Visa Required? {req}. "
                f"Visa Type: {v_type}."
            )
            
            # Generate Vector
            vector = model.encode(rule_text).tolist()
            
            # Create a specific 'VisaRule' node (independent of the Country graph for pure retrieval)
            # We create a unique ID like "United States-France"
            rule_id = f"{origin}-{dest}"
            
            session.run("""
                MERGE (v:VisaRule {id: $rid})
                SET v.text = $text,
                    v.embedding = $vec,
                    v.origin = $origin,
                    v.destination = $dest
            """, rid=rule_id, text=rule_text, vec=vector, origin=origin, dest=dest)
            
            count += 1
            if count % 50 == 0: print(f"   ... processed {count} rules")

    print(f"✅ Success! Created {count} VisaRule nodes with embeddings.")

if __name__ == "__main__":
    try:
        enrich_visa_rules()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        driver.close()