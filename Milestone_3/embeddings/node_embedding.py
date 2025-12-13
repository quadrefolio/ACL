from neo4j import GraphDatabase
import time
import os
from dotenv import load_dotenv
load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_query(session, query, params={}):
    start = time.time()
    result = session.run(query, params)
    # We consume the result to ensure the query actually finished
    summary = result.consume() 
    elapsed = time.time() - start
    return summary, elapsed

def enrich_with_topology():
    with driver.session() as session:
        print("🚀 Connecting to Neo4j GDS...")
        
        # 1. CLEANUP: Drop old graphs if they exist (to avoid errors)
        try:
            session.run("CALL gds.graph.drop('hotel_graph', false)")
            print("   - Cleared old in-memory graphs.")
        except Exception:
            pass


        # 2. PROJECTION: Load the Graph into Memory
        # We need to project Hotels and Reviews to learn the structure.
        # GraphSAGE needs a feature, so we load the 'embedding_minilm' we made earlier.
        # First, ensure ALL nodes have the embedding property (fill missing with zeros)
        print("📦 Preparing nodes (ensuring all have embeddings)...")
        
        # Create a zero vector for nodes without embeddings (MiniLM uses 384 dimensions)
        zero_vector = [0.0] * 384
        
        # Set default embeddings for any Hotel nodes missing the property
        session.run("""
            MATCH (h:Hotel)
            WHERE h.embedding_minilm IS NULL
            SET h.embedding_minilm = $zeroVec
        """, {'zeroVec': zero_vector})
        
        # Set default embeddings for any Review nodes missing the property
        session.run("""
            MATCH (r:Review)
            WHERE r.embedding_minilm IS NULL
            SET r.embedding_minilm = $zeroVec
        """, {'zeroVec': zero_vector})
        
        print("   - All nodes now have embeddings.")
        print("📦 Projecting Graph into Memory...")
        
        # Now project the graph (all nodes guaranteed to have the property)
        project_query = """
        CALL gds.graph.project(
            'hotel_graph',
            {
                Hotel: { properties: ['embedding_minilm'] },
                Review: { properties: ['embedding_minilm'] }
            },
            {
                REVIEWED: { orientation: 'UNDIRECTED' }
            }
        )
        """
        try:
            run_query(session, project_query)
            print("   - Graph projected successfully.")
        except Exception as e:
            print(f"❌ Error projecting graph. Do you have GDS installed? \nError: {e}")
            return

        # 3. ALGORITHM 1: Node2Vec (Pure Structure)
        # "Tell me which hotels are similar based on who reviewed them."
        print("🕸️ Running Node2Vec (Structure Embedding)...")
        node2vec_query = """
        CALL gds.node2vec.write(
            'hotel_graph',
            {
                writeProperty: 'embedding_node2vec',
                embeddingDimension: 64,
                walkLength: 10,
                iterations: 1
            }
        )
        """
        _, t1 = run_query(session, node2vec_query)
        print(f"   ✅ Node2Vec finished in {round(t1, 2)}s.")

        # 4. ALGORITHM 2: GraphSAGE (Structure + Features)
        # "Tell me which hotels are similar based on their neighbors' features."
        # Note: We use the 'embedding_minilm' as the input feature for SAGE.
        print("🧠 Training GraphSAGE (Feature + Structure Embedding)...")
        
        # Step A: Train the model
        # We define a 'modelName' to store the trained logic
        try:
            session.run("CALL gds.model.drop('sage_model', false)")
        except: pass

        train_query = """
        CALL gds.beta.graphSage.train(
            'hotel_graph',
            {
                modelName: 'sage_model',
                featureProperties: ['embedding_minilm'],
                projectedFeatureDimension: 64,
                embeddingDimension: 64,
                epochs: 10,
                searchDepth: 2,
                activationFunction: 'relu',
                aggregator: 'mean'
            }
        )
        """
        print("   - Training GraphSAGE Model (this takes a moment)...")
        _, t2 = run_query(session, train_query)
        
        # Step B: Write the results
        print("   - Writing GraphSAGE results to database...")
        write_sage_query = """
        CALL gds.beta.graphSage.write(
            'hotel_graph',
            {
                writeProperty: 'embedding_graphsage',
                modelName: 'sage_model'
            }
        )
        """
        _, t3 = run_query(session, write_sage_query)
        print(f"   ✅ GraphSAGE finished in {round(t2 + t3, 2)}s.")

        # 5. CREATE INDEXES
        print("⚙️ Creating Indexes for new embeddings...")
        session.run("""
            CREATE VECTOR INDEX hotel_node2vec IF NOT EXISTS
            FOR (h:Hotel) ON (h.embedding_node2vec)
            OPTIONS {indexConfig: {`vector.dimensions`: 64, `vector.similarity_function`: 'cosine'}}
        """)
        session.run("""
            CREATE VECTOR INDEX hotel_graphsage IF NOT EXISTS
            FOR (h:Hotel) ON (h.embedding_graphsage)
            OPTIONS {indexConfig: {`vector.dimensions`: 64, `vector.similarity_function`: 'cosine'}}
        """)
        
        # 6. CLEANUP
        session.run("CALL gds.graph.drop('hotel_graph', false)")
        session.run("CALL gds.model.drop('sage_model', false)")
        print("🧹 Cleanup done.")

if __name__ == "__main__":
    try:
        enrich_with_topology()
        print("\n🎉 SUCCESS! Your graph now has 'embedding_node2vec' and 'embedding_graphsage'.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print("Check: 1. Is Neo4j Running? 2. Is GDS Plugin Installed?")
    finally:
        driver.close()