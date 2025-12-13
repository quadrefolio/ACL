"""
End-to-End Test: Intent → Entities → Cypher Query → Results
This file demonstrates the complete pipeline from user query to database results.
"""

import sys
import os


# Add parent directory to path to import from Milestone_3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from intent import classify_intent
from entities import extract_hotel_entities
from cypher_queries.baseline import resolve_cypher_query
from neo4j import GraphDatabase

import os
from dotenv import load_dotenv
load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def execute_cypher_query(driver, query, params):
    """Execute a Cypher query and return results."""
    with driver.session() as session:
        result = session.run(query, params)
        return [dict(record) for record in result]

def run_end_to_end_test(user_query):
    """
    Complete pipeline test:
    1. Classify intent
    2. Extract entities
    3. Route to Cypher query
    4. Execute query
    5. Display results
    """
    print(f"\n{'='*80}")
    print(f"🔍 USER QUERY: '{user_query}'")
    print(f"{'='*80}\n")
    
    # Step 1: Classify Intent
    print("📌 STEP 1: Intent Classification")
    intent = classify_intent(user_query)
    print(f"   → Intent: {intent}\n")
    
    # Step 2: Extract Entities
    print("🔍 STEP 2: Entity Extraction")
    entities = extract_hotel_entities(user_query)
    print(f"   → Entities: {entities}\n")
    
    # Step 3: Route to Cypher Query
    print("🗄️ STEP 3: Query Routing")
    query, params = resolve_cypher_query(intent, entities)
    
    if query:
        print(f"   → Selected Query Template")
        print(f"   → Parameters: {params}\n")
        
        # Step 4: Execute Query
        print("⚡ STEP 4: Executing Cypher Query")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        try:
            results = execute_cypher_query(driver, query, params)
            print(f"   → Found {len(results)} results\n")
            
            # Step 5: Display Results
            print("📊 STEP 5: Results")
            print("-" * 80)
            
            if results:
                for i, result in enumerate(results[:10], 1):  # Show top 10
                    print(f"\n   Result {i}:")
                    for key, value in result.items():
                        if isinstance(value, float):
                            print(f"      {key}: {value:.2f}")
                        else:
                            # Truncate long text
                            val_str = str(value)
                            if len(val_str) > 100:
                                val_str = val_str[:100] + "..."
                            print(f"      {key}: {val_str}")
            else:
                print("   ⚠️ No results found")
            
        except Exception as e:
            print(f"   ❌ Error executing query: {e}")
        finally:
            driver.close()
    else:
        print("   ⚠️ No query template matched for this intent/entity combination")
    
    print(f"\n{'='*80}\n")

def main():
    """Run multiple test cases."""
    
    print("\n" + "="*80)
    print("🚀 END-TO-END GRAPH-RAG PIPELINE TEST")
    print("="*80)
    
    test_queries = [
        # Hotel Search Tests
        # "Find me hotels in Paris",
        # "Show me 5-star hotels in Tokyo",
        # "I'm looking for hotels in Egypt",
        
        # Visa Tests
        "Do I need a visa from United States to United Kingdom?",
        "What are the visa requirements from Egypt to United Kingdom?",
        
        # # Recommendation Tests
        # "Recommend hotels for families",
        # "Best hotels for solo female travelers",
        # "Show me hotels similar to The Royal Compass",
        
        # # Review Tests
        # "Show me reviews for Nile Grandeur",
        # "What do people say about hotels in Dubai?",
        
        # # Booking Tests
        # "Tell me about The Azure Tower",
        # "I want to book The Golden Oasis"
    ]
    
    for query in test_queries:
        run_end_to_end_test(query)
        input("Press Enter to continue to next test...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()