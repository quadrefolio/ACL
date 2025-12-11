import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print("⏳ Loading Text Model for Intent-Based Search...")
model = SentenceTransformer('all-MiniLM-L6-v2')
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def embedding_based_search(intent, query, top_k=5):
    """
    Route the query to the appropriate search index based on the intent.
    
    Args:
        intent (str): One of: BOOKING_ACTION, VISA_INFO, RECOMMEND_HOTEL, SEARCH_REVIEW, HOTEL_SEARCH
        query (str): The user's search query
        top_k (int): Number of top results to return
        
    Returns:
        dict: Search results with metadata
    """
    print(f"\n{'='*80}")
    print(f"🎯 INTENT: {intent}")
    print(f"🔎 QUERY: '{query}'")
    print(f"{'='*80}\n")
    
    # Generate embedding vector for the query
    vector = model.encode(query).tolist()
    
    # Route based on intent
    if intent == "VISA_INFO":
        return _search_visa_embeddings(query, vector, top_k)
    else:
        # All other intents use hotel search
        return _search_hotel_embeddings(intent, query, vector, top_k)

def _search_visa_embeddings(query, vector, top_k=3):
    """
    Search the visa embeddings index for visa requirements.
    
    Args:
        query (str): The user's search query
        vector (list): Pre-computed embedding vector
        top_k (int): Number of top results to return
        
    Returns:
        dict: Visa search results
    """
    with driver.session() as session:
        # Query the VISA embeddings index
        result = session.run("""
            CALL db.index.vector.queryNodes('visa_embeddings', $k, $vec)
            YIELD node, score
            RETURN node.text as rule, node.origin as origin, node.destination as destination, score
            ORDER BY score DESC
        """, vec=vector, k=top_k).data()
        
        print("📊 VISA VECTOR RESULTS:")
        if result:
            for i, row in enumerate(result, 1):
                print(f"   {i}. [{round(row['score'], 4)}] {row['origin']} → {row['destination']}")
                print(f"      {row['rule']}")
                print()
        else:
            print("   ❌ No visa rules found")
        
        return {
            "intent": "VISA_INFO",
            "query": query,
            "results": result,
            "count": len(result)
        }

def _search_hotel_embeddings(intent, query, vector, top_k=5):
    """
    Search the hotel embeddings index for hotel-related queries.
    
    Args:
        intent (str): The specific intent type
        query (str): The user's search query
        vector (list): Pre-computed embedding vector
        top_k (int): Number of top results to return
        
    Returns:
        dict: Hotel search results
    """
    with driver.session() as session:
        # Query the HOTEL embeddings index (MiniLM)
        result = session.run("""
            CALL db.index.vector.queryNodes('hotel_minilm', $k, $vec)
            YIELD node, score
            RETURN node.name as hotel, score
            ORDER BY score DESC
        """, vec=vector, k=top_k).data()
        
        print(f"📊 HOTEL VECTOR RESULTS ({intent}):")
        if result:
            for i, row in enumerate(result, 1):
                print(f"   {i}. [{round(row['score'], 4)}] {row['hotel']}")
        else:
            print("   ❌ No hotels found")
        
        return {
            "intent": intent,
            "query": query,
            "results": result,
            "count": len(result)
        }

def batch_intent_search(queries_with_intents, top_k=5):
    """
    Process multiple queries with their intents in batch.
    
    Args:
        queries_with_intents (list): List of tuples [(intent, query), ...]
        top_k (int): Number of top results to return per query
        
    Returns:
        list: List of search results for each query
    """
    all_results = []
    
    for intent, query in queries_with_intents:
        result = embedding_based_search(intent, query, top_k)
        all_results.append(result)
        print(f"\n{'='*80}\n")
    
    return all_results

if __name__ == "__main__":
    try:
        # Test queries with their intents
        test_queries = [
            # VISA_INFO - should use visa embeddings
            ("VISA_INFO", "Do I need a visa from United States to United Kingdom?"),
            ("VISA_INFO", "What are the visa requirements for traveling from India to France?"),
            ("VISA_INFO", "Can I travel visa-free from Germany to Japan?"),
            
            # HOTEL_SEARCH - should use hotel embeddings
            ("HOTEL_SEARCH", "I want a hotel in Paris with excellent cleanliness standards"),
            ("HOTEL_SEARCH", "Find me a cheap hotel in Rome"),
            
            # RECOMMEND_HOTEL - should use hotel embeddings
            ("RECOMMEND_HOTEL", "Which hotel is the best in Dubai?"),
            ("RECOMMEND_HOTEL", "Looking for a luxury hotel with a swimming pool"),
            
            # SEARCH_REVIEW - should use hotel embeddings
            ("SEARCH_REVIEW", "Show me reviews for hotels in New York"),
            ("SEARCH_REVIEW", "What do people say about hotels near Times Square?"),
            
            # BOOKING_ACTION - should use hotel embeddings
            ("BOOKING_ACTION", "I want to book a room for next weekend in London"),
            ("BOOKING_ACTION", "Find available hotels for my reservation"),
        ]
        
        print("\n" + "="*80)
        print("🚀 INTENT-BASED SEARCH ROUTING SYSTEM")
        print("="*80)
        
        # Process all queries
        results = batch_intent_search(test_queries, top_k=5)
        
        # Summary
        print("\n" + "="*80)
        print("📋 SUMMARY")
        print("="*80)
        visa_count = sum(1 for r in results if r['intent'] == 'VISA_INFO')
        hotel_count = sum(1 for r in results if r['intent'] != 'VISA_INFO')
        print(f"✅ Processed {len(results)} queries")
        print(f"   • {visa_count} routed to VISA embeddings")
        print(f"   • {hotel_count} routed to HOTEL embeddings")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()
