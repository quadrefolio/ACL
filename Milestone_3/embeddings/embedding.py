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


def embedding_based_search(intent, query, top_k=5, entities=None):
    """
    Route the query to the appropriate search index based on the intent.
    
    Args:
        intent (str): One of: BOOKING_ACTION, VISA_INFO, RECOMMEND_HOTEL, SEARCH_REVIEW, HOTEL_SEARCH
        query (str): The user's search query
        top_k (int): Number of top results to return
        entities (dict, optional): Extracted entities (cities, countries, hotels, etc.)
        
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
    elif intent == "SEARCH_REVIEW":
        return _search_review_embeddings(intent, query, vector, top_k, entities)
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
                # Parse the rule text to extract requires_visa and visa_type
                rule_text = row['rule']
                
                # Extract "Is a Visa Required? Yes/No"
                requires_visa = "Unknown"
                if "Is a Visa Required?" in rule_text:
                    parts = rule_text.split("Is a Visa Required?")[1].split(".")[0].strip()
                    requires_visa = parts
                
                # Extract "Visa Type: ..."
                visa_type = "Unknown"
                if "Visa Type:" in rule_text:
                    visa_type = rule_text.split("Visa Type:")[1].strip().rstrip(".")
                
                # Add parsed fields to the row
                row['requires_visa'] = requires_visa
                row['visa_type'] = visa_type
                
                print(f"   {i}. [{round(row['score'], 4)}] {row['origin']} → {row['destination']}")
                print(f"      Visa Required: {requires_visa} | Type: {visa_type}")
                print()
        else:
            print("   ❌ No visa rules found")
        
        return {
            "intent": "VISA_INFO",
            "query": query,
            "results": result,
            "count": len(result)
        }

def _search_review_embeddings(intent, query, vector, top_k=5, entities=None):
    """
    Search for reviews using a two-step process:
    1. Find relevant hotels using semantic search on hotel embeddings
    2. Return reviews for those hotels
    
    Args:
        intent (str): The specific intent type (SEARCH_REVIEW)
        query (str): The user's search query
        vector (list): Pre-computed embedding vector
        top_k (int): Number of top results to return
        entities (dict, optional): Extracted entities for filtering (cities, countries, etc.)
        
    Returns:
        dict: Review search results with review text and hotel names
    """
    with driver.session() as session:
        # STEP 1: Find relevant hotels using hotel embeddings
        # This allows queries like "Times Square" to find "The Azure Tower"
        hotel_results = session.run("""
            CALL db.index.vector.queryNodes('hotel_minilm', 3, $vec)
            YIELD node, score
            MATCH (node)-[:LOCATED_IN]->(c:City)
            RETURN node.name as hotel_name, c.name as city, score
            ORDER BY score DESC
        """, vec=vector).data()
        
        if not hotel_results:
            print(f"📊 REVIEW VECTOR RESULTS ({intent}):")
            print("   ❌ No hotels found matching the query")
            return {
                "intent": intent,
                "query": query,
                "results": [],
                "count": 0
            }
        
        # Extract hotel names with their ranking order
        hotel_names = [h['hotel_name'] for h in hotel_results]
        
        # STEP 2: Get reviews for those hotels, preserving hotel ranking order
        # Reviews from the top-ranked hotel should appear first
        result = session.run("""
            MATCH (r:Review)-[:REVIEWED]->(h:Hotel)-[:LOCATED_IN]->(c:City)
            WHERE h.name IN $hotel_names
            WITH r, h, c,
                 CASE h.name
                     WHEN $hotel1 THEN 1
                     WHEN $hotel2 THEN 2
                     WHEN $hotel3 THEN 3
                     ELSE 4
                 END as hotel_rank
            RETURN r.text_representation as review_text, 
                   h.name as hotel_name,
                   c.name as city,
                   r.score_overall as rating,
                   hotel_rank
            ORDER BY hotel_rank ASC, r.score_overall DESC
            LIMIT $k
        """, k=top_k, hotel_names=hotel_names,
             hotel1=hotel_names[0] if len(hotel_names) > 0 else '',
             hotel2=hotel_names[1] if len(hotel_names) > 1 else '',
             hotel3=hotel_names[2] if len(hotel_names) > 2 else '').data()
        
        # Add a dummy score for consistency with other search results
        for row in result:
            row['score'] = row['rating'] / 10.0  # Normalize rating to 0-1 range
        
        print(f"📊 REVIEW VECTOR RESULTS ({intent}):")
        hotel_list = ', '.join([f"{h['hotel_name']} ({h['city']})" for h in hotel_results])
        print(f"   Found hotels: {hotel_list}")
        print()
        
        if result:
            for i, row in enumerate(result, 1):
                hotel_name = row.get('hotel_name', 'Unknown Hotel')
                city = row.get('city', 'Unknown City')
                review_text = row.get('review_text', 'No review text')
                rating = row.get('rating', 'N/A')
                print(f"   {i}. [{round(row['score'], 4)}] {hotel_name} ({city}) - Rating: {rating}/10")
                if review_text and review_text != 'No review text':
                    print(f"      Review: {review_text[:100]}..." if len(review_text) > 100 else f"      Review: {review_text}")
                else:
                    print(f"      Review: No review text available")
                print()
        else:
            print("   ❌ No reviews found for these hotels")
        
        return {
            "intent": intent,
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
        dict: Hotel search results with ratings and city information
    """
    with driver.session() as session:
        # Query the HOTEL embeddings index (MiniLM) with base properties from Hotel nodes
        result = session.run("""
            CALL db.index.vector.queryNodes('hotel_minilm', $k, $vec)
            YIELD node, score
            MATCH (node)-[:LOCATED_IN]->(c:City)
            WITH node, c, score,
                 (node.cleanliness_base + node.comfort_base + node.facilities_base) / 3.0 as overall_rating
            RETURN node.name as hotel, 
                   c.name as city,
                   node.star_rating as stars,
                   round(overall_rating * 10) / 10 as rating,
                   node.comfort_base as comfort_rating,
                   node.cleanliness_base as cleanliness_rating,
                   node.facilities_base as value_for_money_rating,
                   score
            ORDER BY score DESC
        """, vec=vector, k=top_k).data()
        
        print(f"📊 HOTEL VECTOR RESULTS ({intent}):")
        if result:
            for i, row in enumerate(result, 1):
                hotel = row.get('hotel', 'Unknown')
                city = row.get('city', 'Unknown')
                stars = row.get('stars', 'N/A')
                rating = row.get('rating', 'N/A')
                comfort = row.get('comfort_rating', 'N/A')
                cleanliness = row.get('cleanliness_rating', 'N/A')
                value = row.get('value_for_money_rating', 'N/A')
                
                print(f"   {i}. [{round(row['score'], 4)}] {hotel} ({city}) - {stars} stars")
                print(f"      Rating: {rating}/10 | Comfort: {comfort}/10 | Cleanliness: {cleanliness}/10 | Value: {value}/10")
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
