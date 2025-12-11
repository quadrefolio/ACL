# -*- coding: utf-8 -*-
"""
Integrated Graph-RAG Backend System
Milestone 3 - Advanced Computational Linguistics

This backend combines:
1. Intent Classification (from intent.py)
2. Entity Extraction (from entities.py)
3. Baseline Cypher Queries (from cypher_queries/baselline.py)
4. Feature Embedding Search (direct query encoding)

The system processes user queries and returns combined results from both
structured (Cypher) and semantic (embedding) retrieval methods.
"""

import os
import sys
import io
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# Fix Windows console encoding (handled by embedding.py)
# if sys.platform == 'win32':
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#     sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# Import our modules
from intent import classify_intent
from entities import extract_hotel_entities
from cypher_queries.baselline import resolve_cypher_query
from embeddings.embedding import embedding_based_search


class GraphRAGBackend:
    """
    Unified backend for Graph-RAG hotel assistant.
    
    Pipeline:
    1. User Query → Intent Classification
    2. User Query → Entity Extraction
    3. Intent + Entities → Baseline Cypher Query
    4. User Query → Feature Embedding Search (direct)
    5. Return Both Results Separately
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize backend with Neo4j connection."""
        # Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print("✅ Backend initialized (using shared embedding model from embedding.py)\n")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def execute_cypher(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results."""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def search_by_embedding(self, intent: str, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search using intent-based embedding routing.
        Routes VISA_INFO to visa embeddings, all others to hotel embeddings.
        
        Args:
            intent: The classified intent (VISA_INFO, HOTEL_SEARCH, etc.)
            query_text: The user's query
            top_k: Number of results to return
            
        Returns:
            Dictionary with intent, query, results, and count
        """
        # Use the intent-based search from embedding.py
        return embedding_based_search(intent, query_text, top_k)
    
    def process_query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main pipeline: Process user query through all components.
        
        Args:
            user_query: The user's natural language query
            top_k: Number of results to return from embedding search
        
        Returns:
            Dictionary containing:
            - query: original query
            - intent: classified intent
            - entities: extracted entities
            - baseline_results: results from Cypher baseline queries
            - embedding_results: results from feature embedding search
        """
        print(f"\n{'='*80}")
        print(f"🔍 Processing Query: '{user_query}'")
        print(f"{'='*80}\n")
        
        # ===== STEP 1: Intent Classification =====
        print("📌 Step 1: Intent Classification")
        intent = classify_intent(user_query)
        print(f"   → Intent: {intent}\n")
        
        # ===== STEP 2: Entity Extraction =====
        print("🔍 Step 2: Entity Extraction")
        entities = extract_hotel_entities(user_query)
        print(f"   → Entities:")
        print(f"      - Hotels: {entities.get('hotels')}")
        print(f"      - Cities: {entities.get('cities')}")
        print(f"      - Countries: {entities.get('countries')}")
        print(f"      - Traveller Type: {entities.get('traveller_type')}")
        print(f"      - Demographics: {entities.get('demographics')}\n")
        
        # ===== STEP 3: Baseline Cypher Query =====
        print("🗄️ Step 3: Executing Baseline Cypher Query")
        baseline_query, baseline_params = resolve_cypher_query(intent, entities)
        
        baseline_results = []
        if baseline_query:
            print(f"   → Query Template Selected")
            print(f"   → Parameters: {baseline_params}")
            baseline_results = self.execute_cypher(baseline_query, baseline_params)
            print(f"   → Found {len(baseline_results)} results from baseline\n")
        else:
            print(f"   → No baseline query matched\n")
        
        # ===== STEP 4: Intent-Based Embedding Search =====
        print("🎯 Step 4: Intent-Based Embedding Search (Semantic)")
        print(f"   → Routing based on intent: {intent}")
        embedding_response = self.search_by_embedding(intent, user_query, top_k=top_k)
        embedding_results = embedding_response.get('results', [])
        print(f"   → Found {len(embedding_results)} results from embeddings\n")
        
        return {
            "query": user_query,
            "intent": intent,
            "entities": entities,
            "baseline_results": baseline_results,
            "embedding_results": embedding_results,
            "embedding_metadata": {
                "routed_to": "visa_embeddings" if intent == "VISA_INFO" else "hotel_embeddings",
                "result_count": embedding_response.get('count', 0)
            }
        }
    

    
    def display_results(self, response: Dict[str, Any]):
        """Pretty print the results."""
        print(f"\n{'='*80}")
        print(f"📊 RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Query: {response['query']}")
        print(f"Intent: {response['intent']}\n")
        
        # Baseline Results
        print(f"🗄️ Baseline Results ({len(response['baseline_results'])} found):")
        print("-" * 80)
        for i, result in enumerate(response['baseline_results'][:5], 1):
            hotel = result.get('Hotel') or result.get('hotel_name') or 'Unknown'
            city = result.get('City') or result.get('city') or result.get('Destination') or 'N/A'
            
            # Handle different result types
            if 'Visa_Type' in result:
                print(f"   {i}. {city}: {result.get('Visa_Type')}")
            elif 'Stars' in result or 'stars' in result:
                stars = result.get('Stars') or result.get('stars') or 'N/A'
                rating = result.get('Rating') or result.get('rating') or 'N/A'
                print(f"   {i}. {hotel} ({city}) - {stars}★, Rating: {rating}")
            else:
                print(f"   {i}. {hotel} ({city})")
        
        if not response['baseline_results']:
            print("   (No results)")
        print()
        
        # Embedding Results
        intent = response['intent']
        embedding_meta = response.get('embedding_metadata', {})
        routed_to = embedding_meta.get('routed_to', 'unknown')
        
        print(f"🎯 Embedding Results ({len(response['embedding_results'])} found):")
        print(f"   Routed to: {routed_to}")
        print("-" * 80)
        
        if intent == "VISA_INFO":
            # Display visa results
            for i, result in enumerate(response['embedding_results'][:5], 1):
                origin = result.get('origin', 'N/A')
                destination = result.get('destination', 'N/A')
                score = result.get('score', 0.0)
                rule = result.get('rule', 'N/A')
                print(f"   {i}. {origin} → {destination} (Score: {score:.3f})")
                print(f"      {rule[:100]}..." if len(rule) > 100 else f"      {rule}")
        else:
            # Display hotel results
            for i, result in enumerate(response['embedding_results'][:5], 1):
                hotel = result.get('hotel', result.get('hotel_name', 'Unknown'))
                score = result.get('score', result.get('similarity_score', 0.0))
                print(f"   {i}. {hotel} - Similarity: {score:.3f}")
        
        if not response['embedding_results']:
            print("   (No results)")
        
        print(f"\n{'='*80}\n")


# ============================================================================
# TESTING
# ============================================================================

def test_backend():
    """Test the integrated backend with sample queries."""
    
    # Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"
    
    # Initialize backend
    backend = GraphRAGBackend(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Test queries
    test_queries = [
        "Find me luxury hotels in Paris",
        "Do I need a visa from United States to United Kingdom?",
        "Show me hotels with excellent facilities in Tokyo",
        "Best hotels for families",
        "Hotels in Cairo with good cleanliness",
    ]
    
    print("\n" + "="*80)
    print("🚀 INTEGRATED GRAPH-RAG BACKEND TEST")
    print("="*80)
    
    for query in test_queries:
        response = backend.process_query(query, top_k=5)
        backend.display_results(response)
        
        # Wait for user input to continue
        input("Press Enter to continue to next query...")
    
    backend.close()
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    try:
        test_backend()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
