"""
Backend Integration Helper
Handles all interactions with the Milestone_3 backend components.

Setup Instructions:
1. Create .env file with Neo4j credentials (see .env.example)
2. Install UI dependencies: pip install -r requirements.txt
3. Ensure Milestone_3 backend modules are available
4. Set up Neo4j vector index (for embeddings mode):
   Run in Neo4j Browser or Milestone_3 setup scripts to create:
   - hotel_minilm index for hotel embeddings
   If not set up, app will automatically fallback to baseline mode
5. Run: streamlit run app.py

Note: The app gracefully handles missing vector indexes by falling back to baseline retrieval.
"""

import sys
import os
from pathlib import Path
import graphviz

# Add Milestone_3 folder to Python path
ui_dir = Path(__file__).resolve().parent
acl_root = ui_dir.parent
milestone3_path = acl_root / "Milestone_3"

if milestone3_path.exists():
    sys.path.insert(0, str(milestone3_path))
else:
    raise FileNotFoundError(f"⚠️ Milestone_3 not found at: {milestone3_path}")

# Import backend modules
from rag_pipeline import run_rag
from backend import GraphRAGBackend


def initialize_backend():
    """
    Initialize Neo4j backend connection.
    
    Returns:
        GraphRAGBackend: Connected backend instance or None if connection fails
    """
    try:
        backend = GraphRAGBackend(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        print("✅ Backend connected successfully")
        return backend
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return None


def process_query(prompt, model_name="gpt-4o-mini", retrieval_mode="hybrid"):
    """
    Process user query through the RAG pipeline.
    
    Args:
        prompt (str): User's input query
        model_name (str): LLM model to use ("gpt-4o-mini" or "gpt-4.1-mini")
        retrieval_mode (str): Retrieval strategy ("baseline", "embeddings", or "hybrid")
    
    Returns:
        dict: Contains intent, entities, baseline_results, embedding_results, 
              merged_context, llm_answer, and metrics
    """
    import time
    start_time = time.time()
    
    try:
        # Run RAG pipeline (handles intent classification, entity extraction, and retrieval)
        rag_response = run_rag(
            user_query=prompt,
            model_name=model_name,
            retrieval_mode=retrieval_mode
        )
        
        # Calculate metrics
        latency = round(time.time() - start_time, 2)
        
        # Estimate token count (rough approximation)
        answer_text = rag_response.get("llm_answer", "")
        context_text = rag_response.get("merged_context", "")
        token_count = int(len(answer_text.split()) * 1.3 + len(context_text.split()) * 1.3)
        
        # Estimate cost based on model
        cost_per_1k_tokens = 0.00015 if "gpt-4o-mini" in model_name else 0.0005
        cost = round((token_count / 1000) * cost_per_1k_tokens, 4)
        
        baseline_results = rag_response.get("baseline_results", [])
        embedding_results = rag_response.get("embedding_results", [])
        
        return {
            "success": True,
            "intent": rag_response.get("intent", "UNKNOWN"),
            "entities": rag_response.get("entities", {}),
            "baseline_results": baseline_results,
            "embedding_results": embedding_results,
            "merged_context": context_text,
            "llm_answer": answer_text,
            "metrics": {
                "latency": latency,
                "tokens": token_count,
                "cost": cost,
                "baseline_count": len(baseline_results),
                "embedding_count": len(embedding_results)
            }
        }
        
    except Exception as e:
        print(f"❌ Query processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "llm_answer": "I encountered an error processing your query. Please try again.",
            "intent": "UNKNOWN",
            "entities": {},
            "baseline_results": [],
            "embedding_results": [],
            "merged_context": "",
            "metrics": {
                "latency": 0,
                "tokens": 0,
                "cost": 0,
                "baseline_count": 0,
                "embedding_count": 0
            }
        }


def create_knowledge_graph_visualization(baseline_results, embedding_results, intent, entities):
    """
    Creates a Knowledge Graph visualization from query results.
    
    Args:
        baseline_results (list): Results from baseline Cypher query
        embedding_results (list): Results from embedding search
        intent (str): Classified intent
        entities (dict): Extracted entities
    
    Returns:
        graphviz.Digraph: Renderable knowledge graph
    """
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', bgcolor='transparent', size='10,6')
    graph.attr('node', fontname='Arial', fontsize='10')
    graph.attr('edge', fontname='Arial', fontsize='9')
    
    # Node styling by type
    node_styles = {
        "Hotel": {"shape": "circle", "style": "filled,rounded", "fillcolor": "#4A90E2", "fontcolor": "white"},
        "City": {"shape": "circle", "style": "filled", "fillcolor": "#F5A623", "fontcolor": "white"},
        "Review": {"shape": "circle", "style": "filled", "fillcolor": "#7ED321", "fontcolor": "white"},
        "User": {"shape": "circle", "style": "filled", "fillcolor": "#BD10E0", "fontcolor": "white"},
        "Traveller": {"shape": "circle", "style": "filled", "fillcolor": "#BD10E0", "fontcolor": "white"},
        "Rating": {"shape": "circle", "style": "filled", "fillcolor": "#50E3C2", "fontcolor": "white"},
        "Visa": {"shape": "circle", "style": "filled", "fillcolor": "#E94B3C", "fontcolor": "white"},
        "Country": {"shape": "circle", "style": "filled", "fillcolor": "#FF6B6B", "fontcolor": "white"}
    }
    
    added_nodes = set()
    
    # Combine results from both baseline and embeddings
    all_results = baseline_results[:5] + embedding_results[:5]
    
    for idx, result in enumerate(all_results):
        # Handle Hotel nodes
        hotel_name = result.get("Hotel") or result.get("hotel_name") or result.get("hotel")
        if hotel_name and hotel_name not in added_nodes:
            graph.node(f"hotel_{idx}", hotel_name, **node_styles["Hotel"])
            added_nodes.add(hotel_name)
            
            # Add city relationship
            city = result.get("City") or result.get("city")
            if city:
                city_id = f"city_{city.replace(' ', '_')}"
                if city not in added_nodes:
                    graph.node(city_id, city, **node_styles["City"])
                    added_nodes.add(city)
                graph.edge(f"hotel_{idx}", city_id, label="LOCATED_IN")
            
            # Add stars node (5-star, 4-star, etc.)
            stars = result.get("Stars") or result.get("stars")
            if stars:
                stars_id = f"stars_{idx}"
                graph.node(stars_id, f"{'⭐' * int(stars)}", **node_styles["Rating"])
                graph.edge(f"hotel_{idx}", stars_id, label="STARS")
            
            # Add overall rating node
            rating = result.get("Rating") or result.get("rating") or result.get("average_reviews_score") or result.get("Score")
            if rating:
                rating_id = f"rating_{idx}"
                graph.node(rating_id, f"★ {rating}", **node_styles["Rating"])
                graph.edge(f"hotel_{idx}", rating_id, label="RATING")
            
            # Add detailed ratings if available (from embeddings)
            comfort = result.get("comfort_rating")
            cleanliness = result.get("cleanliness_rating")
            value = result.get("value_for_money_rating")
            
            if comfort:
                comfort_id = f"comfort_{idx}"
                graph.node(comfort_id, f"Comfort: {comfort}", **node_styles["Rating"])
                graph.edge(f"hotel_{idx}", comfort_id, label="COMFORT")
            
            if cleanliness:
                clean_id = f"clean_{idx}"
                graph.node(clean_id, f"Clean: {cleanliness}", **node_styles["Rating"])
                graph.edge(f"hotel_{idx}", clean_id, label="CLEANLINESS")
            
            if value:
                value_id = f"value_{idx}"
                graph.node(value_id, f"Value: {value}", **node_styles["Rating"])
                graph.edge(f"hotel_{idx}", value_id, label="VALUE")
        
        # Handle Visa nodes
        destination = result.get("Destination")
        visa_type = result.get("Visa_Type")
        if destination and visa_type:
            dest_id = f"country_{destination.replace(' ', '_')}"
            if destination not in added_nodes:
                graph.node(dest_id, destination, **node_styles["Country"])
                added_nodes.add(destination)
            
            visa_id = f"visa_{idx}"
            graph.node(visa_id, visa_type, **node_styles["Visa"])
            graph.edge(dest_id, visa_id, label="REQUIRES")
    
    # Add query context node
    if entities:
        query_entities = []
        if entities.get("cities"):
            query_entities.extend(entities["cities"])
        if entities.get("countries"):
            query_entities.extend(entities["countries"])
        if entities.get("hotels"):
            query_entities.extend(entities["hotels"])
        
        if query_entities:
            graph.node(
                "query",
                f"Query: {', '.join(query_entities[:3])}",
                shape="plaintext",
                fontcolor="#333333",
                fontsize="11"
            )
    
    return graph