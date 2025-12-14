"""
Visual Flow Diagram - Complete Integration

Run this to see the complete data flow through your system
"""

print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    COMPLETE GRAPHRAG PIPELINE FLOW                             ║
╚════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: USER INTERACTION                                                        │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                 │
│  User opens: http://localhost:8501                                              │
│                                                                                 │
│  [Streamlit UI - app.py]                                                        │
│   ├─ Chat Input: "Find me luxury hotels in Cairo"                              │
│   ├─ Model Selector: [GPT-4]                                                    │
│   └─ Strategy Selector: [Hybrid (Graph + Vector)]                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ User submits query
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: UI PROCESSING                                                           │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                 │
│  Function: generate_response()                                                  │
│   ├─ prompt = "Find me luxury hotels in Cairo"                                 │
│   ├─ model = "GPT-4"                                                            │
│   └─ strategy = "Hybrid (Graph + Vector)"                                      │
│                                                                                 │
│  Calls: query_backend(prompt, model, strategy)                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ query_backend()
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: API WRAPPER                                                             │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                 │
│  [api_wrapper.py] BackendAPI.process_query()                                   │
│                                                                                 │
│  1. Map UI params to backend params:                                            │
│     ├─ "GPT-4" → "gpt-4o-mini"                                                 │
│     └─ "Hybrid (Graph + Vector)" → "hybrid"                                     │
│                                                                                 │
│  2. Call RAG Pipeline:                                                          │
│     run_rag(                                                                    │
│       user_query="Find me luxury hotels in Cairo",                             │
│       model_name="gpt-4o-mini",                                                │
│       retrieval_mode="hybrid"                                                   │
│     )                                                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ run_rag()
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: RAG PIPELINE                                                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                 │
│  [rag_pipeline.py] run_rag()                                                   │
│                                                                                 │
│  1. Call backend for data:                                                      │
│     resp = backend.process_query("Find me luxury hotels in Cairo")            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ backend.process_query()
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: BACKEND PROCESSING                                                      │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                 │
│  [backend.py] GraphRAGBackend.process_query()                                  │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐        │
│  │ A. CLASSIFY INTENT                                                 │        │
│  │    [intent.py] classify_intent()                                   │        │
│  │    ├─ Calls OpenAI GPT                                             │        │
│  │    └─ Returns: "HOTEL_SEARCH"                                      │        │
│  └────────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐        │
│  │ B. EXTRACT ENTITIES                                                │        │
│  │    [entities.py] extract_hotel_entities()                          │        │
│  │    ├─ Calls OpenAI GPT                                             │        │
│  │    └─ Returns: {cities: ["Cairo"], hotels: null, ...}             │        │
│  └────────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐        │
│  │ C. BASELINE RETRIEVAL (Cypher Queries)                             │        │
│  │    [baseline.py] resolve_cypher_query(intent, entities)            │        │
│  │    ├─ Generates: MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)          │        │
│  │    │              WHERE c.name = 'Cairo'                            │        │
│  │    │              RETURN h, c                                       │        │
│  │    ├─ Executes query on Neo4j                                      │        │
│  │    └─ Returns: [                                                   │        │
│  │                  {hotel: "Cairo Marriott", city: "Cairo", ...},   │        │
│  │                  {hotel: "Nile Ritz-Carlton", city: "Cairo", ...} │        │
│  │                ]                                                   │        │
│  └────────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐        │
│  │ D. EMBEDDING RETRIEVAL (Vector Search)                             │        │
│  │    [embedding.py] embedding_based_search()                         │        │
│  │    ├─ Embeds query: "Find me luxury hotels in Cairo"              │        │
│  │    ├─ Searches vector index                                        │        │
│  │    └─ Returns: [                                                   │        │
│  │                  {hotel: "Four Seasons Cairo", score: 0.92, ...}, │        │
│  │                  {hotel: "Marriott Mena House", score: 0.88, ...} │        │
│  │                ]                                                   │        │
│  └────────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
│  Returns: {                                                                     │
│    intent: "HOTEL_SEARCH",                                                      │
│    entities: {cities: ["Cairo"]},                                              │
│    baseline_results: [...],                                                     │
│    embedding_results: [...]                                                     │
│  }                                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Returns to RAG Pipeline
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: MERGE & FORMAT CONTEXT                                                  │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                 │
│  [rag_pipeline.py]                                                             │
│                                                                                 │
│  1. Merge baseline + embedding results:                                         │
│     merged = merge_results(baseline_results, embedding_results)                │
│     Result: [                                                                   │
│       {hotel: "Cairo Marriott", city: "Cairo", rating: 8.5},                  │
│       {hotel: "Nile Ritz-Carlton", city: "Cairo", rating: 9.2},              │
│       {hotel: "Four Seasons Cairo", rating: 8.9},                             │
│       {hotel: "Marriott Mena House", rating: 8.8}                             │
│     ]                                                                           │
│                                                                                 │
│  2. Format for LLM:                                                             │
│     context_text = format_for_context(merged)                                  │
│     Result: "Hotel: Cairo Marriott | Score: 8.5\n                              │
│              Hotel: Nile Ritz-Carlton | Score: 9.2\n                           │
│              Hotel: Four Seasons Cairo | Score: 8.9\n                          │
│              Hotel: Marriott Mena House | Score: 8.8"                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Formatted context
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: LLM GENERATION                                                          │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                 │
│  [rag_pipeline.py] LangChain + OpenAI                                          │
│                                                                                 │
│  Prompt:                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │ SYSTEM: You are a grounded travel assistant. You must ONLY      │           │
│  │         use the provided context.                                │           │
│  │                                                                  │           │
│  │ CONTEXT:                                                         │           │
│  │ Hotel: Cairo Marriott | Score: 8.5                              │           │
│  │ Hotel: Nile Ritz-Carlton | Score: 9.2                           │           │
│  │ Hotel: Four Seasons Cairo | Score: 8.9                          │           │
│  │ Hotel: Marriott Mena House | Score: 8.8                         │           │
│  │                                                                  │           │
│  │ TASK: Answer the user strictly using the context above.         │           │
│  │                                                                  │           │
│  │ USER QUERY: Find me luxury hotels in Cairo                      │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                                 │
│  OpenAI GPT Response:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │ Based on your search, here are some excellent luxury options    │           │
│  │ in Cairo:                                                        │           │
│  │                                                                  │           │
│  │ 1. **Nile Ritz-Carlton** - Rating: 9.2/10                       │           │
│  │    Top choice for luxury travelers with exceptional service     │           │
│  │                                                                  │           │
│  │ 2. **Four Seasons Cairo** - Rating: 8.9/10                      │           │
│  │    Excellent amenities and prime location                       │           │
│  │                                                                  │           │
│  │ 3. **Cairo Marriott** - Rating: 8.5/10                          │           │
│  │    Great value for luxury seekers                               │           │
│  │                                                                  │           │
│  │ 4. **Marriott Mena House** - Rating: 8.8/10                     │           │
│  │    Historic property near the pyramids                          │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                                 │
│  Returns: {                                                                     │
│    llm_answer: "Based on your search...",                                      │
│    baseline_results: [...],                                                     │
│    embedding_results: [...],                                                    │
│    merged_context: "..."                                                        │
│  }                                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Returns to API Wrapper
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: FORMAT GRAPH VISUALIZATION                                             │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                 │
│  [api_wrapper.py] _format_graph_for_ui()                                       │
│                                                                                 │
│  Creates graph structure:                                                       │
│                                                                                 │
│  {                                                                              │
│    "nodes": [                                                                   │
│      {id: "H_Cairo_Marriott", label: "Cairo Marriott", type: "Hotel"},        │
│      {id: "H_Nile_Ritz_Carlton", label: "Nile Ritz-Carlton", type: "Hotel"},  │
│      {id: "C_Cairo", label: "Cairo", type: "City"},                           │
│      {id: "R_H_Cairo_Marriott", label: "Rating: 8.5", type: "Rating"},        │
│      {id: "R_H_Nile_Ritz_Carlton", label: "Rating: 9.2", type: "Rating"}      │
│    ],                                                                           │
│    "edges": [                                                                   │
│      {from: "H_Cairo_Marriott", to: "C_Cairo", label: "LOCATED_IN"},          │
│      {from: "H_Nile_Ritz_Carlton", to: "C_Cairo", label: "LOCATED_IN"},       │
│      {from: "H_Cairo_Marriott", to: "R_H_Cairo_Marriott", label: "HAS_RATING"},│
│      {from: "H_Nile_Ritz_Carlton", to: "R_H_Nile_Ritz_Carlton", label: "HAS_RATING"}│
│    ]                                                                            │
│  }                                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Complete response
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 9: RETURN TO UI                                                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                 │
│  Returns to app.py:                                                             │
│  {                                                                              │
│    "answer": "Based on your search...",                                        │
│    "intent": "HOTEL_SEARCH",                                                    │
│    "entities": {cities: ["Cairo"]},                                            │
│    "cypher_query": "MATCH (h:Hotel)...",                                       │
│    "retrieved_nodes": [...],                                                    │
│    "graph_data": {nodes: [...], edges: [...]},                                 │
│    "latency": 2.3,                                                              │
│    "token_count": 250,                                                          │
│    "cost": 0.0075                                                               │
│  }                                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Display results
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 10: UI DISPLAY                                                             │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                 │
│  [app.py] Displays:                                                             │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │ 🤖 Assistant                                                      │          │
│  │                                                                   │          │
│  │ Based on your search, here are some excellent luxury options    │          │
│  │ in Cairo:                                                         │          │
│  │                                                                   │          │
│  │ 1. **Nile Ritz-Carlton** - Rating: 9.2/10                        │          │
│  │ 2. **Four Seasons Cairo** - Rating: 8.9/10                       │          │
│  │ 3. **Cairo Marriott** - Rating: 8.5/10                           │          │
│  │ 4. **Marriott Mena House** - Rating: 8.8/10                      │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                 │
│  Expander: "🕵️ View RAG Context & Metrics (Latency: 2.3s)"                     │
│                                                                                 │
│  Tab 1: 📊 Knowledge Graph                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │      [Cairo Marriott] ──LOCATED_IN──> [Cairo]                    │          │
│  │              │                                                    │          │
│  │              └─────HAS_RATING─────> [Rating: 8.5]                │          │
│  │                                                                   │          │
│  │   [Nile Ritz-Carlton] ──LOCATED_IN──> [Cairo]                    │          │
│  │              │                                                    │          │
│  │              └─────HAS_RATING─────> [Rating: 9.2]                │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                 │
│  Tab 2: 🔍 Cypher Query                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │ MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)                          │          │
│  │ WHERE c.name = 'Cairo'                                            │          │
│  │ RETURN h, c                                                       │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                 │
│  Tab 3: 📦 Retrieved Nodes                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │ [                                                                 │          │
│  │   {"hotel": "Cairo Marriott", "city": "Cairo", "rating": 8.5},  │          │
│  │   {"hotel": "Nile Ritz-Carlton", "city": "Cairo", "rating": 9.2}│          │
│  │ ]                                                                 │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                 │
│  Tab 4: 🎯 Intent & Entities                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │ Intent: `HOTEL_SEARCH`                                            │          │
│  │ Entities: {cities: ["Cairo"], hotels: null}                      │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════════╗
║                              PIPELINE COMPLETE!                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")
