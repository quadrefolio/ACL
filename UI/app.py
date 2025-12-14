"""
LuxStay AI - GraphRAG Chat Interface
Milestone 3 - ACL Project

A Streamlit-based chat interface for hotel recommendations and visa information
using Graph Retrieval-Augmented Generation (GraphRAG).

Features:
- 🏨 Hotel search and recommendations
- 🛂 Visa requirement information
- 💬 Multi-turn conversation support
- 📊 Knowledge Graph visualization
- 🎯 Intent classification and entity extraction
- 🔍 Three retrieval modes: baseline, embeddings, hybrid
- ⚡ Real-time metrics (latency, tokens, cost)

Architecture:
- UI Layer: app.py (this file) - handles Streamlit UI components
- Integration Layer: backend_helper.py - connects to Milestone_3 backend
- Backend Layer: Milestone_3/ - RAG pipeline, Neo4j, OpenAI integration
"""

import streamlit as st
import time
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import backend helper functions
from backend_helper import (
    initialize_backend,
    process_query,
    create_knowledge_graph_visualization
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LuxStay AI - GraphRAG",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  /* Message Bubbles */
  .bubble {
    border-radius: 12px;
    padding: 12px;
    margin: 2px 0 !important;
    max-width: 70% !important;
    min-width: fit-content !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    word-break: break-word;
    line-height: 1.3;
    white-space: pre-wrap;
  }
            
  .bubble-user {
    background: #e7fce3;
    border: 1px solid #b2e5ad;
    color: #093d05;
    animation: bubbleInRight 250ms ease-out;
  }
  
  .bubble-assistant {
    background: #f5f5f5;
    border: 1px solid #e1e1e1;
    color: #1f2937;
    animation: bubbleInLeft 250ms ease-out;
  }

  /* Chat Row Layout */
  div[data-testid="stChatMessage"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    margin: 8px 0 !important;
    display: flex !important;
    align-items: flex-start !important;
    gap: 8px !important;
    width: fit-content !important;
    max-width: 100% !important;
  }
  
  div[data-testid="stChatMessage"] > * {
    flex: 0 0 auto !important;
  }

  /* User row: right side */
  div[data-testid="stChatMessage"]:has(.bubble-user) {
    justify-content: flex-end !important;
    flex-direction: row-reverse !important;
    gap: 6px !important;
    margin-left: auto !important;
  }

  /* Assistant row: left side */
  div[data-testid="stChatMessage"]:has(.bubble-assistant) {
    justify-content: flex-start !important;
    flex-direction: row !important;
    margin-right: auto !important;
  }

  /* Avatar positioning */
  div[data-testid="stChatMessage"] [data-testid*="chatAvatarIcon-"],
  div[data-testid="stChatMessage"] [data-testid*="chatAvatarImg-"] {
    margin: 0 !important;
    align-self: center !important;
  }

  /* Typing Animations */
  @keyframes bubbleInLeft {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  
  @keyframes bubbleInRight {
    from { opacity: 0; transform: translateX(12px); }
    to   { opacity: 1; transform: translateX(0); }
  }

  /* Chat Input */
  [data-testid="stChatInput"] textarea,
  [data-testid="stChatInput"] input {
    width: 100% !important;
    min-height: 44px !important;
    padding: 10px 14px !important;
    font-size: 16px !important;
    background: transparent !important;
    border: 0px solid #e5e7eb !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    color: #ffffff !important;
  }

  [data-testid="stChatInput"] textarea::placeholder,
  [data-testid="stChatInput"] input::placeholder {
    color: #9ca3af !important;
    opacity: 0.7 !important;
  }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - Conversation Management
# ═══════════════════════════════════════════════════════════════════════════════

def derive_title(messages):
    """Derive conversation title from first user message."""
    for m in messages:
        if m["role"] == "user":
            t = m["content"].strip().split("\n")[0]
            return (t[:12] + "...") if len(t) > 30 else t
    return "New Chat"


def ensure_conversation():
    """Ensure there is an active conversation."""
    if not st.session_state.active_conversation_id:
        cid = str(uuid.uuid4())
        st.session_state.conversations[cid] = {
            "title": "New Chat",
            "messages": [],
            "ts": time.time()
        }
        st.session_state.active_conversation_id = cid


def get_active_messages():
    """Get messages for the active conversation."""
    cid = st.session_state.active_conversation_id
    if not cid or cid not in st.session_state.conversations:
        return []
    return st.session_state.conversations[cid]["messages"]


def add_message(role, content, **kwargs):
    """Add a message to the active conversation."""
    ensure_conversation()
    cid = st.session_state.active_conversation_id
    msg = {"role": role, "content": content, **kwargs}
    st.session_state.conversations[cid]["messages"].append(msg)
    
    # Update conversation title
    conv = st.session_state.conversations[cid]
    if conv["title"] == "New Chat":
        conv["title"] = derive_title(conv["messages"])


def delete_conversation(cid):
    """Delete a conversation by ID."""
    if cid in st.session_state.conversations:
        del st.session_state.conversations[cid]
    if st.session_state.active_conversation_id == cid:
        st.session_state.active_conversation_id = None


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if "config" not in st.session_state:
    st.session_state.config = {
        "model": "gpt-4o-mini",
        "strategy": "hybrid"
    }

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

if "pre_selected_prompt" not in st.session_state:
    st.session_state.pre_selected_prompt = ""

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None

if "backend" not in st.session_state:
    st.session_state.backend = initialize_backend()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - Configuration & Chat History
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("LuxStay AI")
    st.divider()

    # System Status
    st.subheader("📡 System Status")
    if st.session_state.backend:
        st.success("✅ Backend Connected")
        st.caption("Neo4j database online")
    else:
        st.error("❌ Backend Offline")
        st.caption("Check connection settings")
    
    st.divider()
    
    # System Configuration
    st.subheader("⚙️ System Config")
    
    # LLM Model Selection
    selected_model = st.selectbox(
        "LLM Model",
        ["gpt-4o-mini", "gpt-4.1-mini"],
        index=["gpt-4o-mini", "gpt-4.1-mini"].index(
            st.session_state.config.get("model", "gpt-4o-mini")
        ),
        help="Select the LLM model for response generation",
        key="sb_model"
    )
    
    # Retrieval Method Selection
    selected_strategy = st.selectbox(
        "Retrieval Method",
        ["baseline", "embeddings", "hybrid"],
        index=["baseline", "embeddings", "hybrid"].index(
            st.session_state.config.get("strategy", "hybrid")
        ),
        help="Choose retrieval strategy: baseline (Cypher only), embeddings (vector search), or hybrid (both)",
        key="sb_strategy"
    )
    
    # Update config
    st.session_state.config["model"] = selected_model
    st.session_state.config["strategy"] = selected_strategy

    st.divider()
    
    # Chat History
    st.subheader("💬 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.active_conversation_id = None
        st.session_state.input_key += 1
        st.rerun()

    if st.session_state.conversations:
        items = sorted(
            st.session_state.conversations.items(),
            key=lambda kv: kv[1]["ts"],
            reverse=True
        )
        
        for cid, conv in items:
            cols = st.columns([8, 3])
            
            with cols[0]:
                if st.button(
                    conv["title"],
                    key=f"sel_{cid}",
                    use_container_width=True,
                    type="primary" if cid == st.session_state.active_conversation_id else "secondary"
                ):
                    st.session_state.active_conversation_id = cid
                    st.rerun()
            
            with cols[1]:
                if st.button("🗑️", key=f"del_{cid}", help="Delete chat", use_container_width=True):
                    delete_conversation(cid)
                    st.rerun()
    else:
        st.caption("No conversations yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════

st.title("Welcome to LuxStay AI")
st.markdown("Ask me about hotels, visa requirements, or travel recommendations.")

# Sample Questions
st.markdown("#### Try these questions:")
sample_questions = [
    "Find me a luxury hotel in Cairo with a pool and a rating above 9.0.",
    "Do travelers from the United States need a visa for France?",
    "What are the top 3 highest-rated hotels by Business travellers?",
    "Show me the reviews for 'The Azure Tower'."
]

cols = st.columns(len(sample_questions))
for i, q in enumerate(sample_questions):
    if cols[i].button(q, key=f"q_btn_{i}", use_container_width=True):
        st.session_state.pre_selected_prompt = q
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

messages = get_active_messages()

for i, msg in enumerate(messages):
    role = msg["role"]
    bubble_class = "bubble bubble-user" if role == "user" else "bubble bubble-assistant"

    with st.chat_message(role):
        st.markdown(
            f'<div class="{bubble_class}">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

        # Display RAG Context & Metrics for Assistant Messages
        if role == "assistant" and "context" in msg:
            with st.expander(f"🕵️ View RAG Context & Metrics (Latency: {msg['metrics']['latency']}s)", expanded=False):
                # Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("⏱️ Latency", f"{msg['metrics']['latency']}s")
                col2.metric("🔤 Tokens", f"{msg['metrics']['tokens']}")
                col3.metric("💰 Cost", f"${msg['metrics']['cost']:.4f}")
                col4.metric("📊 Baseline", f"{msg['metrics'].get('baseline_count', 0)}")
                col5.metric("🎯 Embeddings", f"{msg['metrics'].get('embedding_count', 0)}")
                
                st.divider()
                
                # Context Tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🎯 Intent & Entities", 
                    "📊 Knowledge Graph", 
                    "📦 Retrieved Data", 
                    "🔍 Merged Context"
                ])
                
                with tab1:
                    st.subheader("Query Understanding")
                    
                    intent = msg["context"].get("intent", "Unknown")
                    intent_icons = {
                        "HOTEL_SEARCH": "🏨",
                        "RECOMMEND_HOTEL": "⭐",
                        "VISA_INFO": "🛂",
                        "BOOKING_ACTION": "📅",
                        "SEARCH_REVIEW": "💬"
                    }
                    st.markdown(f"### Intent: {intent_icons.get(intent, '❓')} `{intent}`")
                    
                    entities = msg["context"].get("entities", {})
                    st.markdown("### Extracted Entities")
                    
                    if isinstance(entities, dict) and entities:
                        if entities.get("hotels"):
                            st.markdown(f"**🏨 Hotels:** {', '.join(entities['hotels'])}")
                        if entities.get("cities"):
                            st.markdown(f"**🏙️ Cities:** {', '.join(entities['cities'])}")
                        if entities.get("countries"):
                            st.markdown(f"**🌍 Countries:** {', '.join(entities['countries'])}")
                        if entities.get("traveller_type"):
                            st.markdown(f"**👥 Traveller Type:** {entities['traveller_type']}")
                        if entities.get("demographics"):
                            demo = entities["demographics"]
                            demo_str = []
                            if demo.get("gender"):
                                demo_str.append(f"Gender: {demo['gender']}")
                            if demo.get("age_group"):
                                demo_str.append(f"Age: {demo['age_group']}")
                            if demo_str:
                                st.markdown(f"**👤 Demographics:** {', '.join(demo_str)}")
                    else:
                        st.info("No entities extracted.")
                
                with tab2:
                    st.subheader("Knowledge Graph Visualization")
                    if msg["context"].get("graph_viz"):
                        st.graphviz_chart(msg["context"]["graph_viz"])
                    else:
                        st.info("No graph visualization available.")
                
                with tab3:
                    st.subheader("Retrieved Data from Neo4j")
                    
                    baseline = msg["context"].get("baseline_results", [])
                    if baseline:
                        st.markdown("#### 🗄️ Baseline Results (Cypher Query)")
                        st.json(baseline[:5])
                    
                    embeddings = msg["context"].get("embedding_results", [])
                    if embeddings:
                        st.markdown("#### 🎯 Embedding Results (Vector Search)")
                        st.json(embeddings[:5])
                    
                    if not baseline and not embeddings:
                        st.info("No data retrieved.")
                
                with tab4:
                    st.subheader("Merged Context (RAG Input)")
                    context_text = msg["context"].get("merged_context", "")
                    if context_text:
                        st.text_area("Context provided to LLM:", context_text, height=300, key=f"ctx_{i}")
                    else:
                        st.info("No merged context available.")

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

prompt = st.chat_input(
    "Ask about hotels, visa requirements, or travel recommendations...",
    key=st.session_state.input_key
)

if st.session_state.pre_selected_prompt:
    prompt = st.session_state.pre_selected_prompt
    st.session_state.pre_selected_prompt = ""

if prompt:
    model = st.session_state.config["model"]
    strategy = st.session_state.config["strategy"]
    
    # Add user message
    add_message("user", prompt)

    # Generate assistant response
    with st.chat_message("assistant", avatar="🏨"):
        with st.spinner("🔍 Consulting Knowledge Graph..."):
            response = process_query(prompt, model, strategy)
        
        answer_text = response["llm_answer"]
        
        # Display response
        st.markdown(
            f'<div class="bubble bubble-assistant">{answer_text}</div>',
            unsafe_allow_html=True
        )
        
        # Display RAG Context & Metrics
        with st.expander(f"🕵️ View RAG Context & Metrics (Latency: {response['metrics']['latency']}s)", expanded=False):
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("⏱️ Latency", f"{response['metrics']['latency']}s")
            col2.metric("🔤 Tokens", f"{response['metrics']['tokens']}")
            col3.metric("💰 Cost", f"${response['metrics']['cost']:.4f}")
            col4.metric("📊 Baseline", f"{response['metrics'].get('baseline_count', 0)}")
            col5.metric("🎯 Embeddings", f"{response['metrics'].get('embedding_count', 0)}")
            
            st.divider()
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Intent & Entities",
                "📊 Knowledge Graph",
                "📦 Retrieved Data",
                "🔍 Merged Context"
            ])
            
            with tab1:
                st.subheader("Query Understanding")
                intent = response.get("intent", "Unknown")
                intent_icons = {
                    "HOTEL_SEARCH": "🏨",
                    "RECOMMEND_HOTEL": "⭐",
                    "VISA_INFO": "🛂",
                    "BOOKING_ACTION": "📅",
                    "SEARCH_REVIEW": "💬"
                }
                st.markdown(f"### Intent: {intent_icons.get(intent, '❓')} `{intent}`")
                
                entities = response.get("entities", {})
                st.markdown("### Extracted Entities")
                
                if isinstance(entities, dict) and entities:
                    if entities.get("hotels"):
                        st.markdown(f"**🏨 Hotels:** {', '.join(entities['hotels'])}")
                    if entities.get("cities"):
                        st.markdown(f"**🏙️ Cities:** {', '.join(entities['cities'])}")
                    if entities.get("countries"):
                        st.markdown(f"**🌍 Countries:** {', '.join(entities['countries'])}")
                    if entities.get("traveller_type"):
                        st.markdown(f"**👥 Traveller Type:** {entities['traveller_type']}")
                    if entities.get("demographics"):
                        demo = entities["demographics"]
                        demo_str = []
                        if demo.get("gender"):
                            demo_str.append(f"Gender: {demo['gender']}")
                        if demo.get("age_group"):
                            demo_str.append(f"Age: {demo['age_group']}")
                        if demo_str:
                            st.markdown(f"**👤 Demographics:** {', '.join(demo_str)}")
                else:
                    st.info("No entities extracted.")
            
            with tab2:
                st.subheader("Knowledge Graph Visualization")
                graph_viz = create_knowledge_graph_visualization(
                    response.get("baseline_results", []),
                    response.get("embedding_results", []),
                    response.get("intent", ""),
                    response.get("entities", {})
                )
                st.graphviz_chart(graph_viz)
            
            with tab3:
                st.subheader("Retrieved Data from Neo4j")
                
                baseline = response.get("baseline_results", [])
                if baseline:
                    st.markdown("#### 🗄️ Baseline Results (Cypher Query)")
                    st.json(baseline[:5])
                
                embeddings = response.get("embedding_results", [])
                if embeddings:
                    st.markdown("#### 🎯 Embedding Results (Vector Search)")
                    st.json(embeddings[:5])
                
                if not baseline and not embeddings:
                    st.info("No data retrieved.")
            
            with tab4:
                st.subheader("Merged Context (RAG Input)")
                context_text = response.get("merged_context", "")
                if context_text:
                    st.text_area("Context provided to LLM:", context_text, height=300, key="ctx_new")
                else:
                    st.info("No merged context available.")

        # Save assistant message with context
        context_data = {
            "intent": response["intent"],
            "entities": response["entities"],
            "baseline_results": response["baseline_results"],
            "embedding_results": response["embedding_results"],
            "merged_context": response["merged_context"],
            "graph_viz": create_knowledge_graph_visualization(
                response["baseline_results"],
                response["embedding_results"],
                response["intent"],
                response["entities"]
            )
        }
        
        add_message(
            "assistant",
            answer_text,
            context=context_data,
            metrics=response["metrics"]
        )

    st.session_state.input_key += 1
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🏨 LuxStay AI - GraphRAG System")
with col2:
    st.caption(f"🤖 Model: {st.session_state.config['model']}")
with col3:
    st.caption("📊 Milestone 3 - ACL Project")

# Cleanup handler
import atexit

def cleanup():
    if st.session_state.backend:
        try:
            st.session_state.backend.close()
            print("✅ Backend connection closed")
        except:
            pass

atexit.register(cleanup)

