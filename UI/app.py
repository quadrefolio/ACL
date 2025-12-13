import streamlit as st
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import uuid

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION (Must be first Streamlit command)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LuxStay AI - GraphRAG",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  /* ===== Message Bubbles ===== */
  .bubble {
    border-radius: 12px;
    padding: 12px;
    margin: 2px 0 !important;     /* tight vertical spacing */
    max-width: 70% !important;
    min-width: fit-content !important;  /* prevent compression */
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    word-break: break-word;        /* prevent overflow for long words */
    line-height: 1.3;
    white-space: pre-wrap;         /* preserve formatting */
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

  /* ===== Chat Row Layout ===== */
  /* Base row: remove any wrapping background and keep compact */
  div[data-testid="stChatMessage"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    margin: 8px 0 !important;
    display: flex !important;
    align-items: flex-start !important;
    gap: 8px !important;                  /* default spacing */
    width: fit-content !important;        /* row hugs its contents */
    max-width: 100% !important;           /* avoid overflow */
  }
  /* Prevent inner stretch to keep icon close to bubble */
  div[data-testid="stChatMessage"] > * {
    flex: 0 0 auto !important;
  }

  /* User row: right side, icon tight to bubble */
  div[data-testid="stChatMessage"]:has(.bubble-user) {
    justify-content: flex-end !important;
    flex-direction: row-reverse !important; /* bubble then icon */
    gap: 6px !important;                    /* tighter gap for right */
    margin-left: auto !important;           /* pin to right edge */
  }

  /* Assistant row: left side */
  div[data-testid="stChatMessage"]:has(.bubble-assistant) {
    justify-content: flex-start !important;
    flex-direction: row !important;
    margin-right: auto !important;          /* pin to left edge */
  }

  /* Avatar icon positioning */
  div[data-testid="stChatMessage"] [data-testid*="chatAvatarIcon-"],
  div[data-testid="stChatMessage"] [data-testid*="chatAvatarImg-"] {
    margin: 0 !important;
    align-self: center !important;
  }

  /* ===== Typing Animations ===== */
  @keyframes bubbleInLeft {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  @keyframes bubbleInRight {
    from { opacity: 0; transform: translateX(12px); }
    to   { opacity: 1; transform: translateX(0); }
  }

  /* ===== Chat Input (Bottom) ===== */
  /* Remove all wrapping backgrounds/shadows */
  [data-testid="stChatInput"],
  [data-testid="stChatInput"] > div {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    width: 100% !important;
    max-width: none !important;
  }

  /* Actual input field — transparent with visible dark text */
  [data-testid="stChatInput"] textarea,
  [data-testid="stChatInput"] input {
    width: 100% !important;           /* stretch across content column */
    min-height: 44px !important;      /* taller for readability */
    padding: 10px 14px !important;    /* comfy padding */
    font-size: 16px !important;       /* readable text */
    background: transparent !important;   /* transparent background */
    border: 0px solid #e5e7eb !important; /* light border */
    border-radius: 12px !important;
    box-shadow: none !important;
    color: #ffffff !important;        /* white text for dark mode */
  }

  /* Placeholder text color */
  [data-testid="stChatInput"] textarea::placeholder,
  [data-testid="stChatInput"] input::placeholder {
    color: #9ca3af !important;        /* light gray placeholder */
    opacity: 0.7 !important;
  }

  /* Optional: reduce page side paddings to visually widen input */
  section.main > div {
    padding-left: 12px !important;
    padding-right: 12px !important;
  }

  /* ===== Misc: remove any stray background from inner chat containers ===== */
  div[data-testid="stChatMessage"] > div {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
  }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS - Conversation Management
# ═══════════════════════════════════════════════════════════════════════════════

def _derive_title(messages):
    """
    Derives a conversation title from the first user message.
    Truncates to 30 characters if too long.
    
    Args:
        messages (list): List of message dictionaries with 'role' and 'content'
    
    Returns:
        str: Title for the conversation (max 30 chars + "...")
    """
    for m in messages:
        if m["role"] == "user":
            t = m["content"].strip().split("\n")[0]
            return (t[:12] + "...") if len(t) > 30 else t
    return "New Chat"


def ensure_conversation():
    """
    Ensures there is an active conversation in session state.
    Creates a new conversation with a unique ID if none exists.
    Called when the user sends their first message.
    """
    if not st.session_state.active_conversation_id:
        cid = str(uuid.uuid4())  # Generate unique conversation ID
        st.session_state.conversations[cid] = {
            "title": "New Chat",
            "messages": [],
            "ts": time.time()
        }
        st.session_state.active_conversation_id = cid


def get_active_messages():
    """
    Retrieves the messages for the currently active conversation.
    
    Returns:
        list: List of message dictionaries for active conversation, or empty list
    """
    cid = st.session_state.active_conversation_id
    if not cid or cid not in st.session_state.conversations:
        return []
    return st.session_state.conversations[cid]["messages"]


def add_message(role, content, **kwargs):
    """
    Adds a new message to the active conversation.
    Creates a conversation if none exists and updates the title if needed.
    
    Args:
        role (str): 'user' or 'assistant'
        content (str): The message text
        **kwargs: Additional fields (e.g., context, metrics, is_new)
    """
    ensure_conversation()  # Create conversation if needed
    cid = st.session_state.active_conversation_id
    msg = {"role": role, "content": content, **kwargs}
    st.session_state.conversations[cid]["messages"].append(msg)
    
    # Update conversation title from first user message
    conv = st.session_state.conversations[cid]
    if conv["title"] == "New Chat":
        conv["title"] = _derive_title(conv["messages"])


def delete_conversation(cid):
    """
    Deletes a conversation by ID and clears it if active.
    
    Args:
        cid (str): The conversation ID to delete
    """
    if cid in st.session_state.conversations:
        del st.session_state.conversations[cid]
    if st.session_state.active_conversation_id == cid:
        st.session_state.active_conversation_id = None

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Legacy messages list (kept for compatibility)
if "messages" not in st.session_state:
    st.session_state.messages = []

# LLM and retrieval configuration
if "config" not in st.session_state:
    st.session_state.config = {
        "model": "GPT-4",
        "strategy": "Hybrid (Graph + Vector)"
    }

# Input key for clearing chat_input after submission
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Pre-selected prompt from sample question buttons
if "pre_selected_prompt" not in st.session_state:
    st.session_state.pre_selected_prompt = ""

# Conversations dictionary: {id: {"title": str, "messages": list, "ts": float}}
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

# Currently active conversation ID
if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MOCK BACKEND FUNCTIONS - Replace with Real RAG Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def get_mock_graph_data(query, location):
    """
    Simulates fetching data from Neo4j knowledge graph.
    Creates a mock graph visualization and Cypher query.
    
    Args:
        query (str): The user's query
        location (str): Detected location/city name
    
    Returns:
        dict: Contains intent, entities, cypher query, nodes, and graph visualization
    """
    # Create a simple Graphviz chart for visualization
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', bgcolor='transparent')
    
    # Nodes
    graph.node('C', f'City: {location}', shape='ellipse', style='filled', fillcolor='#f59e0b', color='white')
    graph.node('H1', f'{location} Ritz', shape='box', style='filled', fillcolor='#bfdbfe', color='white')
    graph.node('H2', f'{location} Marriott', shape='box', style='filled', fillcolor='#bfdbfe', color='white')
    graph.node('R1', 'Rating: 4.9', shape='plaintext', fontcolor='green')
    graph.node('V', 'Visa: Required', shape='diamond', style='filled', fillcolor='#fca5a5', color='white')

    # Edges (relationships)
    graph.edge('H1', 'C', label='LOCATED_IN')
    graph.edge('H2', 'C', label='LOCATED_IN')
    graph.edge('H1', 'R1', label='HAS_RATING')
    graph.edge('C', 'V', label='VISA_RULE', style='dashed')

    return {
        "intent": "HOTEL_SEARCH" if "hotel" in query.lower() else "VISA_CHECK",
        "entities": [location, "Luxury", "Rating > 4.5"],
        "cypher": f"""MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {{name: '{location}'}})
WHERE h.rating >= 4.5
RETURN h.name, h.rating, c.visa_required
LIMIT 3""",
        "nodes": [
            {"name": f"{location} Ritz", "type": "Hotel", "score": 4.9},
            {"name": f"{location} Marriott", "type": "Hotel", "score": 4.7}
        ],
        "graph_viz": graph
    }


def generate_response(prompt, model, strategy):
    """
    Simulates the complete RAG pipeline: Retrieval -> Augmentation -> Generation.
    In production, this would query Neo4j, retrieve embeddings, and call LLM API.
    
    Args:
        prompt (str): User's input query
        model (str): Selected LLM model name
        strategy (str): Retrieval strategy (Baseline, Embeddings, Hybrid)
    
    Returns:
        tuple: (answer_text, context_data, metrics_dict)
    """
    start_time = time.time()
    
    # Simulate processing delay based on model
    delay = 2.0 if model == "GPT-4" else 1.0
    time.sleep(delay)

    # Simple location detection logic
    location = "Cairo" if "cairo" in prompt.lower() else "Paris"
    if "cairo" not in prompt.lower() and "paris" not in prompt.lower():
        location = "Unknown"

    # Retrieve context from knowledge graph
    context_data = get_mock_graph_data(prompt, location)
    
    # Generate answer text
    if location == "Unknown":
        answer = "I can help you with hotel bookings and visa requirements. Could you specify which city you are interested in?"
    else:
        answer = f"Based on our {strategy} search, I found excellent options in **{location}**. \n\n" \
                 f"1. **{location} Ritz**: Rated 4.9/5. Top choice for luxury travelers.\n" \
                 f"2. **{location} Marriott**: Rated 4.7/5. Great amenities and central location.\n\n" \
                 f"⚠️ **Visa Alert**: Travelers from your region require a visa for entry."

    end_time = time.time()
    
    # Calculate performance metrics
    metrics = {
        "latency": round(end_time - start_time, 2),
        "tokens": random.randint(150, 400),
        "cost": 0.0045 if model == "GPT-4" else 0.0002
    }
    
    return answer, context_data, metrics

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SIDEBAR - System Configuration & Chat History
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Header
    st.title("LuxStay AI")
    st.divider()

    # --- System Configuration Section ---
    st.subheader("⚙️ System Config")
    
    # LLM Model Selection
    selected_model = st.selectbox(
        "LLM Model",
        ["GPT-4", "GPT-3.5 Turbo", "Llama-3-70b", "Gemini Pro"],
        index=["GPT-4", "GPT-3.5 Turbo", "Llama-3-70b", "Gemini Pro"].index(
            st.session_state.config.get("model", "GPT-4")
        ),
        help="Select the LLM",
        key="sb_model"
    )
    
    # Retrieval Method Selection
    selected_strategy = st.selectbox(
        "Retrieval Method",
        ["Baseline (Cypher Only)", "Embeddings (Vector)", "Hybrid (Graph + Vector)"],
        index=["Baseline (Cypher Only)", "Embeddings (Vector)", "Hybrid (Graph + Vector)"].index(
            st.session_state.config.get("strategy", "Hybrid (Graph + Vector)")
        ),
        help="Choose retrieval method",
        key="sb_strategy"
    )
    
    # Persist configuration to session state
    st.session_state.config["model"] = selected_model
    st.session_state.config["strategy"] = selected_strategy

    st.divider()
    
    # --- Chat History Section ---
    st.subheader("💬 Chat History")

    # New Chat Button (clears active conversation without creating empty chat)
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.active_conversation_id = None
        st.session_state.input_key += 1
        st.rerun()
    # List existing conversations with select and delete options
    if st.session_state.conversations:
        # Sort conversations by timestamp (most recent first)
        items = sorted(
            st.session_state.conversations.items(),
            key=lambda kv: kv[1]["ts"],
            reverse=True
        )
        
        for cid, conv in items:
            # Create a horizontal layout for each conversation
            cols = st.columns([8, 3])
            
            # Select conversation button
            with cols[0]:
                if st.button(
                    conv["title"],
                    key=f"sel_{cid}",
                    use_container_width=True,
                    type="primary" if cid == st.session_state.active_conversation_id else "secondary"
                ):
                    st.session_state.active_conversation_id = cid
                    st.rerun()
            
            # Delete conversation button (centered with minimal margin)
            with cols[1]:
                if st.button("🗑️", key=f"del_{cid}", help="Delete chat", use_container_width=True):
                    delete_conversation(cid)
                    st.rerun()
    else:
        st.caption("No conversations yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN CHAT AREA - Welcome Message & Sample Questions
# ═══════════════════════════════════════════════════════════════════════════════

st.title("Welcome to LuxStay")
st.markdown("Ask me about hotels, visa requirements, or recommendations.")

# --- Sample Questions Section ---
st.markdown("#### Try these questions:")
sample_questions = [
    "Find me a luxury hotel in Cairo with a pool and a rating above 9.0.",
    "Do travelers from the United States need a visa for France?",
    "What are the top 3 highest-rated hotels by Business travellers?",
    "Show me the reviews for 'The Azure Tower'."
]

# Create clickable buttons for each sample question
cols = st.columns(len(sample_questions))
for i, q in enumerate(sample_questions):
    if cols[i].button(q, key=f"q_btn_{i}", use_container_width=True):
        st.session_state.pre_selected_prompt = q  # Store selected prompt
        st.rerun()  # Trigger rerun to process the prompt

# ═══════════════════════════════════════════════════════════════════════════════
# 8. MESSAGE DISPLAY - Render Chat History with Typing Animation
# ═══════════════════════════════════════════════════════════════════════════════

messages = get_active_messages()

for i, msg in enumerate(messages):
    is_last = (i == len(messages) - 1)  # Check if this is the latest message
    role = msg["role"]  # 'user' or 'assistant'
    bubble_class = "bubble bubble-user" if role == "user" else "bubble bubble-assistant"

    with st.chat_message(role):
        container = st.container()
        
        # Typing animation for new assistant messages
        if role == "assistant" and msg.get("is_new") and is_last:
            placeholder = container.empty()
            shown = ""
            
            # Animate character by character
            for ch in msg["content"]:
                shown += ch
                placeholder.markdown(
                    f'<div class="{bubble_class}">{shown}</div>',
                    unsafe_allow_html=True
                )
                time.sleep(0.01)  # Delay between characters
            
            # Final render and mark as not new
            placeholder.markdown(
                f'<div class="{bubble_class}">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            msg["is_new"] = False
        else:
            # Regular display for existing messages
            container.markdown(
                f'<div class="{bubble_class}">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

        # --- Display RAG Context & Metrics for Assistant Messages ---
        if role == "assistant" and "context" in msg:
            with st.expander(f"🕵️ View RAG Context & Metrics (Latency: {msg['metrics']['latency']}s)"):
                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{msg['metrics']['latency']}s")
                m2.metric("Tokens", f"{msg['metrics']['tokens']}")
                m3.metric("Est. Cost", f"${msg['metrics']['cost']}")
                
                # Context tabs
                tab1, tab2, tab3 = st.tabs(["Graph Viz", "Cypher Query", "Raw Nodes"])
                
                with tab1:
                    st.graphviz_chart(msg["context"]["graph_viz"])
                
                with tab2:
                    st.code(msg["context"]["cypher"], language="cypher")
                    st.caption("Parameters: " + str(msg["context"]["entities"]))
                
                with tab3:
                    st.json(msg["context"]["nodes"])

# ═══════════════════════════════════════════════════════════════════════════════
# 9. CHAT INPUT HANDLER - Process User Input & Generate Response
# ═══════════════════════════════════════════════════════════════════════════════

# Get prompt from chat input or pre-selected sample question
prompt = st.chat_input(
    "Ex: Find me a luxury hotel in Cairo with a pool...",
    key=st.session_state.input_key  # Unique key to enable clearing after submission
)

# If a sample question button was clicked, use that prompt
if st.session_state.pre_selected_prompt:
    prompt = st.session_state.pre_selected_prompt
    st.session_state.pre_selected_prompt = ""  # Clear immediately

# Process the prompt if it exists (from typing or button click)
if prompt:
    # Get current configuration
    model = st.session_state.config["model"]
    strategy = st.session_state.config["strategy"]
    
    # Add user message to conversation
    add_message("user", prompt)

    # Generate assistant response
    with st.chat_message("assistant", avatar="🏨"):
        with st.spinner("Thinking (Consulting Knowledge Graph)..."):
            # Call RAG pipeline
            response_text, context_data, metrics = generate_response(prompt, model, strategy)
            st.markdown(response_text)

        # Add assistant message with context and metrics, marked for animation
        add_message(
            "assistant",
            response_text,
            context=context_data,
            metrics=metrics,
            is_new=True  # Flag to trigger typing animation on next render
        )

    # Increment input key to clear chat_input and trigger rerun
    st.session_state.input_key += 1
    st.rerun()