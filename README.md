# LuxStay AI - Graph RAG Hotel Recommendation System 🏨

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Neo4j-5.0+-green.svg" alt="Neo4j">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4-orange.svg" alt="OpenAI">
  <img src="https://img.shields.io/badge/Streamlit-UI-red.svg" alt="Streamlit">
</p>

## 💡 What is LuxStay AI?

**LuxStay AI** is a smart chatbot that helps you find the perfect hotel and check visa requirements for any country. Simply chat with it naturally like "Find me a 5-star hotel in Paris" or "Do I need a visa from USA to France?" and it understands what you need.

**How it works:**
- Uses a **Knowledge Graph** (Neo4j) to store hotel data, reviews, and visa information as connected nodes
- Employs **AI embeddings** to understand the meaning of your questions, not just keywords
- Leverages **GPT-4** to generate natural, conversational responses
- Combines structured data (database queries) with AI intelligence for accurate, context-aware recommendations

**In simple terms:** Traditional search engines match keywords, but LuxStay AI understands intent. It knows "cheap hotel near Eiffel Tower" means you want a budget-friendly hotel in Paris with proximity to landmarks, and finds exactly that by connecting the dots in its knowledge graph.

---

An intelligent hotel recommendation and visa information system built with **Graph Retrieval-Augmented Generation (GraphRAG)** technology. This project leverages Neo4j knowledge graphs, OpenAI embeddings, and advanced NLP techniques to provide personalized hotel recommendations and visa requirement information through an interactive chat interface.

## 🌟 Features

### Core Capabilities
- 🏨 **Smart Hotel Recommendations**: Get personalized hotel suggestions based on preferences (location, budget, stars, amenities)
- 🛂 **Visa Information**: Query visa requirements between countries with real-time information
- 💬 **Multi-turn Conversations**: Context-aware chat interface that maintains conversation history
- 📊 **Knowledge Graph Visualization**: Interactive graph visualizations of query results
- 🎯 **Intent Classification**: Automatic detection of user intent (hotel search, visa info, reviews, etc.)
- 🔍 **Entity Extraction**: Advanced NLP for extracting locations, preferences, and constraints from queries

### Retrieval Modes
1. **Baseline Mode**: Traditional Cypher query-based retrieval
2. **Embeddings Mode**: Semantic search using sentence transformers
3. **Hybrid Mode**: Combines both approaches for optimal results

### Performance Metrics
- ⚡ **Real-time Latency Tracking**: Monitor response times for each query
- 💰 **Cost Estimation**: Track OpenAI API token usage and costs
- 📈 **Result Quality**: Automatic deduplication and result merging

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      UI Layer (Streamlit)                   │
│                         app.py                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   Integration Layer                         │
│                   backend_helper.py                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   Backend Layer                             │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Intent     │  │   Entity     │  │   RAG Pipeline  │  │
│  │Classification│  │  Extraction  │  │                 │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              GraphRAGBackend                         │  │
│  │  • Cypher Queries (Baseline)                        │  │
│  │  • Embedding Search (Semantic)                      │  │
│  │  • Hybrid Retrieval                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────▼────────┐                 ┌────────▼─────────┐
│   Neo4j KG     │                 │   OpenAI API     │
│   Database     │                 │   (Embeddings    │
│                │                 │    & GPT-4)      │
└────────────────┘                 └──────────────────┘
```

## 📊 Knowledge Graph Schema

```
(Traveller)-[:FROM_COUNTRY]->(Country)
(Traveller)-[:REVIEWED]->(Hotel)
(Traveller)-[:WROTE]->(Review)
(Review)-[:ABOUT]->(Hotel)
(Hotel)-[:LOCATED_IN]->(City)
(City)-[:IN_COUNTRY]->(Country)
(Country)-[:VISA_REQUIRED {type, duration}]->(Country)
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Neo4j Database (local or cloud)
- OpenAI API Key

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/luxstay-ai.git
cd luxstay-ai
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv fire_env_tf
fire_env_tf\Scripts\activate

# macOS/Linux
python3 -m venv fire_env_tf
source fire_env_tf/bin/activate
```

### Step 3: Install All Dependencies
```bash
pip install -r requirements.txt
```

This single command installs everything you need:
- Web UI, graph database drivers, OpenAI integration
- Machine learning libraries, embeddings, NLP tools
- All utilities and dependencies

### Step 4: Set Up Neo4j Database

#### Option A: Neo4j Desktop (Recommended for Development)
1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and database
3. Start the database
4. Note your connection URI, username, and password

#### Option B: Neo4j AuraDB (Cloud)
1. Sign up for [Neo4j AuraDB](https://neo4j.com/cloud/aura/)
2. Create a free instance
3. Download connection credentials

### Step 5: Configure Environment Variables

Create a `.env` file in the `Milestone_3/` directory:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

Create another `.env` file in the `UI/` directory with the same content.

### Step 6: Load Data into Neo4j

#### Configure Database Connection
Update `Milestone_2/config.txt`:
```txt
URI=bolt://localhost:7687
USERNAME=neo4j
PASSWORD=your_password
```

#### Run the Knowledge Graph Creation Script
```bash
cd Milestone_2
python Create_kg.py
```

This will:
- Create nodes for Travellers, Hotels, Cities, Countries, Reviews
- Establish relationships between entities
- Load data from CSV files in the `archive/` folder

### Step 7: Generate Embeddings

```bash
cd ../Milestone_3/embeddings

# Generate feature embeddings f and Verify Setup

```bash
cd Milestone_3/embeddings
python feature_embedding.py
python visa_enrich.py
cd ..
python backend.py
```

**Important:** Update file paths in the embedding scripts if your data is in a different location.

This generates semantic embeddings for hotels and visa data, then verifies the backend is working
```bash
cd UI
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Example QueApplication

```bash
cd UI
streamlit run app.py
```

The app will automatically
**Visa Information:**
- "Do I need a visa from USA to France?"
- "What are the visa requirements for Egypt from UK?"
- "Tell me about tourist visa from India to Canada"

**Hotel Reviews:**
- "Show me reviews for the Ritz Paris"
- "What do people say about Holiday Inn in London?"

### UI Features

1. **Chat Interface**: Type natural language queries
2. **Sidebar Controls**:
   - Select retrieval mode (baseline/embeddings/hybrid)
   - Adjust number of results
   - View conversation history
3. **Results Display**:
   - Hotel cards with ratings and amenities
   - Visa requirement details
   - Knowledge graph visualizations
4. **Metrics Dashboard**:
   - Query latency
   - Token usage
   - Cost estimation

## 📁 Project Structure

```
ACL/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── FLOW_DIAGRAM.py                   # System architecture diagram
│
├── archive/                          # Raw data files
│   ├── hotels.csv                   # Hotel information
│   ├── reviews.csv                  # User reviews
│   ├── users.csv                    # Traveller profiles
│   └── visa.csv                     # Visa requirements
│
├── Milestone_2/                      # Knowledge Graph Setup
│   ├── Create_kg.py                 # KG creation script
│   ├── config.txt                   # Neo4j connection config
│   ├── queries.txt                  # Sample Cypher queries
│   └── rule.txt                     # Graph schema rules
│
├── Milestone_3/                      # Backend & RAG Pipeline
│   ├── backend.py                   # Main GraphRAG backend
│   ├── rag_pipeline.py              # RAG orchestration
│   ├── intent.py                    # Intent classification
│   ├── entities.py                  # Entity extraction
│   ├── model_eval.py                # Model evaluation
│   ├── .env                         # Environment variables
│   │
│   ├── cypher_queries/              # Query templates
│   │   ├── baseline.py              # Baseline Cypher queries
│   │   └── intent_entities_baseline_test.py
│   │
│   ├── embeddings/                  # Embedding modules
│   │   ├── embedding.py             # Embedding utilities
│   │   ├── feature_embedding.py     # Hotel feature embeddings
│   │   ├── node_embedding.py        # Node embeddings
│   │   ├── visa_enrich.py           # Visa embedding enrichment
│   │   └── feature_embeddings_models_comparison.py
│   │
│   └── Results/                     # Evaluation results
│       └── model_comparison.csv
│
└── UI/                               # Streamlit Web Interface
    ├── app.py                       # Main Streamlit app
    ├── backend_helper.py            # Backend integration
    ├── .env                         # UI environment variables
    └── UI_VISUAL_WORKFLOW.md        # UI documentation
```

## 🔧 Configuration

### Model Settings

**Embedding Model (in `embeddings/feature_embedding.py`):**
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**OpenAI Model (in `rag_pipeline.py`):**
```python
llm = ChatOpenAI(model="gpt-4", temperature=0)
```

### Neo4j Configuration

**Connection Parameters:**
- URI: `bolt://localhost:7687` (default local)
- Authentication: Username/password from config files

### Retrieval Parameters

**Default Settings:**
- Top-K Results: 5
- Similarity Threshold: 0.7
- Max Tokens: 4096

## 🧪 Testing

### Run Backend Tests
```bash
cd Milestone_3
python backend.py
```

### Test Individual Components
```bash
# Test embeddings
cd embeddings
python node_test.py
python visa_test.py

# Test Cypher queries
cd ../cypher_queries
python intent_entities_baseline_test.py
```

### Model Evaluation
```bash
cd Milestone_3
python model_eval.py
```

## 🛠️ Technologies Used

- **Backend**: Python 3.10+
- **Database**: Neo4j Graph Database
- **LLM**: OpenAI GPT-4
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Web Framework**: Streamlit
- **ML Libraries**: TensorFlow, Scikit-learn, SHAP, LIME
- **Data Processing**: Pandas, NumPy
- **Orchestration**: LangChain

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

