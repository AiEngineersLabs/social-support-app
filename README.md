# Social Support Application Workflow Automation

An AI-powered system that automates social support assessments for a government social security department. A **single AI chatbot** guides applicants through the entire process — from collecting information conversationally to uploading documents, running a 4-agent AI pipeline, and delivering a transparent decision — all in under 3 minutes.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Demo Walkthrough](#demo-walkthrough)
6. [Running Tests](#running-tests)
7. [Project Structure](#project-structure)
8. [Architecture Overview](#architecture-overview)
9. [Technology Stack](#technology-stack)
10. [API Endpoints](#api-endpoints)
11. [Langfuse Observability (Optional)](#langfuse-observability-optional)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Tool | Minimum Version | Installation |
|------|----------------|-------------|
| **Python** | 3.11+ | `brew install python@3.12` |
| **PostgreSQL** | 14+ | `brew install postgresql@14` |
| **Ollama** | 0.1.0+ | [ollama.com/download](https://ollama.com/download) |
| **Docker + Docker Compose** | — | [docker.com](https://docs.docker.com/get-docker/) (optional, for Langfuse) |
| **Homebrew** (macOS) | — | [brew.sh](https://brew.sh) |

### Verify installations

```bash
python3 --version        # 3.11+
psql --version           # 14+
ollama --version         # installed
docker --version         # optional
```

---

## Installation

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd social-support-app
```

### Step 2: Pull required Ollama models

```bash
# Main reasoning model (8B parameters)
ollama pull llama3

# Fast model for validation tasks (3B parameters)
ollama pull llama3.2

# Vision/OCR model for Emirates ID image extraction (7B parameters)
ollama pull llava:7b

# Embedding model for RAG policy retrieval
ollama pull nomic-embed-text
```

Verify:

```bash
ollama list
# Should show: llama3, llama3.2, llava:7b, nomic-embed-text
```

### Step 3: Create Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Set up PostgreSQL

```bash
brew services start postgresql@14
createdb social_support_db
psql -d social_support_db -c "SELECT 1"   # Verify
```

### Step 6: Generate synthetic training data

```bash
python -m app.utils.synthetic_data
```

This creates:
- `data/synthetic/training_data.json` — 500 training records for the ML model
- `data/synthetic/sample_applicants.json` — 5 sample applicants with documents
- Sample bank statements (CSV), resumes (TXT), assets (XLSX), credit reports (TXT)

### Step 7: Initialise database tables

```bash
python -c "from app.database import init_db; init_db(); print('Tables created')"
```

---

## Configuration

Copy and edit the `.env` file:

```env
# Database
DATABASE_URL=postgresql://<your-username>@localhost:5432/social_support_db

# Ollama models
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3
LIGHT_LLM_MODEL=llama3.2
VISION_MODEL=llava:7b
EMBEDDING_MODEL=nomic-embed-text

# Storage
CHROMA_PERSIST_DIR=./data/chroma_db
UPLOAD_DIR=./data/uploads
POLICY_DIR=./data/policies

# Langfuse observability (leave blank to disable)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000
```

Replace `<your-username>` with your macOS username (`whoami`).

---

## Running the Application

Open **three terminal windows**:

### Terminal 1 — Ollama

```bash
ollama serve
```

### Terminal 2 — FastAPI backend

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

On first start, the backend trains the ML classifier and ingests policy documents into ChromaDB.

### Terminal 3 — Streamlit chatbot

```bash
source venv/bin/activate
streamlit run frontend/streamlit_app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

---

## Demo Walkthrough

The application is a **single AI chatbot** — no forms or wizards. The chatbot guides applicants through 7 phases automatically:

### Phase 1: GREETING
The chatbot welcomes the applicant and explains the process.

### Phase 2: INTAKE (Conversational data collection)
The AI extracts structured fields from natural conversation:

> **You:** "Hi, my name is Fatima Hassan. I'm 28, divorced with 3 kids and I'm currently unemployed."
>
> **AI:** "Thank you Fatima. What is your monthly income in AED?"
>
> **You:** "About 800 AED from informal work."
>
> **AI:** "Understood. What is your Emirates ID number?"

The sidebar tracks collected fields in real time. Once all 14 required fields are collected, the chatbot moves automatically to CONFIRM.

**Required fields collected conversationally:**
`full_name`, `emirates_id`, `age`, `gender`, `nationality`, `marital_status`, `family_size`, `dependents`, `education_level`, `employment_status`, `monthly_income`, `total_assets`, `total_liabilities`, `years_of_experience`

### Phase 3: CONFIRM
The AI summarises all collected information and asks for confirmation.

> **AI:** "Here's what I've collected: Name: Fatima Hassan, Age: 28, Status: Unemployed, Income: AED 800... Is this correct?"
>
> **You:** "Yes, that's right."

### Phase 4: UPLOAD
A document upload panel appears in the sidebar. Upload any combination of:

| Document | Sample file |
|----------|------------|
| Bank Statement | `data/synthetic/bank_statement_*.csv` |
| Emirates ID (image) | `data/synthetic/eid_*.txt` |
| Resume / CV | `data/synthetic/resume_*.txt` |
| Assets & Liabilities | `data/synthetic/assets_*.xlsx` |
| Credit Report | `data/synthetic/credit_*.txt` |

Type "done" or "ready" when finished uploading. Documents are optional — the AI works with whatever is provided.

### Phase 5: ASSESS
The chatbot triggers the 4-agent pipeline and shows live progress:

```
⚡ Running Document Processing Agent...  ✓
⚡ Running Validation Agent...           ✓
⚡ Running Eligibility Assessment...     ✓
⚡ Running Enablement Recommender...     ✓
```

### Phase 6: RESULTS
A decision card appears in the chat:

```
✅ APPROVED — Tier 1: Emergency Support

Eligibility Score: 83/100
Income Need:       84/100
Employment:        90/100
Family Burden:     70/100
Wealth Deficit:    65/100
Demographic:       20/30

Recommended Programs:
• Emergency Financial Assistance (High Priority)
• Job Matching Service (High Priority)

AI Reasoning: Applicant shows high financial need...
```

An expandable **Agent Reasoning Trace** shows every Thought→Action→Observation step from each agent.

### Phase 7: QA (Open-ended chat)
Ask anything about the decision or available programs:

> "Why was I approved?"
> "What training programs are available in my area?"
> "How can I improve my eligibility score?"
> "Explain the job matching service."

### Testing Different Scenarios

**High-need (likely APPROVE — Tier 1):**
Unemployed, income AED 500, family size 6, high school education, low assets

**Medium-need (likely APPROVE — Tier 2):**
Part-time, income AED 4,000, family size 3, diploma education

**Low-need (likely SOFT DECLINE):**
Employed, income AED 20,000, family size 2, master's degree, high assets

Click **"New Application"** in the sidebar to reset the chatbot.

---

## Running Tests

```bash
source venv/bin/activate

# Run all tests (no Ollama required — all LLM calls mocked)
pytest

# Verbose output
pytest -v

# Single test file
pytest tests/test_agents.py -v
pytest tests/test_ml_classifier.py -v
pytest tests/test_document_processor.py -v
pytest tests/test_api.py -v

# With coverage
pytest --cov=app --cov-report=term-missing
```

Tests use `monkeypatch` to mock all LLM calls — the full test suite runs in < 10 seconds without a running Ollama instance.

---

## Project Structure

```
social-support-app/
├── app/
│   ├── config.py                   # Pydantic Settings (env vars)
│   ├── database.py                 # PostgreSQL session management
│   ├── models.py                   # SQLAlchemy ORM models
│   ├── schemas.py                  # Pydantic request/response schemas
│   ├── main.py                     # FastAPI app + all endpoints
│   ├── agents/
│   │   ├── orchestrator.py         # LangGraph StateGraph pipeline
│   │   ├── document_agent.py       # ReAct document extraction
│   │   ├── validation_agent.py     # Reflexion 2-pass validation
│   │   ├── eligibility_agent.py    # ReAct + ML scoring
│   │   └── enablement_agent.py     # ReAct + RAG recommendations
│   ├── services/
│   │   ├── llm_service.py          # Ollama + Langfuse integration
│   │   ├── document_processor.py   # PDF/CSV/XLSX/image extraction
│   │   ├── ml_classifier.py        # GradientBoosting classifier
│   │   └── vector_store.py         # ChromaDB + LlamaIndex RAG
│   └── utils/
│       └── synthetic_data.py       # Training data generator
├── data/
│   ├── policies/                   # Policy docs for RAG
│   ├── synthetic/                  # Generated training data
│   ├── uploads/                    # Runtime document uploads
│   └── chroma_db/                  # ChromaDB vector store
├── docs/
│   └── solution_document.md        # 13-section technical solution document
├── frontend/
│   └── streamlit_app.py            # Single AI chatbot UI
├── tests/
│   ├── test_api.py                 # API endpoint tests
│   ├── test_agents.py              # Agent pipeline tests (ReAct/Reflexion)
│   ├── test_document_processor.py  # Document extraction tests
│   └── test_ml_classifier.py       # ML model tests
├── docker-compose.yml              # Langfuse self-hosted observability
├── pytest.ini                      # pytest configuration
├── requirements.txt                # Python dependencies
└── .env                            # Environment variables
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit AI Chatbot (Single Page)                     │
│  GREETING → INTAKE → CONFIRM → UPLOAD → ASSESS →        │
│  RESULTS → QA                                           │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP REST
┌──────────────────────────▼──────────────────────────────┐
│  FastAPI Backend                                        │
│  /api/chat-intake  /api/submit-application             │
│  /api/assess  /api/decision  /api/chat                 │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  LangGraph Orchestrator (StateGraph)                    │
│  document_node → validation_node → eligibility_node    │
│  → enablement_node → final_decision_node               │
└──────┬────────────────┬──────────────┬──────────────────┘
       │                │              │
  ┌────▼────┐      ┌────▼────┐    ┌───▼─────┐
  │ Agents  │      │ Ollama  │    │  Data   │
  │ (ReAct/ │      │ llama3  │    │ Postgres│
  │ Reflex) │      │ llava:7b│    │ ChromaDB│
  └─────────┘      └─────────┘    └─────────┘
```

### Agent Pipeline

| # | Agent | Reasoning | Responsibility |
|---|-------|-----------|---------------|
| 1 | Document Agent | ReAct (T→A→O) | Extract structured data from all uploaded documents |
| 2 | Validation Agent | Reflexion (2-pass) | Cross-check data for inconsistencies; flag discrepancies |
| 3 | Eligibility Agent | ReAct + ML | Score applicant using GradientBoosting + LLM reasoning |
| 4 | Enablement Agent | ReAct + RAG | Recommend programs from policy vector store |

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit | Single-page AI chatbot |
| API | FastAPI | Async REST backend |
| Orchestration | LangGraph | Stateful multi-agent workflow |
| Agent Framework | LangChain | LLM integration and tooling |
| Reasoning | ReAct + Reflexion | Agent reasoning frameworks |
| LLM Hosting | Ollama | 100% local model serving |
| Main LLM | llama3 (8B) | Reasoning, eligibility, enablement |
| Fast LLM | llama3.2 (3B) | Validation, intake extraction |
| Vision LLM | llava:7b | Emirates ID image OCR |
| Embeddings | nomic-embed-text | RAG vector embeddings |
| ML Model | Scikit-learn GradientBoosting | Eligibility classification |
| Vector Store | ChromaDB | Policy document retrieval |
| RAG | LlamaIndex | Policy ingestion and retrieval |
| Database | PostgreSQL 14 | Applicant and decision storage |
| PDF Parsing | pdfplumber + pymupdf | Multi-format PDF extraction |
| OCR Fallback | pytesseract | CPU-based text OCR |
| Observability | Langfuse (Docker) | LLM call tracing and monitoring |
| Testing | pytest | Unit + integration + API tests |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/chat-intake` | LLM field extraction from natural language message |
| POST | `/api/submit-application` | Register new applicant with collected fields |
| POST | `/api/upload-document/{id}` | Upload a document file |
| POST | `/api/assess/{id}` | Run the 4-agent AI pipeline |
| GET | `/api/decision/{id}` | Retrieve decision with full agent traces |
| POST | `/api/chat` | Open-ended QA chat with applicant context |
| POST | `/api/reassess/{id}` | Clear existing decision for re-assessment |
| GET | `/api/applicants` | List all applicants |
| DELETE | `/api/applicant/{id}` | Delete applicant and all related data |

**Interactive API docs:** http://localhost:8000/docs

### Example: Chat intake

```bash
curl -X POST http://localhost:8000/api/chat-intake \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My name is Ahmed, I am 32 years old, unemployed with a family of 5",
    "collected_fields": {}
  }'
```

### Example: Run assessment

```bash
curl -X POST http://localhost:8000/api/assess/1
```

---

## Langfuse Observability (Optional)

Langfuse is a self-hosted, open-source LLM observability platform. It traces every LLM call made during an assessment.

### Start Langfuse

```bash
docker-compose up -d
```

### Set up account

1. Open http://localhost:3000
2. Register a local account (no email verification required)
3. Create a project and copy the API keys
4. Add to `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

5. Restart the FastAPI backend

### What you'll see

Every assessment run creates traces showing:
- Prompt sent to each LLM call
- Response received
- Latency per call
- Token usage
- Grouped by agent (document, validation, eligibility, enablement)

The chatbot sidebar links directly to the Langfuse dashboard.

**Note:** Langfuse is entirely optional. The system works identically with empty keys — only the trace dashboard is unavailable.

---

## Troubleshooting

### Ollama not running

```
ConnectionError: Cannot connect to http://localhost:11434
```

Fix:
```bash
ollama serve
```

### Model not found

```
Error: model 'llava:7b' not found
```

Fix:
```bash
ollama pull llava:7b
```

### PostgreSQL not running

```
Error: connection refused on port 5432
```

Fix:
```bash
brew services start postgresql@14
```

### Assessment returns "already assessed"

To re-assess an applicant:
```bash
curl -X POST http://localhost:8000/api/reassess/1
```

### Slow LLM responses

- Use a smaller main model: set `LLM_MODEL=llama3.2` in `.env`
- Ensure Ollama has GPU access (Metal on macOS M-series, CUDA on NVIDIA)
- Check memory: `ollama ps` shows currently loaded models

### Port conflicts

```bash
# Kill process on port 8000
lsof -ti :8000 | xargs kill -9

# Kill process on port 8501
lsof -ti :8501 | xargs kill -9
```

### Langfuse not connecting

If Langfuse Docker is not running and keys are set, you'll see a warning in logs:
```
WARNING - Langfuse callback unavailable: connection refused
```

The system continues normally. Start Docker if you want traces:
```bash
docker-compose up -d
```

---

## Further Reading

See [docs/solution_document.md](docs/solution_document.md) for the full technical solution document including:
- Detailed architecture diagrams (Mermaid)
- Technology stack justification
- ReAct and Reflexion reasoning framework design
- ML algorithm selection and feature engineering
- Security considerations and production deployment guide
