# Social Support Application Workflow Automation
## Technical Solution Document

**Version:** 2.0
**Date:** February 2026
**Classification:** Internal Technical Reference

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Architecture Diagram (Mermaid)](#3-architecture-diagram)
4. [Technology Stack Justification](#4-technology-stack-justification)
5. [AI Reasoning Frameworks](#5-ai-reasoning-frameworks)
6. [Multi-Agent Pipeline Design](#6-multi-agent-pipeline-design)
7. [Multimodal Document Processing](#7-multimodal-document-processing)
8. [ML Classification Algorithm](#8-ml-classification-algorithm)
9. [Observability & Langfuse Integration](#9-observability--langfuse-integration)
10. [API Design & Integration Guide](#10-api-design--integration-guide)
11. [Test Strategy](#11-test-strategy)
12. [Deployment Guide](#12-deployment-guide)
13. [Security Considerations](#13-security-considerations)

---

## 1. Executive Summary

The Social Support Application Workflow Automation system replaces a manual 5–20 working-day government social security process with a fully automated, AI-driven pipeline that delivers decisions in **under 3 minutes**. Built entirely on locally-hosted open-source technology, the system processes text, tabular, and image documents, reasons through applicant data using multiple AI agents, and delivers a transparent recommendation with full traceability.

### Key Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Processing time | 5–20 working days | < 3 minutes |
| Document handling | Manual officer review | Automated multimodal AI extraction |
| Decision consistency | Variable (officer-dependent) | 100% rule-based + ML deterministic |
| Auditability | Paper trail only | Full agent reasoning trace per decision |
| Privacy | Documents leave organisation | 100% on-premise — no cloud APIs |
| Cost | Staff hours per application | Near-zero marginal cost at scale |

### Core Design Principles

1. **Privacy-first**: All LLM inference runs locally via Ollama — no applicant data leaves the server
2. **Transparent reasoning**: Every agent decision is backed by a traceable Thought→Action→Observation log
3. **Graceful degradation**: If a document is missing or OCR fails, the pipeline continues with available data
4. **Conversational UX**: A single AI chatbot guides applicants through the entire process — no forms

---

## 2. System Architecture Overview

The system is organised into five horizontal layers:

```
┌─────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                     │
│  Streamlit AI Chatbot (Single Page)                     │
│  Phase: GREETING → INTAKE → CONFIRM → UPLOAD →         │
│         ASSESS → RESULTS → QA                           │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP REST (FastAPI)
┌────────────────────▼────────────────────────────────────┐
│  API LAYER (FastAPI)                                    │
│  /api/health  /api/submit-application                  │
│  /api/chat-intake  /api/upload-document                │
│  /api/assess  /api/decision  /api/chat                 │
│  /api/reassess  /api/applicants                        │
└────────────────────┬────────────────────────────────────┘
                     │ Python function calls
┌────────────────────▼────────────────────────────────────┐
│  ORCHESTRATION LAYER (LangGraph StateGraph)             │
│                                                         │
│  document_node → validation_node → eligibility_node    │
│       → enablement_node → final_decision_node          │
│                                                         │
│  WorkflowState carries:  applicant_data, documents,    │
│  validation_result, eligibility_result,                 │
│  recommendations, final_decision, react_traces          │
└──────┬────────────────┬──────────────┬──────────────────┘
       │                │              │
┌──────▼──────┐  ┌──────▼──────┐  ┌───▼─────────┐
│  AGENT LAYER│  │ LLM LAYER   │  │ DATA LAYER  │
│             │  │             │  │             │
│ Document    │  │ Ollama      │  │ PostgreSQL  │
│ Agent(ReAct)│  │ llama3      │  │ Applicants  │
│             │  │ llama3.2    │  │ Documents   │
│ Validation  │  │ llava:7b    │  │ Decisions   │
│ Agent(Refl) │  │ nomic-embed │  │             │
│             │  │             │  │ ChromaDB    │
│ Eligibility │  │ Langfuse    │  │ Policies    │
│ Agent(ReAct)│  │ (observ.)   │  │ (RAG)       │
│             │  │             │  │             │
│ Enablement  │  │ ML Model    │  │ File System │
│ Agent(ReAct)│  │ GradBoost   │  │ Uploads     │
└─────────────┘  └─────────────┘  └─────────────┘
```

### Data Flow Summary

1. **Applicant opens chatbot** → Greeted by AI; conversation begins in natural language
2. **INTAKE phase** → `/api/chat-intake` extracts structured fields via LLM from each message
3. **CONFIRM phase** → AI summarises collected fields; applicant confirms or corrects
4. **UPLOAD phase** → Applicant drags-and-drops documents; each is stored and typed
5. **ASSESS phase** → LangGraph pipeline executes all 4 agents; results streamed to UI
6. **RESULTS phase** → Decision card rendered with scores, programs, reasoning traces
7. **QA phase** → Open-ended chat with LLM in context of applicant's profile and decision

---

## 3. Architecture Diagram

```mermaid
flowchart TB
    subgraph UI["🖥️ Streamlit Chatbot UI"]
        direction TB
        A[GREETING] --> B[INTAKE\nLLM field extraction]
        B --> C[CONFIRM\nSummary review]
        C --> D[UPLOAD\nDocument upload]
        D --> E[ASSESS\nAgent pipeline]
        E --> F[RESULTS\nDecision card]
        F --> G[QA\nOpen chat]
    end

    subgraph API["⚡ FastAPI Backend"]
        H[/api/chat-intake/]
        I[/api/submit-application/]
        J[/api/upload-document/]
        K[/api/assess/]
        L[/api/decision/]
        M[/api/chat/]
    end

    subgraph Orchestrator["🔀 LangGraph Orchestrator"]
        N[document_node] --> O[validation_node]
        O --> P{flags >= 2?}
        P -- No --> Q[eligibility_node]
        P -- Yes --> R[MANUAL_REVIEW]
        Q --> S[enablement_node]
        S --> T[final_decision_node]
    end

    subgraph Agents["🤖 Agents"]
        U["📄 Document Agent\n(ReAct: T→A→O)"]
        V["✅ Validation Agent\n(Reflexion: P1→critique→P2)"]
        W["📊 Eligibility Agent\n(ReAct: ML + LLM)"]
        X["🎯 Enablement Agent\n(ReAct + RAG)"]
    end

    subgraph LLM["🧠 LLM Layer (Ollama)"]
        Y[llama3\nmain reasoning]
        Z[llama3.2\nfast/light tasks]
        AA[llava:7b\nvision OCR]
        BB[nomic-embed-text\nembeddings]
    end

    subgraph Data["💾 Data Layer"]
        CC[(PostgreSQL\nApplicants/Docs/Decisions)]
        DD[(ChromaDB\nPolicy RAG)]
        EE[File System\nUploads]
    end

    subgraph Observability["📈 Langfuse\n(Self-hosted Docker)"]
        FF[Trace Dashboard]
        GG[LLM Call Logs]
    end

    B -->|POST| H
    C -->|POST| I
    D -->|POST| J
    E -->|POST| K
    F -->|GET| L
    G -->|POST| M

    K --> Orchestrator
    N --> U
    O --> V
    Q --> W
    S --> X

    U --> Y
    U --> Z
    U --> AA
    V --> Z
    W --> Y
    W --> Agents
    X --> Y
    X --> DD
    BB --> DD

    U --> CC
    W --> CC
    T --> CC
    J --> EE

    Y --> FF
    Z --> FF
    AA --> FF
```

---

## 4. Technology Stack Justification

### 4.1 Ollama (Local LLM Hosting)

**Choice:** Ollama over cloud APIs (OpenAI, Anthropic, Gemini)

| Criterion | Cloud APIs | Ollama (local) |
|-----------|-----------|----------------|
| Data privacy | PII sent to cloud | 100% on-premise |
| Latency | Network round-trip | Sub-second on GPU |
| Cost at scale | Per-token billing | Hardware fixed cost |
| Regulatory compliance | Requires data processing agreements | No external data transfer |
| Offline operation | Fails without internet | Works airgapped |

**Models chosen:**

| Model | Size | Role | Reason |
|-------|------|------|--------|
| `llama3` | 8B | Main reasoning, eligibility, enablement | Best reasoning quality at local size |
| `llama3.2` | 3B | Fast validation tasks | 3× faster, sufficient for structured tasks |
| `llava:7b` | 7B | Vision OCR (Emirates ID images) | Only widely-available open-source multimodal |
| `nomic-embed-text` | 137M | RAG embeddings | Highest quality open-source embedding model |

### 4.2 LangGraph (Agent Orchestration)

**Choice:** LangGraph over bare Python loops or CrewAI

- **StateGraph pattern**: Each node receives and returns a typed `WorkflowState` dict, making inter-agent data flow explicit and inspectable
- **Conditional edges**: The validation node can route to `MANUAL_REVIEW` if critical flags are raised, without modifying agent code
- **Persistence-ready**: LangGraph's checkpoint mechanism enables resuming failed workflows
- **Debuggability**: The full state is visible at every graph step — crucial for a government system requiring auditability

### 4.3 FastAPI (API Framework)

**Choice:** FastAPI over Flask or Django REST

- **Async support**: `async def` handlers allow concurrent document uploads without blocking
- **Auto-validation**: Pydantic schemas validate all inputs; malformed requests return 422 immediately
- **Auto-docs**: Swagger UI at `/docs` provides a free interactive API explorer for testing
- **Performance**: ASGI with uvicorn handles 10,000+ req/s — far exceeds government portal requirements

### 4.4 ChromaDB + LlamaIndex (RAG)

**Choice:** ChromaDB over Pinecone/Weaviate; LlamaIndex over raw LangChain retrievers

- **ChromaDB**: Embedded (in-process), no separate service needed, persists to disk
- **LlamaIndex**: Policy documents are chunked, embedded, and stored once on startup; retrieval is sub-50ms
- **Domain impact**: The enablement agent retrieves relevant policy text before generating recommendations, grounding responses in actual government programs rather than hallucinating them

### 4.5 Scikit-learn GradientBoostingClassifier (ML Model)

See Section 8 for detailed ML justification.

### 4.6 pdfplumber + pymupdf (PDF Parsing)

| Library | Role | Reason |
|---------|------|--------|
| `pdfplumber` | Primary PDF text extraction | Best table/layout-aware extraction; preserves column structure in bank statements |
| `pymupdf` (fitz) | Fallback PDF extraction | Faster, handles encrypted/malformed PDFs better |

### 4.7 Langfuse (Observability)

**Choice:** Self-hosted Langfuse over LangSmith or bare logging

- **LLM-native traces**: Captures prompt text, completion, latency, and token counts per call
- **Self-hosted**: No data leaves the network — compatible with government data policies
- **Free tier**: Open-source Docker image; no licensing cost
- **LangChain integration**: Drop-in `CallbackHandler` with zero agent code changes

---

## 5. AI Reasoning Frameworks

### 5.1 ReAct (Reason + Act)

**Applied to:** Document Agent, Eligibility Agent, Enablement Agent

ReAct interleaves **Thought** (what should I do next?) with **Action** (execute a tool) and **Observation** (what did the tool return?) in a loop until the task is complete.

**Example — Document Agent processing 3 documents:**

```
[Thought]  I need to process 3 documents: bank_statement, resume, emirates_id.
           Starting with bank_statement (highest financial relevance).

[Action]   Tool: extract_bank_statement
           Input: {"file_path": "uploads/bank_statement_456.csv"}

[Observation] Tool: extract_bank_statement | Status: success
              Result: {"estimated_monthly_income": 3200, "average_monthly_balance": 8450,
                       "salary_detected": true}

[Thought]  Bank statement processed. Income: AED 3200. Now processing resume.

[Action]   Tool: extract_resume
           Input: {"file_path": "uploads/resume_456.pdf"}

[Observation] Tool: extract_resume | Status: success
              Result: {"years_of_experience": 4.5, "highest_education": "Diploma",
                       "last_job_title": "Customer Service"}

[Thought]  Resume processed. Now processing Emirates ID (image — routing to vision LLM).

[Action]   Tool: extract_emirates_id
           Input: {"file_path": "uploads/eid_456.jpg"}

[Observation] Tool: extract_emirates_id | Status: success
              Result: {"id_number": "784-1990-1234567-8", "name": "Ahmed Al-Rashid",
                       "nationality": "UAE"}

[Thought]  All 3 documents processed. Compilation complete.
```

**Why ReAct over a simple loop:** The trace is stored per-agent in `WorkflowState.react_traces["document_agent"]`, giving auditors a complete record of what data was extracted from which document. This is required for government decision transparency.

### 5.2 Reflexion

**Applied to:** Validation Agent exclusively

Reflexion adds a **self-critique** step after an initial pass: the agent evaluates its own output, identifies gaps, and re-runs targeted checks.

**Two-pass validation flow:**

```
PASS 1: Run 5 deterministic rule checks
  ├─ check_identity_consistency   → Emirates ID regex + age match
  ├─ check_income_consistency     → Declared income vs bank statement (>30% delta = flag)
  ├─ check_employment_consistency → Declared status vs resume history
  ├─ check_wealth_consistency     → Total assets vs liabilities ratio
  └─ check_credit_standing        → Outstanding debt vs income ratio

REFLEXION: LLM critique of Pass 1
  Prompt: "You validated this applicant. What checks might you have missed?
           Are the flags justified? Are any warnings too harsh or too lenient?"

  Example critique: "The income discrepancy flag is correct. However, I did not
  account for informal income sources common in this demographic. The credit
  standing check should also consider whether debts are secured vs unsecured."

PASS 2: Gap-filling based on critique
  ├─ If critique mentions missed checks → run additional contextual checks
  └─ If critique suggests false positive → downgrade flag to warning
```

**Why Reflexion for validation (not eligibility/enablement):** Validation errors in a government context have high consequence — a false positive flag can unfairly block an application. The self-critique loop catches over-flagging and under-flagging that pure rules miss.

---

## 6. Multi-Agent Pipeline Design

### 6.1 LangGraph State Definition

```python
class WorkflowState(TypedDict):
    applicant_data: dict          # Submitted application fields
    documents: list               # Uploaded document metadata
    extracted_doc_data: dict      # Parsed data from all documents
    validation_result: dict       # Flags, warnings, is_valid
    eligibility_result: dict      # Score, recommendation, support tier
    recommendations: dict         # Programs, career pathway, actions
    final_decision: dict          # Final merged decision
    react_traces: dict            # Per-agent trace: {"document_agent": [...], ...}
    error: str                    # Any pipeline-level error
```

### 6.2 Agent Responsibilities

| Agent | Input | Output | Reasoning |
|-------|-------|--------|-----------|
| **Document Agent** | Raw file paths + types | Structured extracted data per doc type | ReAct loop |
| **Validation Agent** | Applicant fields + extracted data | Flags list, warnings, is_valid, reflexion_critique | Reflexion (2-pass) |
| **Eligibility Agent** | Applicant fields | eligibility_score, recommendation, support_tier, reasoning | ReAct + ML |
| **Enablement Agent** | Applicant + eligibility result | recommended_programs, career_pathway, immediate_actions | ReAct + RAG |

### 6.3 Routing Logic

```
document_node
    ↓
validation_node
    ↓
 has critical flags (≥2)?
    ├── YES → MANUAL_REVIEW (human officer assigned)
    └── NO  → eligibility_node
                  ↓
              enablement_node
                  ↓
              final_decision_node
```

The conditional routing ensures applications with data integrity issues are escalated before the ML model runs, preventing corrupted data from influencing the eligibility score.

### 6.4 Support Tier Thresholds

| Score Range | Recommendation | Tier |
|-------------|---------------|------|
| ≥ 75 | APPROVE | Tier 1 — Emergency Support |
| 55–74 | APPROVE | Tier 2 — Transitional Support |
| 40–54 | SOFT_DECLINE | Tier 3 — Preventive Support |
| < 40 | SOFT_DECLINE | Tier 4 — Self-Sufficiency Programs |
| Any (flags ≥ 2) | MANUAL_REVIEW | Human review required |

---

## 7. Multimodal Document Processing

### 7.1 Document Type Routing

| Document Type | Format(s) | Extraction Method |
|--------------|-----------|------------------|
| Bank Statement | CSV, XLSX, PDF | Pandas → LLM analysis |
| Emirates ID | JPEG, PNG, PDF | llava:7b Vision OCR → pytesseract fallback |
| Resume / CV | PDF, TXT, DOCX | pdfplumber → LLM extraction |
| Assets & Liabilities | XLSX, CSV | Pandas/openpyxl → LLM analysis |
| Credit Report | PDF, TXT | pdfplumber → LLM extraction |

### 7.2 PDF Extraction Pipeline

```python
def _extract_text_from_pdf(file_path: str) -> str:
    # Primary: pdfplumber (table-aware, layout-preserving)
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()   # Respects column layout
                tables = page.extract_tables() # Extracts tabular data
        return text
    except Exception:
        pass

    # Fallback: pymupdf (handles encrypted/damaged PDFs)
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception:
        return ""  # Graceful degradation — pipeline continues
```

### 7.3 Vision OCR Pipeline (Emirates ID)

```python
def invoke_llm_with_image(prompt: str, image_path: str) -> str:
    # Stage 1: llava:7b via Ollama (primary)
    try:
        vision_llm = ChatOllama(model="llava:7b")
        image_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ])
        return vision_llm.invoke([message]).content
    except Exception:
        pass

    # Stage 2: pytesseract (CPU fallback — no GPU required)
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        ocr_text = pytesseract.image_to_string(img)
        # Feed OCR text to text LLM for structured extraction
        return invoke_llm(f"{prompt}\n\nOCR Text:\n{ocr_text}")
    except Exception:
        pass

    # Stage 3: Graceful placeholder (pipeline does not crash)
    return '{"error": "Vision extraction unavailable", "id_number": "", "name": ""}'
```

**Three-stage fallback ensures the pipeline never crashes on a single document failure.**

---

## 8. ML Classification Algorithm

### 8.1 Algorithm Choice: GradientBoostingClassifier

**Why Gradient Boosting over alternatives:**

| Algorithm | Why Not Chosen |
|-----------|---------------|
| Logistic Regression | Too linear — misses interaction between income + family size |
| Random Forest | Lower accuracy on tabular data with mixed ordinal/categorical features |
| Neural Network | Requires far more training data; not interpretable for government decisions |
| SVM | Poor scaling to high-dimensional feature spaces; slow inference |
| **GradientBoosting** ✓ | Best tabular accuracy, fast inference, feature importances for explainability |

### 8.2 Feature Engineering (14 Features)

```python
def _build_features(applicant: dict) -> np.ndarray:
    employment_map = {"Unemployed": 1, "Part-time": 2, "Self-employed": 3,
                      "Employed": 4, "Retired": 5}
    education_map  = {"None": 1, "Primary": 2, "High School": 3,
                      "Diploma": 4, "Bachelor": 5, "Master": 6, "PhD": 7}
    marital_map    = {"Single": 1, "Married": 2, "Divorced": 3, "Widowed": 4}

    features = [
        employment_map.get(employment_status, 1),   # Ordinal employment
        monthly_income / 1000,                       # Normalised income (AED '000s)
        family_size,                                 # Raw count
        dependents,                                  # Raw count
        age,                                         # Raw age
        gender_encoded,                              # Binary (0/1)
        marital_map.get(marital_status, 1),          # Ordinal marital
        education_map.get(education_level, 1),       # Ordinal education
        total_assets / 10000,                        # Normalised assets
        total_liabilities / 10000,                   # Normalised liabilities
        years_of_experience,                         # Raw years
        income_score / 100,                          # Derived: income need score
        family_score / 100,                          # Derived: family burden score
        wealth_score / 100,                          # Derived: wealth deficit score
    ]
```

### 8.3 Component Scoring Logic

Before ML inference, five interpretable component scores are computed deterministically:

| Score | Formula | Range |
|-------|---------|-------|
| `income_score` | `max(0, 100 - (income / 5000) * 100)` | 0–100 |
| `employment_score` | Unemployed=90, Part-time=60, Self-employed=45, Employed=20, Retired=30 | Fixed map |
| `family_score` | `min(100, (family_size - 1) * 15 + dependents * 10)` | 0–100 |
| `wealth_score` | `max(0, 100 - (assets - liabilities) / 1000)` | 0–100 |
| `demographic_score` | Age 18-30=+10, Female=+5, Divorced/Widowed=+5, Low edu=+10 | 0–30 |
| `eligibility_score` | `income×0.30 + employment×0.25 + family×0.20 + wealth×0.15 + demographic×0.10` | 0–100 |

These component scores feed the ML model as features **and** are returned to the UI as the score breakdown chart.

### 8.4 Training Data

- 500 synthetic records generated by `app/utils/synthetic_data.py`
- Balanced: ~60% eligible (True), ~40% not eligible (False)
- Stratified by employment status, income bands, and family size distributions
- Model is retrained on each application startup with the current dataset

### 8.5 Model Outputs

```python
{
    "eligibility_score": 78.4,      # 0-100 need score
    "recommendation": "APPROVE",    # APPROVE / SOFT_DECLINE
    "income_score": 84.0,           # Component score
    "employment_score": 90.0,       # Component score
    "family_score": 70.0,           # Component score
    "wealth_score": 65.0,           # Component score
    "demographic_score": 20.0,      # Component score
    "confidence": 0.91,             # Classifier probability
    "support_tier": "Tier 1 — Emergency Support"
}
```

---

## 9. Observability & Langfuse Integration

### 9.1 Architecture

Langfuse is self-hosted via Docker Compose alongside the application. It receives LLM call traces via LangChain's `CallbackHandler` mechanism — no code changes required in agent logic.

```yaml
# docker-compose.yml
services:
  langfuse-server:
    image: langfuse/langfuse:2
    ports: ["3000:3000"]
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse-db:5432/langfuse
      NEXTAUTH_SECRET: <generated-secret>
      NEXTAUTH_URL: http://localhost:3000

  langfuse-db:
    image: postgres:16-alpine
    ports: ["5433:5432"]   # Avoids conflict with app DB on 5432
```

### 9.2 What is Traced

| Event | Trace Name | Data Captured |
|-------|-----------|---------------|
| Document extraction | `document_extraction_{type}` | Prompt, response, latency |
| Validation critique | `validation_critique` | Prompt, LLM critique text |
| Eligibility reasoning | `eligibility_reasoning` | Prompt, recommendation text |
| Enablement RAG | `enablement_rag_{tier}` | RAG query, retrieved chunks, programs |
| Chat-intake extraction | `chat_intake_extraction` | User message, extracted JSON |
| QA chat | `qa_chat_{id}` | Full conversation turn |

### 9.3 Langfuse Dashboard Setup

```bash
# Start Langfuse
docker-compose up -d

# Open browser
open http://localhost:3000

# Register account (local — no email verification needed)
# Copy API keys to .env:
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

### 9.4 Graceful Degradation

If Langfuse is not configured (empty keys), `get_langfuse_callback()` returns `None` and `_build_config()` passes an empty callback list. All LLM calls proceed normally — zero performance impact.

---

## 10. API Design & Integration Guide

### 10.1 Endpoint Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/health` | None | System health check |
| POST | `/api/submit-application` | None | Register new applicant |
| POST | `/api/chat-intake` | None | LLM-driven field extraction from natural language |
| POST | `/api/upload-document/{applicant_id}` | None | Upload a document file |
| POST | `/api/assess/{applicant_id}` | None | Run the 4-agent pipeline |
| GET | `/api/decision/{applicant_id}` | None | Retrieve stored decision |
| POST | `/api/chat` | None | Open-ended chat with applicant context |
| POST | `/api/reassess/{applicant_id}` | None | Clear decision, allow re-assessment |
| GET | `/api/applicants` | None | List all applicants |
| DELETE | `/api/applicant/{applicant_id}` | None | Delete applicant and all data |

### 10.2 Chat Intake Protocol

The `/api/chat-intake` endpoint is the core of the conversational UI. It enables natural language data collection without a traditional form.

**Request:**
```json
{
  "message": "My name is Fatima Hassan, I'm 28 years old and divorced with 3 children",
  "collected_fields": {
    "full_name": "Fatima Hassan"
  },
  "conversation_history": [
    {"role": "assistant", "content": "Hello! What is your full name?"},
    {"role": "user", "content": "My name is Fatima Hassan"}
  ]
}
```

**Response:**
```json
{
  "extracted_fields": {
    "full_name": "Fatima Hassan",
    "age": 28,
    "marital_status": "Divorced",
    "dependents": 3
  },
  "next_question": "Thank you Fatima. What is your current employment status? (Employed, Unemployed, Part-time, Self-employed, or Retired)",
  "is_complete": false,
  "missing_fields": ["emirates_id", "gender", "nationality", "family_size",
                     "education_level", "employment_status", "monthly_income",
                     "total_assets", "total_liabilities", "years_of_experience"]
}
```

**Required fields for completion:**
`full_name`, `emirates_id`, `age`, `gender`, `nationality`, `marital_status`, `family_size`, `dependents`, `education_level`, `employment_status`, `monthly_income`, `total_assets`, `total_liabilities`, `years_of_experience`

### 10.3 Assessment Pipeline Response

```json
{
  "applicant_id": 42,
  "final_decision": {
    "recommendation": "APPROVE",
    "support_tier": "Tier 1 — Emergency Support",
    "eligibility_score": 83.2,
    "confidence": 0.94,
    "reasoning": "Applicant demonstrates high financial need (income AED 800, family of 4), unemployment status, and no significant assets. Scores highly on income, employment, and family criteria."
  },
  "validation_summary": {
    "is_valid": true,
    "flags": [],
    "warnings": ["Declared income (AED 800) is lower than bank statement average (AED 950). Minor discrepancy."],
    "reflexion_critique": "Validation is comprehensive. No additional flags identified."
  },
  "recommendations": {
    "recommended_programs": [
      {
        "program_name": "Emergency Financial Assistance",
        "priority": "high",
        "reason": "Immediate income support needed for family of 4"
      },
      {
        "program_name": "Job Matching Service",
        "priority": "high",
        "reason": "Unemployed with 2 years experience — job placement viable"
      }
    ],
    "career_pathway": "Customer service / retail sector (matching work history)",
    "immediate_actions": ["Register at nearest employment office", "Apply for emergency rent subsidy"],
    "long_term_plan": "Secure stable employment within 3 months; reassess support tier in 6 months"
  },
  "agent_trace": {
    "document_agent": [...],
    "validation_agent": [...],
    "eligibility_agent": [...],
    "enablement_agent": [...]
  }
}
```

### 10.4 Integration Patterns

**Government Portal Integration:**
The API is stateless JSON REST — it can be embedded behind any government identity gateway by adding an Authorization header check in FastAPI middleware.

**Batch Processing:**
For bulk re-assessments, call `/api/assess/{id}` sequentially or in a thread pool (Ollama handles one request at a time; parallel calls queue internally).

**Webhook Notifications:**
Add a `webhook_url` field to the assessment request; the orchestrator can POST results to it on completion (not yet implemented — extension point in `final_decision_node`).

---

## 11. Test Strategy

### 11.1 Test Coverage

| Test File | Coverage Area | Key Tests |
|-----------|--------------|-----------|
| `test_ml_classifier.py` | ML model | Component scores in [0,100]; high-need > low-need; tier thresholds; model trains on synthetic data; feature vector shape (1,14) |
| `test_document_processor.py` | Document extraction | PDF extraction; CSV bank statement (mocked LLM); text resume; credit report; dispatcher routing |
| `test_agents.py` | Agent pipeline | ReAct trace structure (thought/action/observation); Reflexion critique presence; income discrepancy flagging; end-to-end orchestrator with mocked LLMs |
| `test_api.py` | REST endpoints | Health check; missing fields 422; chat-intake response structure; applicant 404 |

### 11.2 Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_agents.py -v

# Run with coverage report
pytest --cov=app --cov-report=term-missing
```

### 11.3 Mock Strategy

All tests mock LLM calls via `monkeypatch.setattr` to:
- Eliminate dependency on a running Ollama instance
- Make tests deterministic and fast (< 5 seconds total)
- Test agent logic independently of LLM response quality

```python
# Example: mock LLM and test validation logic
monkeypatch.setattr(
    validation_agent, "invoke_light_llm",
    lambda prompt, **kwargs: "Validation is comprehensive."
)
result = validation_agent.run_validation(applicant, extracted_docs)
assert len(result["flags"]) > 0  # Rule-based flags still fire
```

---

## 12. Deployment Guide

### 12.1 Local Development Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd social-support-app

# 2. Pull Ollama models (one-time)
ollama pull llama3
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text

# 3. Create Python environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 4. Start PostgreSQL
brew services start postgresql@14
createdb social_support_db

# 5. Start Langfuse (optional)
docker-compose up -d
# Register at http://localhost:3000, copy keys to .env

# 6. Configure .env
cp .env.example .env
# Edit DATABASE_URL, add LANGFUSE keys if using

# 7. Generate training data
python -m app.utils.synthetic_data

# 8. Start backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 9. Start frontend (new terminal)
streamlit run frontend/streamlit_app.py --server.port 8501
```

### 12.2 Environment Variables

```env
# Database
DATABASE_URL=postgresql://<username>@localhost:5432/social_support_db

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3
LIGHT_LLM_MODEL=llama3.2
VISION_MODEL=llava:7b
EMBEDDING_MODEL=nomic-embed-text

# Storage
CHROMA_PERSIST_DIR=./data/chroma_db
UPLOAD_DIR=./data/uploads
POLICY_DIR=./data/policies

# Langfuse (optional — leave blank to disable)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

### 12.3 Production Considerations

For a production government deployment:

| Concern | Recommendation |
|---------|---------------|
| Authentication | Add OAuth2/JWT middleware in FastAPI |
| Document encryption | Encrypt uploaded files at rest (AES-256) |
| Audit logging | Persist `react_traces` to a separate audit table |
| Model updates | Schedule monthly re-training with real (anonymised) case outcomes |
| Langfuse in production | Run on dedicated server; enable backup of Langfuse PostgreSQL |
| GPU acceleration | Ollama automatically uses GPU if CUDA/Metal is available |
| Rate limiting | Add `slowapi` middleware to prevent API abuse |

---

## 13. Security Considerations

### 13.1 Data Privacy

- **All inference is local**: No applicant PII is sent to any external API
- **Document retention**: Uploaded documents are stored in `./data/uploads/{applicant_id}/` with no external exposure
- **Database encryption**: PostgreSQL supports transparent data encryption (TDE) at the OS level

### 13.2 Input Validation

- Pydantic schemas enforce field types, ranges, and formats at the API boundary
- Emirates ID format is validated with regex: `^\d{3}-\d{4}-\d{7}-\d$`
- File upload size limits should be configured at the reverse proxy level (e.g., nginx: `client_max_body_size 10M`)

### 13.3 Known Limitations

- **LLM hallucination risk**: The enablement agent generates program recommendations from RAG context — always validate recommended programs against the official policy database before presenting to applicants
- **Adversarial documents**: A user could upload a crafted document with injected text. Mitigation: run documents through a text sanitiser before passing to LLM prompts
- **Model bias**: The GradientBoosting model is trained on synthetic data. Real-world deployment requires bias auditing against demographic subgroups before going live

---

*End of Technical Solution Document*

*For questions, refer to the project repository or contact the technical team.*
