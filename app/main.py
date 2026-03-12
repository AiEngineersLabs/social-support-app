"""
FastAPI Backend for Social Support Application Workflow Automation
"""
import os
import re
import json
import logging
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, init_db
from app.models import Applicant, Document, Decision
from app.schemas import (
    ApplicantCreate, ApplicantResponse, DecisionResponse,
    ChatRequest, ChatResponse, ChatIntakeRequest, ChatIntakeResponse,
)
from app.agents.orchestrator import run_application_workflow, handle_chat_message
from app.services.document_processor import process_document
from app.services.ml_classifier import train_model, load_model
from app.services.vector_store import ingest_policy_documents, init_vector_store
from app.utils.synthetic_data import generate_training_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field type metadata for deterministic fallback extraction
# ---------------------------------------------------------------------------
INT_FIELDS = {"age", "family_size", "dependents"}
FLOAT_FIELDS = {"monthly_income", "total_assets", "total_liabilities", "years_of_experience"}
ENUM_FIELDS = {
    "gender": ["Male", "Female"],
    "marital_status": ["Single", "Married", "Divorced", "Widowed"],
    "education_level": ["None", "Primary", "High School", "Diploma", "Bachelor", "Master", "PhD"],
    "employment_status": ["Employed", "Unemployed", "Part-Time", "Self-Employed"],
}
STRING_FIELDS = {"full_name", "nationality"}
# Matches both 784-YYYY-NNNNNNN-N (with dashes) and 784YYYYNNNNNNN (15 digits, no dashes)
EMIRATES_ID_DASHED = re.compile(r"784-\d{4}-\d{7}-\d")
EMIRATES_ID_PLAIN = re.compile(r"784\d{12}")


def _coerce_field_types(extracted: dict) -> dict:
    """Coerce string representations to correct types (e.g. '32' -> 32 for int fields)."""
    for field, value in list(extracted.items()):
        if field in INT_FIELDS and not isinstance(value, int):
            try:
                extracted[field] = int(float(str(value)))
            except (ValueError, TypeError):
                del extracted[field]
        elif field in FLOAT_FIELDS and not isinstance(value, (int, float)):
            try:
                extracted[field] = float(str(value))
            except (ValueError, TypeError):
                del extracted[field]
    return extracted


def _fallback_extract(field: str, message: str) -> object:
    """Deterministic regex-based extraction when LLM fails for a specific field."""
    msg = message.strip()

    if field in INT_FIELDS:
        match = re.search(r"\d+", msg)
        if match:
            return int(match.group())

    elif field in FLOAT_FIELDS:
        match = re.search(r"[\d,]+\.?\d*", msg.replace(",", ""))
        if match:
            return float(match.group())

    elif field == "emirates_id":
        # Try dashed format first (784-YYYY-NNNNNNN-N)
        match = EMIRATES_ID_DASHED.search(msg)
        if match:
            return match.group()
        # Try plain digits (784YYYYNNNNNNN) and normalize to dashed format
        match = EMIRATES_ID_PLAIN.search(msg)
        if match:
            digits = match.group()
            return f"{digits[:3]}-{digits[3:7]}-{digits[7:14]}-{digits[14]}"

    elif field in ENUM_FIELDS:
        for option in ENUM_FIELDS[field]:
            if option.lower() in msg.lower():
                return option

    elif field in STRING_FIELDS:
        cleaned = msg.strip()
        if cleaned:
            return cleaned

    return None


app = FastAPI(
    title="Social Support Application System",
    description="AI-powered social support application assessment with agentic workflow",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database, ML model, and vector store on startup."""
    # Create database tables
    init_db()

    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Train ML model if not already trained
    if not load_model():
        print("Training ML eligibility classifier...")
        training_data = generate_training_data(500)
        train_model(training_data)
        print("ML model trained successfully.")

    # Initialize vector store with policy documents
    print("Initializing vector store with policy documents...")
    try:
        policy_dir = Path(settings.POLICY_DIR)
        if policy_dir.exists():
            docs = []
            for f in policy_dir.glob("*.txt"):
                content = f.read_text()
                docs.append({"text": content, "metadata": {"source": f.name}})
            if docs:
                ingest_policy_documents(docs)
                print(f"Ingested {len(docs)} policy documents.")
        else:
            init_vector_store()
    except Exception as e:
        print(f"Vector store initialization warning: {e}")
        print("Continuing without vector store...")


# Application Endpoints

@app.post("/api/submit-application", response_model=ApplicantResponse)
async def submit_application(applicant: ApplicantCreate, db: Session = Depends(get_db)):
    """Submit a new support application."""
    # Check for duplicate Emirates ID
    existing = db.query(Applicant).filter(Applicant.emirates_id == applicant.emirates_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Application with this Emirates ID already exists")

    db_applicant = Applicant(**applicant.model_dump())
    db.add(db_applicant)
    db.commit()
    db.refresh(db_applicant)
    return db_applicant


@app.post("/api/upload-document/{applicant_id}")
async def upload_document(
    applicant_id: int,
    doc_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a document for an applicant."""
    applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    # Save file
    upload_dir = Path(settings.UPLOAD_DIR) / str(applicant_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{doc_type}_{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Store document record
    db_doc = Document(
        applicant_id=applicant_id,
        doc_type=doc_type,
        file_path=str(file_path),
    )
    db.add(db_doc)
    db.commit()

    # Run OCR/processing immediately and return extracted data
    try:
        extracted_data = process_document(doc_type, str(file_path))
    except Exception as exc:
        logger.warning("Document processing failed for %s: %s", doc_type, exc)
        extracted_data = {}

    return {"message": "Document uploaded", "doc_id": db_doc.id, "doc_type": doc_type, "extracted_data": extracted_data}


@app.post("/api/assess/{applicant_id}")
async def assess_application(applicant_id: int, db: Session = Depends(get_db)):
    """Run the full AI assessment workflow for an applicant."""
    applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    # Check if already assessed
    existing_decision = db.query(Decision).filter(Decision.applicant_id == applicant_id).first()
    if existing_decision:
        raise HTTPException(status_code=400, detail="Application already assessed. Delete existing decision first.")

    # Gather applicant data
    applicant_data = {
        "full_name": applicant.full_name,
        "emirates_id": applicant.emirates_id,
        "age": applicant.age,
        "gender": applicant.gender,
        "nationality": applicant.nationality,
        "marital_status": applicant.marital_status,
        "family_size": applicant.family_size,
        "dependents": applicant.dependents,
        "education_level": applicant.education_level,
        "employment_status": applicant.employment_status,
        "years_of_experience": applicant.years_of_experience,
        "monthly_income": applicant.monthly_income,
        "total_assets": applicant.total_assets,
        "total_liabilities": applicant.total_liabilities,
    }

    # Gather documents
    documents = [
        {"doc_type": doc.doc_type, "file_path": doc.file_path}
        for doc in db.query(Document).filter(Document.applicant_id == applicant_id).all()
    ]

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="At least one document must be uploaded before assessment. Please upload a document first.",
        )

    # Run the agentic workflow
    result = run_application_workflow(applicant_data, documents)

    # Save decision
    final = result.get("final_decision", {})
    db_decision = Decision(
        applicant_id=applicant_id,
        recommendation=final.get("recommendation", "REVIEW"),
        confidence_score=final.get("confidence_score", 0),
        eligibility_score=final.get("eligibility_score", 0),
        income_score=final.get("income_score", 0),
        employment_score=final.get("employment_score", 0),
        family_score=final.get("family_score", 0),
        wealth_score=final.get("wealth_score", 0),
        demographic_score=final.get("demographic_score", 0),
        reasoning=final.get("reasoning", ""),
        enablement_recommendations=final.get("enablement_recommendations", []),
        agent_trace=result.get("agent_trace", []),
    )
    db.add(db_decision)
    db.commit()
    db.refresh(db_decision)

    return {
        "applicant_id": applicant_id,
        "decision": final,
        "agent_trace": result.get("agent_trace", []),
    }


@app.get("/api/decision/{applicant_id}")
async def get_decision(applicant_id: int, db: Session = Depends(get_db)):
    """Get the assessment decision for an applicant."""
    decision = db.query(Decision).filter(Decision.applicant_id == applicant_id).first()
    if not decision:
        raise HTTPException(status_code=404, detail="No decision found for this applicant")

    return {
        "applicant_id": applicant_id,
        "recommendation": decision.recommendation,
        "confidence_score": decision.confidence_score,
        "eligibility_score": decision.eligibility_score,
        "income_score": decision.income_score,
        "employment_score": decision.employment_score,
        "family_score": decision.family_score,
        "wealth_score": decision.wealth_score,
        "demographic_score": decision.demographic_score,
        "reasoning": decision.reasoning,
        "enablement_recommendations": decision.enablement_recommendations,
        "agent_trace": decision.agent_trace,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Interactive chat about an application."""
    applicant = db.query(Applicant).filter(Applicant.id == request.applicant_id).first()
    decision = db.query(Decision).filter(Decision.applicant_id == request.applicant_id).first()

    applicant_data = None
    decision_data = None

    if applicant:
        applicant_data = {
            "full_name": applicant.full_name,
            "employment_status": applicant.employment_status,
            "monthly_income": applicant.monthly_income,
            "family_size": applicant.family_size,
        }

    if decision:
        decision_data = {
            "recommendation": decision.recommendation,
            "eligibility_score": decision.eligibility_score,
            "support_tier": "Tier 1" if decision.eligibility_score >= 80 else
                           "Tier 2" if decision.eligibility_score >= 55 else
                           "Tier 3" if decision.eligibility_score >= 40 else "Tier 4",
        }

    chat_history = getattr(request, "chat_history", None)
    response = handle_chat_message(request.message, applicant_data, decision_data, chat_history)
    return ChatResponse(response=response, agent_used="ChatAgent")


@app.get("/api/applicants")
async def list_applicants(db: Session = Depends(get_db)):
    """List all applicants."""
    applicants = db.query(Applicant).all()
    return [
        {
            "id": a.id,
            "full_name": a.full_name,
            "emirates_id": a.emirates_id,
            "employment_status": a.employment_status,
            "monthly_income": a.monthly_income,
            "has_decision": a.decision is not None,
        }
        for a in applicants
    ]


@app.delete("/api/applicant/{applicant_id}")
async def delete_applicant(applicant_id: int, db: Session = Depends(get_db)):
    """Delete an applicant and all related data."""
    applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    db.query(Decision).filter(Decision.applicant_id == applicant_id).delete()
    db.query(Document).filter(Document.applicant_id == applicant_id).delete()
    db.delete(applicant)
    db.commit()
    return {"message": "Applicant deleted"}


@app.post("/api/chat-intake", response_model=ChatIntakeResponse)
async def chat_intake(request: ChatIntakeRequest):
    """
    Conversational intake endpoint for the AI chatbot.

    Extracts structured applicant fields from a free-text user message,
    identifies which required fields are still missing, and returns the
    next natural question to ask.
    """
    from app.services.llm_service import invoke_llm, extract_json_from_response

    REQUIRED_FIELDS = [
        "full_name", "emirates_id", "age", "gender", "nationality",
        "marital_status", "family_size", "dependents",
        "education_level", "employment_status", "monthly_income",
        "total_assets", "total_liabilities", "years_of_experience",
    ]

    # Deterministic questions — LLM must NOT freestyle these
    FIELD_QUESTIONS = {
        "full_name":          "Could you please tell me your full name?",
        "emirates_id":        "What is your Emirates ID number? (format: 784-YYYY-NNNNNNN-N)",
        "age":                "How old are you?",
        "gender":             "What is your gender? (Male or Female)",
        "nationality":        "What is your nationality?",
        "marital_status":     "What is your marital status? (Single, Married, Divorced, or Widowed)",
        "family_size":        "How many people are in your household in total (including yourself)?",
        "dependents":         "How many dependents do you have (children or others you support)?",
        "education_level":    "What is your highest level of education? (None, Primary, High School, Diploma, Bachelor, Master, or PhD)",
        "employment_status":  "What is your current employment status? (Employed, Unemployed, Part-Time, or Self-Employed)",
        "monthly_income":     "What is your monthly income in AED? (Enter 0 if you have no income)",
        "total_assets":       "What is the total value of your assets in AED? (savings, property, vehicles, etc.)",
        "total_liabilities":  "What is the total value of your liabilities/debts in AED?",
        "years_of_experience":"How many years of work experience do you have in total?",
    }

    collected = request.collected_fields or {}
    missing = [f for f in REQUIRED_FIELDS if f not in collected or collected[f] in ("", None)]

    if not missing:
        return ChatIntakeResponse(
            extracted_fields={},
            next_question="",
            is_complete=True,
            missing_fields=[],
        )

    collected_str = json.dumps(collected, indent=2) if collected else "None yet"

    prompt = f"""You are extracting structured data from a user's chat message for a UAE social support application.

Already collected fields:
{collected_str}

Fields still needed (extract ONLY these from the message):
{missing}

User's message: "{request.message}"

Instructions:
- Extract as many of the still-needed fields as possible from the message.
- Do NOT invent or assume values not stated in the message.
- Return ONLY a JSON object with this exact structure:
{{
  "extracted_fields": {{
    "field_name": value
  }}
}}

Field extraction rules:
- full_name: full name as stated
- emirates_id: string matching 784-YYYY-NNNNNNN-N pattern
- age: integer years
- gender: exactly "Male" or "Female"
- nationality: country name as stated (e.g. "UAE", "Indian")
- marital_status: exactly one of "Single", "Married", "Divorced", "Widowed"
- family_size: integer total household members including applicant
- dependents: integer count of dependents
- education_level: exactly one of "None", "Primary", "High School", "Diploma", "Bachelor", "Master", "PhD"
- employment_status: exactly one of "Employed", "Unemployed", "Part-Time", "Self-Employed"
- monthly_income: float AED amount (use 0 if person says unemployed with no income)
- total_assets: float AED value of all assets
- total_liabilities: float AED value of all debts
- years_of_experience: float years of work experience

Return ONLY valid JSON. Do NOT include next_question or any other keys."""

    try:
        response = invoke_llm(prompt, name="chat_intake")
        parsed = extract_json_from_response(response)
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
        parsed = {}

    extracted = parsed.get("extracted_fields", {})
    if not extracted:
        logger.warning("LLM returned no extracted_fields for message: %r", request.message)

    # Coerce string values to correct types (e.g. "32" -> 32 for age)
    extracted = _coerce_field_types(extracted)

    # Merge LLM-extracted fields with already-collected fields
    merged = {**collected, **extracted}
    still_missing = [f for f in REQUIRED_FIELDS if f not in merged or merged[f] in ("", None)]

    # Deterministic fallback: if LLM failed to extract fields that were asked,
    # try regex-based extraction from the raw user message
    # Determine which field was being asked (the first missing field before LLM call)
    current_field = missing[0] if missing else None

    if still_missing and current_field and current_field in still_missing:
        # Only attempt fallback for the field that was actively being asked
        fallback_value = _fallback_extract(current_field, request.message)
        if fallback_value is not None:
            logger.info("Fallback extracted %s=%r from message", current_field, fallback_value)
            extracted[current_field] = fallback_value
            merged[current_field] = fallback_value

            # Recompute missing after fallback
            still_missing = [f for f in REQUIRED_FIELDS if f not in merged or merged[f] in ("", None)]

    is_complete = len(still_missing) == 0

    # Generate next_question deterministically — never rely on LLM for this
    if is_complete:
        next_q = ""
    else:
        next_field = still_missing[0]
        next_q = FIELD_QUESTIONS[next_field]

    return ChatIntakeResponse(
        extracted_fields=extracted,
        next_question=next_q,
        is_complete=is_complete,
        missing_fields=still_missing,
    )


@app.post("/api/reassess/{applicant_id}")
async def reassess_application(applicant_id: int, db: Session = Depends(get_db)):
    """Delete existing decision and re-run assessment (for demo purposes)."""
    applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")
    db.query(Decision).filter(Decision.applicant_id == applicant_id).delete()
    db.commit()
    return {"message": "Decision cleared. Call /api/assess/{id} to re-assess."}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Social Support Application System"}
