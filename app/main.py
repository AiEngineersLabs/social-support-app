"""
FastAPI Backend for Social Support Application Workflow Automation
"""
import os
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, init_db
from app.models import Applicant, Document, Decision
from app.schemas import ApplicantCreate, ApplicantResponse, DecisionResponse, ChatRequest, ChatResponse
from app.agents.orchestrator import run_application_workflow, handle_chat_message
from app.services.ml_classifier import train_model, load_model
from app.services.vector_store import ingest_policy_documents, init_vector_store
from app.utils.synthetic_data import generate_training_data

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

    return {"message": "Document uploaded", "doc_id": db_doc.id, "doc_type": doc_type}


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

    response = handle_chat_message(request.message, applicant_data, decision_data)
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


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Social Support Application System"}
