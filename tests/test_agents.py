"""
Tests for the agent pipeline (document, validation, eligibility, enablement, orchestrator).
All LLM and external service calls are mocked.
"""
import pytest
import json


SAMPLE_APPLICANT = {
    "full_name": "Ahmed Al-Rashid",
    "emirates_id": "784-1990-1234567-8",
    "age": 30,
    "gender": "Male",
    "nationality": "UAE",
    "marital_status": "Married",
    "family_size": 4,
    "dependents": 2,
    "education_level": "High School",
    "employment_status": "Unemployed",
    "monthly_income": 800.0,
    "total_assets": 10000.0,
    "total_liabilities": 5000.0,
    "years_of_experience": 2.0,
}


# ── Document Agent ─────────────────────────────────────────────────────────

def test_process_all_documents_empty():
    from app.agents.document_agent import process_all_documents
    result = process_all_documents([])
    # Should return empty dict (with react trace)
    assert isinstance(result, dict)
    assert "__react_trace__" in result


def test_react_trace_structure(monkeypatch, tmp_path):
    """ReAct trace should contain thought/action/observation entries."""
    import app.agents.document_agent as doc_agent

    monkeypatch.setattr(
        doc_agent, "extract_bank_statement",
        lambda fp: {"estimated_monthly_income": 800, "average_monthly_balance": 2000}
    )

    fake_csv = tmp_path / "bank.csv"
    fake_csv.write_text("Date,Amt\n2024-01-01,800\n")

    result = process_doc = doc_agent.process_all_documents([
        {"doc_type": "bank_statement", "file_path": str(fake_csv)}
    ])

    trace = result.pop("__react_trace__", [])
    assert len(trace) > 0
    types = [e.get("type") for e in trace]
    assert "thought" in types
    assert "action" in types
    assert "observation" in types


# ── Validation Agent ───────────────────────────────────────────────────────

def test_run_validation_no_documents(monkeypatch):
    from app.agents import validation_agent

    monkeypatch.setattr(
        validation_agent, "invoke_light_llm",
        lambda prompt, **kwargs: "Validation is comprehensive."
    )

    result = validation_agent.run_validation(SAMPLE_APPLICANT, {})
    assert "is_valid" in result
    assert "flags" in result
    assert "warnings" in result
    assert isinstance(result["flags"], list)


def test_income_discrepancy_raises_flag(monkeypatch):
    from app.agents import validation_agent

    monkeypatch.setattr(
        validation_agent, "invoke_light_llm",
        lambda prompt, **kwargs: "Validation is comprehensive."
    )

    extracted_docs = {
        "bank_statement": {
            "data": {"estimated_monthly_income": 15000, "average_monthly_balance": 20000}
        }
    }
    applicant = {**SAMPLE_APPLICANT, "monthly_income": 800.0}
    result = validation_agent.run_validation(applicant, extracted_docs)
    # 800 declared vs 15000 bank = >30% discrepancy → should flag
    assert len(result["flags"]) > 0


def test_reflexion_trace_present(monkeypatch):
    from app.agents import validation_agent

    monkeypatch.setattr(
        validation_agent, "invoke_light_llm",
        lambda prompt, **kwargs: "Validation is comprehensive."
    )

    result = validation_agent.run_validation(SAMPLE_APPLICANT, {})
    # reflexion_trace is consumed by orchestrator — it's in the raw return from run_validation
    # but orchestrator pops it; here we test the function directly
    assert "reflexion_critique" in result


# ── Eligibility Agent ──────────────────────────────────────────────────────

def test_eligibility_returns_recommendation(monkeypatch):
    from app.agents import eligibility_agent

    monkeypatch.setattr(
        eligibility_agent, "invoke_llm",
        lambda prompt, **kwargs: "The applicant qualifies due to unemployment and family size."
    )

    result = eligibility_agent.run_eligibility_assessment(SAMPLE_APPLICANT)
    assert result["recommendation"] in ("APPROVE", "SOFT_DECLINE")
    assert 0 <= result["eligibility_score"] <= 100
    assert result["reasoning"] != ""
    assert "react_trace" in result


# ── Enablement Agent ───────────────────────────────────────────────────────

def test_enablement_returns_programs(monkeypatch):
    from app.agents import enablement_agent

    monkeypatch.setattr(
        enablement_agent, "query_policies",
        lambda q, **kwargs: "Career counseling and job matching programs available."
    )
    monkeypatch.setattr(
        enablement_agent, "invoke_llm",
        lambda prompt, **kwargs: json.dumps({
            "recommended_programs": [
                {"program_name": "Job Matching", "priority": "high", "reason": "Unemployed applicant"}
            ],
            "career_pathway": "IT sector",
            "immediate_actions": ["Register at job center"],
            "long_term_plan": "Secure employment within 6 months"
        })
    )

    result = enablement_agent.generate_recommendations(
        SAMPLE_APPLICANT,
        {"recommendation": "APPROVE", "support_tier": "Tier 1 - Emergency Support"}
    )
    assert "recommended_programs" in result
    assert len(result["recommended_programs"]) > 0
    assert "react_trace" in result


# ── Orchestrator (end-to-end mock) ─────────────────────────────────────────

def test_orchestrator_full_pipeline(monkeypatch):
    """End-to-end pipeline test with all LLM calls mocked."""
    from app.agents import orchestrator
    from app.agents import validation_agent, eligibility_agent, enablement_agent

    # Mock all LLM calls
    monkeypatch.setattr(validation_agent, "invoke_light_llm",
                        lambda p, **kw: "Validation is comprehensive.")
    monkeypatch.setattr(eligibility_agent, "invoke_llm",
                        lambda p, **kw: "Eligible based on unemployment.")
    monkeypatch.setattr(enablement_agent, "query_policies",
                        lambda q, **kw: "Career programs available.")
    monkeypatch.setattr(enablement_agent, "invoke_llm",
                        lambda p, **kw: json.dumps({
                            "recommended_programs": [{"program_name": "Test Program", "priority": "high", "reason": "test"}],
                            "career_pathway": "IT", "immediate_actions": ["action1"], "long_term_plan": "plan"
                        }))

    result = orchestrator.run_application_workflow(SAMPLE_APPLICANT, documents=[])

    assert "final_decision" in result
    assert result["final_decision"]["recommendation"] in ("APPROVE", "SOFT_DECLINE", "MANUAL_REVIEW")
    assert "agent_trace" in result
    assert len(result["agent_trace"]) > 0
