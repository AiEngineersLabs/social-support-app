"""
Master Orchestrator — LangGraph StateGraph

Sequential 5-node pipeline with ReAct (Document, Eligibility, Enablement agents)
and Reflexion (Validation agent). Each node contributes to the shared WorkflowState.

Graph: document_processing → validation → eligibility → enablement → final_decision

Conditional edge: if validation flags >= 2 → recommendation overridden to MANUAL_REVIEW
"""
import json
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END

from app.agents.document_agent import process_all_documents
from app.agents.validation_agent import run_validation
from app.agents.eligibility_agent import run_eligibility_assessment
from app.agents.enablement_agent import generate_recommendations
from app.services.llm_service import invoke_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared workflow state
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict):
    applicant_data: dict
    documents: list
    extracted_docs: dict
    validation_result: dict
    eligibility_result: dict
    enablement_result: dict
    final_decision: dict
    agent_trace: list       # high-level trace for UI display
    react_traces: dict      # per-agent detailed ReAct/Reflexion traces
    current_step: str
    error: str


# ---------------------------------------------------------------------------
# Node 1: Document Processing (ReAct)
# ---------------------------------------------------------------------------

def document_processing_node(state: WorkflowState) -> dict:
    trace = state.get("agent_trace", [])
    react_traces = state.get("react_traces", {})

    trace.append({
        "agent": "DocumentProcessingAgent",
        "action": "START",
        "reasoning_framework": "ReAct",
        "step": "document_processing",
    })

    documents = state.get("documents", [])
    if not documents:
        trace.append({
            "agent": "DocumentProcessingAgent",
            "action": "SKIP",
            "reason": "No documents uploaded — proceeding with form data only",
        })
        return {
            "extracted_docs": {},
            "agent_trace": trace,
            "react_traces": react_traces,
            "current_step": "validation",
        }

    try:
        extracted = process_all_documents(documents)

        # Pull out the ReAct trace embedded by the agent
        doc_react_trace = extracted.pop("__react_trace__", [])
        react_traces["document_agent"] = doc_react_trace

        trace.append({
            "agent": "DocumentProcessingAgent",
            "action": "COMPLETE",
            "docs_processed": len(extracted),
            "doc_types": list(extracted.keys()),
            "react_steps": len(doc_react_trace),
        })
        return {
            "extracted_docs": extracted,
            "agent_trace": trace,
            "react_traces": react_traces,
            "current_step": "validation",
        }
    except Exception as e:
        logger.error("Document processing error: %s", e)
        trace.append({"agent": "DocumentProcessingAgent", "action": "ERROR", "error": str(e)})
        return {
            "extracted_docs": {},
            "agent_trace": trace,
            "react_traces": react_traces,
            "current_step": "validation",
            "error": f"Document processing error: {e}",
        }


# ---------------------------------------------------------------------------
# Node 2: Validation (Reflexion)
# ---------------------------------------------------------------------------

def validation_node(state: WorkflowState) -> dict:
    trace = state.get("agent_trace", [])
    react_traces = state.get("react_traces", {})

    trace.append({
        "agent": "DataValidationAgent",
        "action": "START",
        "reasoning_framework": "Reflexion",
        "step": "validation",
    })

    applicant_data = state.get("applicant_data", {})
    extracted_docs = state.get("extracted_docs", {})

    try:
        validation_result = run_validation(applicant_data, extracted_docs)

        # Pull out Reflexion trace
        reflexion_trace = validation_result.pop("reflexion_trace", [])
        react_traces["validation_agent"] = reflexion_trace

        validated_applicant = validation_result.get("validated_data", applicant_data)

        trace.append({
            "agent": "DataValidationAgent",
            "action": "COMPLETE",
            "reasoning_framework": "Reflexion",
            "is_valid": validation_result.get("is_valid", True),
            "flags_count": len(validation_result.get("flags", [])),
            "warnings_count": len(validation_result.get("warnings", [])),
            "reflexion_critique": validation_result.get("reflexion_critique", ""),
        })

        return {
            "validation_result": validation_result,
            "applicant_data": validated_applicant,
            "agent_trace": trace,
            "react_traces": react_traces,
            "current_step": "eligibility",
        }
    except Exception as e:
        logger.error("Validation error: %s", e)
        trace.append({"agent": "DataValidationAgent", "action": "ERROR", "error": str(e)})
        return {
            "validation_result": {"is_valid": True, "flags": [], "warnings": [str(e)]},
            "agent_trace": trace,
            "react_traces": react_traces,
            "current_step": "eligibility",
        }


# ---------------------------------------------------------------------------
# Node 3: Eligibility Assessment (ReAct + ML)
# ---------------------------------------------------------------------------

def eligibility_node(state: WorkflowState) -> dict:
    trace = state.get("agent_trace", [])
    react_traces = state.get("react_traces", {})

    trace.append({
        "agent": "EligibilityAssessmentAgent",
        "action": "START",
        "reasoning_framework": "ReAct + ML (GradientBoosting)",
        "step": "eligibility",
    })

    applicant_data = state.get("applicant_data", {})

    try:
        eligibility_result = run_eligibility_assessment(applicant_data)

        # Pull out ReAct trace from eligibility
        eligibility_react_trace = eligibility_result.pop("react_trace", [])
        react_traces["eligibility_agent"] = eligibility_react_trace

        trace.append({
            "agent": "EligibilityAssessmentAgent",
            "action": "COMPLETE",
            "recommendation": eligibility_result.get("recommendation"),
            "eligibility_score": eligibility_result.get("eligibility_score"),
            "confidence": eligibility_result.get("confidence_score"),
            "react_steps": len(eligibility_react_trace),
        })

        return {
            "eligibility_result": eligibility_result,
            "agent_trace": trace,
            "react_traces": react_traces,
            "current_step": "enablement",
        }
    except Exception as e:
        logger.error("Eligibility error: %s", e)
        trace.append({"agent": "EligibilityAssessmentAgent", "action": "ERROR", "error": str(e)})
        return {
            "eligibility_result": {"recommendation": "REVIEW", "error": str(e)},
            "agent_trace": trace,
            "react_traces": react_traces,
            "current_step": "enablement",
        }


# ---------------------------------------------------------------------------
# Node 4: Enablement Recommendation (ReAct + RAG)
# ---------------------------------------------------------------------------

def enablement_node(state: WorkflowState) -> dict:
    trace = state.get("agent_trace", [])
    react_traces = state.get("react_traces", {})

    trace.append({
        "agent": "EnablementRecommenderAgent",
        "action": "START",
        "reasoning_framework": "ReAct + RAG",
        "step": "enablement",
    })

    applicant_data = state.get("applicant_data", {})
    eligibility_result = state.get("eligibility_result", {})

    try:
        enablement_result = generate_recommendations(applicant_data, eligibility_result)

        enablement_react_trace = enablement_result.pop("react_trace", [])
        react_traces["enablement_agent"] = enablement_react_trace

        trace.append({
            "agent": "EnablementRecommenderAgent",
            "action": "COMPLETE",
            "programs_recommended": len(enablement_result.get("recommended_programs", [])),
            "rag_used": enablement_result.get("rag_used", False),
        })

        return {
            "enablement_result": enablement_result,
            "agent_trace": trace,
            "react_traces": react_traces,
            "current_step": "final_decision",
        }
    except Exception as e:
        logger.error("Enablement error: %s", e)
        trace.append({"agent": "EnablementRecommenderAgent", "action": "ERROR", "error": str(e)})
        return {
            "enablement_result": {"recommended_programs": [], "error": str(e)},
            "agent_trace": trace,
            "react_traces": react_traces,
            "current_step": "final_decision",
        }


# ---------------------------------------------------------------------------
# Node 5: Final Decision Compiler
# ---------------------------------------------------------------------------

def final_decision_node(state: WorkflowState) -> dict:
    trace = state.get("agent_trace", [])
    trace.append({"agent": "MasterOrchestrator", "action": "COMPILING_DECISION"})

    eligibility = state.get("eligibility_result", {})
    validation = state.get("validation_result", {})
    enablement = state.get("enablement_result", {})

    final_decision = {
        "recommendation": eligibility.get("recommendation", "REVIEW"),
        "support_tier": eligibility.get("support_tier", "Unknown"),
        "confidence_score": eligibility.get("confidence_score", 0),
        "eligibility_score": eligibility.get("eligibility_score", 0),
        "income_score": eligibility.get("income_score", 0),
        "employment_score": eligibility.get("employment_score", 0),
        "family_score": eligibility.get("family_score", 0),
        "wealth_score": eligibility.get("wealth_score", 0),
        "demographic_score": eligibility.get("demographic_score", 0),
        "reasoning": eligibility.get("reasoning", ""),
        "validation_flags": validation.get("flags", []),
        "validation_warnings": validation.get("warnings", []),
        "validation_summary": validation.get("validation_summary", ""),
        "enablement_recommendations": enablement.get("recommended_programs", []),
        "career_pathway": enablement.get("career_pathway", ""),
        "immediate_actions": enablement.get("immediate_actions", []),
        "long_term_plan": enablement.get("long_term_plan", ""),
    }

    # Conditional override: 2+ validation flags → MANUAL_REVIEW
    flags = validation.get("flags", [])
    if len(flags) >= 2:
        final_decision["recommendation"] = "MANUAL_REVIEW"
        final_decision["reasoning"] += (
            "\n\nNOTE: Multiple data discrepancies detected by the Reflexion validation agent. "
            "This application requires manual review before a final determination."
        )

    trace.append({
        "agent": "MasterOrchestrator",
        "action": "DECISION_COMPLETE",
        "recommendation": final_decision["recommendation"],
        "score": final_decision["eligibility_score"],
    })

    return {
        "final_decision": final_decision,
        "agent_trace": trace,
        "current_step": "complete",
    }


# ---------------------------------------------------------------------------
# Build and compile LangGraph workflow
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)

    workflow.add_node("step_document_processing", document_processing_node)
    workflow.add_node("step_validation", validation_node)
    workflow.add_node("step_eligibility", eligibility_node)
    workflow.add_node("step_enablement", enablement_node)
    workflow.add_node("step_final_decision", final_decision_node)

    workflow.set_entry_point("step_document_processing")
    workflow.add_edge("step_document_processing", "step_validation")
    workflow.add_edge("step_validation", "step_eligibility")
    workflow.add_edge("step_eligibility", "step_enablement")
    workflow.add_edge("step_enablement", "step_final_decision")
    workflow.add_edge("step_final_decision", END)

    return workflow.compile()


app_workflow = build_workflow()


def run_application_workflow(applicant_data: dict, documents: list = None) -> dict:
    """Execute the full application assessment workflow."""
    initial_state = {
        "applicant_data": applicant_data,
        "documents": documents or [],
        "extracted_docs": {},
        "validation_result": {},
        "eligibility_result": {},
        "enablement_result": {},
        "final_decision": {},
        "agent_trace": [{"agent": "MasterOrchestrator", "action": "WORKFLOW_START"}],
        "react_traces": {},
        "current_step": "document_processing",
        "error": "",
    }
    return app_workflow.invoke(initial_state)


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def handle_chat_message(
    message: str,
    applicant_data: dict = None,
    decision: dict = None,
    chat_history: list = None,
) -> str:
    """Handle an interactive chat message with full applicant/decision context."""
    context = ""
    if applicant_data:
        context += f"\nApplicant: {applicant_data.get('full_name', 'Unknown')}"
        context += f"\nEmployment: {applicant_data.get('employment_status', 'Unknown')}"
        context += f"\nIncome: AED {applicant_data.get('monthly_income', 0):,.0f}/month"
        context += f"\nFamily Size: {applicant_data.get('family_size', 0)} (Dependents: {applicant_data.get('dependents', 0)})"
        context += f"\nEducation: {applicant_data.get('education_level', 'Unknown')}"
        context += f"\nNationality: {applicant_data.get('nationality', 'Unknown')}"

    if decision:
        context += f"\n\nDecision: {decision.get('recommendation', 'Pending')}"
        context += f"\nEligibility Score: {decision.get('eligibility_score', 'N/A')}/100"
        context += f"\nSupport Tier: {decision.get('support_tier', 'N/A')}"
        programs = decision.get("enablement_recommendations", [])
        if programs:
            prog_names = [p.get("program_name", str(p)) if isinstance(p, dict) else str(p) for p in programs[:3]]
            context += f"\nRecommended Programs: {', '.join(prog_names)}"

    # Include recent chat history for context continuity
    history_str = ""
    if chat_history:
        recent = chat_history[-6:]  # last 3 exchanges
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_str += f"{role}: {msg.get('content', '')}\n"

    prompt = f"""You are a helpful, empathetic AI assistant for the UAE Government Social Support Program.
You help applicants understand their application status, assessment results, and available support programs.

{f"APPLICANT CONTEXT:{context}" if context else "No applicant context yet."}

{f"RECENT CONVERSATION:{chr(10)}{history_str}" if history_str else ""}

USER: {message}

Provide a helpful, clear, and empathetic response. 
- Reference the applicant's specific situation when relevant
- For program questions, explain eligibility and how to apply
- Be concise but thorough
- Always be supportive and professional"""

    try:
        return invoke_llm(prompt, name="chat_response")
    except Exception as e:
        return f"I apologise, I'm unable to process your request right now. Please try again. (Error: {e})"
