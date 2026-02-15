"""
Master Orchestrator Agent using LangGraph
Coordinates the entire application assessment workflow:
1. Document Processing -> 2. Data Validation -> 3. Eligibility Assessment -> 4. Enablement Recommendation

Uses ReAct-style reasoning for agent coordination.
"""
import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from app.agents.document_agent import process_all_documents
from app.agents.validation_agent import run_validation
from app.agents.eligibility_agent import run_eligibility_assessment
from app.agents.enablement_agent import generate_recommendations
from app.services.llm_service import invoke_llm


# State Definition

class WorkflowState(TypedDict):
    """State that flows through the orchestration pipeline."""
    applicant_data: dict
    documents: list  # list of {"doc_type": str, "file_path": str}
    extracted_docs: dict
    validation_result: dict
    eligibility_result: dict
    enablement_result: dict
    final_decision: dict
    agent_trace: list  # trace of agent actions for observability
    current_step: str
    error: str


# Node Functions

def document_processing_node(state: WorkflowState) -> dict:
    """Node 1: Process all uploaded documents."""
    trace = state.get("agent_trace", [])
    trace.append({"agent": "DocumentProcessingAgent", "action": "START", "step": "document_processing"})

    documents = state.get("documents", [])
    if not documents:
        trace.append({"agent": "DocumentProcessingAgent", "action": "SKIP", "reason": "No documents uploaded"})
        return {
            "extracted_docs": {},
            "agent_trace": trace,
            "current_step": "validation",
        }

    try:
        extracted = process_all_documents(documents)
        trace.append({
            "agent": "DocumentProcessingAgent",
            "action": "COMPLETE",
            "docs_processed": len(extracted),
            "doc_types": list(extracted.keys()),
        })
        return {
            "extracted_docs": extracted,
            "agent_trace": trace,
            "current_step": "validation",
        }
    except Exception as e:
        trace.append({"agent": "DocumentProcessingAgent", "action": "ERROR", "error": str(e)})
        return {
            "extracted_docs": {},
            "agent_trace": trace,
            "current_step": "validation",
            "error": f"Document processing error: {str(e)}",
        }


def validation_node(state: WorkflowState) -> dict:
    """Node 2: Validate and cross-check extracted data."""
    trace = state.get("agent_trace", [])
    trace.append({"agent": "DataValidationAgent", "action": "START", "step": "validation"})

    applicant_data = state.get("applicant_data", {})
    extracted_docs = state.get("extracted_docs", {})

    try:
        validation_result = run_validation(applicant_data, extracted_docs)

        # Merge validated data back into applicant data
        validated_applicant = validation_result.get("validated_data", applicant_data)

        trace.append({
            "agent": "DataValidationAgent",
            "action": "COMPLETE",
            "is_valid": validation_result.get("is_valid", True),
            "flags_count": len(validation_result.get("flags", [])),
            "warnings_count": len(validation_result.get("warnings", [])),
        })

        return {
            "validation_result": validation_result,
            "applicant_data": validated_applicant,
            "agent_trace": trace,
            "current_step": "eligibility",
        }
    except Exception as e:
        trace.append({"agent": "DataValidationAgent", "action": "ERROR", "error": str(e)})
        return {
            "validation_result": {"is_valid": True, "flags": [], "warnings": [str(e)]},
            "agent_trace": trace,
            "current_step": "eligibility",
        }


def eligibility_node(state: WorkflowState) -> dict:
    """Node 3: Assess eligibility and generate recommendation."""
    trace = state.get("agent_trace", [])
    trace.append({"agent": "EligibilityAssessmentAgent", "action": "START", "step": "eligibility"})

    applicant_data = state.get("applicant_data", {})

    try:
        eligibility_result = run_eligibility_assessment(applicant_data)
        trace.append({
            "agent": "EligibilityAssessmentAgent",
            "action": "COMPLETE",
            "recommendation": eligibility_result.get("recommendation"),
            "eligibility_score": eligibility_result.get("eligibility_score"),
        })

        return {
            "eligibility_result": eligibility_result,
            "agent_trace": trace,
            "current_step": "enablement",
        }
    except Exception as e:
        trace.append({"agent": "EligibilityAssessmentAgent", "action": "ERROR", "error": str(e)})
        return {
            "eligibility_result": {"recommendation": "REVIEW", "error": str(e)},
            "agent_trace": trace,
            "current_step": "enablement",
        }


def enablement_node(state: WorkflowState) -> dict:
    """Node 4: Generate economic enablement recommendations."""
    trace = state.get("agent_trace", [])
    trace.append({"agent": "EnablementRecommenderAgent", "action": "START", "step": "enablement"})

    applicant_data = state.get("applicant_data", {})
    eligibility_result = state.get("eligibility_result", {})

    try:
        enablement_result = generate_recommendations(applicant_data, eligibility_result)
        trace.append({
            "agent": "EnablementRecommenderAgent",
            "action": "COMPLETE",
            "programs_recommended": len(enablement_result.get("recommended_programs", [])),
        })

        return {
            "enablement_result": enablement_result,
            "agent_trace": trace,
            "current_step": "final_decision",
        }
    except Exception as e:
        trace.append({"agent": "EnablementRecommenderAgent", "action": "ERROR", "error": str(e)})
        return {
            "enablement_result": {"recommended_programs": [], "error": str(e)},
            "agent_trace": trace,
            "current_step": "final_decision",
        }


def final_decision_node(state: WorkflowState) -> dict:
    """Node 5: Compile final decision from all agent outputs."""
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
        "enablement_recommendations": enablement.get("recommended_programs", []),
        "career_pathway": enablement.get("career_pathway", ""),
        "immediate_actions": enablement.get("immediate_actions", []),
        "long_term_plan": enablement.get("long_term_plan", ""),
    }

    # If critical validation flags exist, override to manual review
    if validation.get("flags") and len(validation["flags"]) >= 2:
        final_decision["recommendation"] = "MANUAL_REVIEW"
        final_decision["reasoning"] += (
            "\n\nNOTE: Multiple data discrepancies detected. "
            "This application has been flagged for manual review."
        )

    trace.append({"agent": "MasterOrchestrator", "action": "DECISION_COMPLETE",
                  "recommendation": final_decision["recommendation"]})

    return {
        "final_decision": final_decision,
        "agent_trace": trace,
        "current_step": "complete",
    }


# Build the LangGraph Workflow

def build_workflow() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(WorkflowState)

    # Add nodes (node names must not clash with state keys)
    workflow.add_node("step_document_processing", document_processing_node)
    workflow.add_node("step_validation", validation_node)
    workflow.add_node("step_eligibility", eligibility_node)
    workflow.add_node("step_enablement", enablement_node)
    workflow.add_node("step_final_decision", final_decision_node)

    # Define edges (sequential pipeline)
    workflow.set_entry_point("step_document_processing")
    workflow.add_edge("step_document_processing", "step_validation")
    workflow.add_edge("step_validation", "step_eligibility")
    workflow.add_edge("step_eligibility", "step_enablement")
    workflow.add_edge("step_enablement", "step_final_decision")
    workflow.add_edge("step_final_decision", END)

    return workflow.compile()


# Compiled workflow instance
app_workflow = build_workflow()


def run_application_workflow(applicant_data: dict, documents: list[dict] = None) -> dict:
    """Execute the full application assessment workflow.

    Args:
        applicant_data: Applicant form data
        documents: List of {"doc_type": str, "file_path": str}

    Returns:
        Final decision with all agent outputs
    """
    initial_state = {
        "applicant_data": applicant_data,
        "documents": documents or [],
        "extracted_docs": {},
        "validation_result": {},
        "eligibility_result": {},
        "enablement_result": {},
        "final_decision": {},
        "agent_trace": [{"agent": "MasterOrchestrator", "action": "WORKFLOW_START"}],
        "current_step": "document_processing",
        "error": "",
    }

    result = app_workflow.invoke(initial_state)
    return result


# Chat Handler

def handle_chat_message(message: str, applicant_data: dict = None, decision: dict = None) -> str:
    """Handle interactive chat messages about an application.

    Uses ReAct-style reasoning to determine which agent/action to invoke.
    """
    context = ""
    if applicant_data:
        context += f"\nApplicant: {applicant_data.get('full_name', 'Unknown')}"
        context += f"\nEmployment: {applicant_data.get('employment_status', 'Unknown')}"
        context += f"\nIncome: AED {applicant_data.get('monthly_income', 0):,.0f}"
        context += f"\nFamily Size: {applicant_data.get('family_size', 0)}"

    if decision:
        context += f"\nDecision: {decision.get('recommendation', 'Pending')}"
        context += f"\nEligibility Score: {decision.get('eligibility_score', 'N/A')}"
        context += f"\nSupport Tier: {decision.get('support_tier', 'N/A')}"

    prompt = f"""You are a helpful assistant for a government social support program.
You help applicants understand the application process, their assessment results,
and available support programs.

{f"APPLICANT CONTEXT:{context}" if context else "No applicant context available."}

USER MESSAGE: {message}

Provide a helpful, empathetic, and informative response.
If asked about the application status, reference the available context.
If asked about programs, recommend based on the applicant's profile.
Keep responses concise but thorough."""

    try:
        return invoke_llm(prompt)
    except Exception as e:
        return f"I apologize, I'm having trouble processing your request right now. Please try again. (Error: {str(e)})"
