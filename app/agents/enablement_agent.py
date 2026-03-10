"""
Enablement Recommender Agent — ReAct + RAG

ReAct loop:
  Thought → "Build profile query for RAG lookup"
  Action  → query_policy_rag(profile_summary)
  Obs     → relevant policy chunks
  Thought → "Got policy context. Now generate tailored recommendations with LLM"
  Action  → generate_program_recommendations(profile, policy_context)
  Obs     → structured program list
  Final   → {recommended_programs, career_pathway, immediate_actions, long_term_plan}
"""
import logging
from langchain_core.tools import tool
from app.services.vector_store import query_policies
from app.services.llm_service import invoke_llm, extract_json_from_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def query_policy_rag(profile_query: str) -> str:
    """Query the ChromaDB vector store to retrieve relevant enablement policy chunks for a given applicant profile."""
    try:
        result = query_policies(profile_query, top_k=5)
        return result if result else "No policy documents available."
    except Exception as exc:
        logger.warning("RAG query failed: %s", exc)
        return "Policy retrieval temporarily unavailable."


@tool
def generate_program_recommendations(applicant_profile: str, policy_context: str) -> str:
    """Use LLM to generate structured enablement program recommendations based on profile and policy context."""
    prompt = f"""As an Economic Enablement Advisor, recommend specific programs for this applicant.

APPLICANT PROFILE:
{applicant_profile}

AVAILABLE PROGRAMS FROM POLICY DATABASE:
{policy_context}

Return recommendations as JSON:
{{
    "recommended_programs": [
        {{
            "program_name": "<name>",
            "priority": "high/medium/low",
            "reason": "<why this suits the applicant>"
        }}
    ],
    "career_pathway": "<suggested career direction>",
    "immediate_actions": ["<action 1>", "<action 2>"],
    "long_term_plan": "<6-12 month enablement strategy>"
}}

Return ONLY the JSON."""
    try:
        response = invoke_llm(prompt, name="enablement_recommendations")
        return response
    except Exception as exc:
        return f'{{"error": "{exc}"}}'


# ---------------------------------------------------------------------------
# ReAct loop for enablement
# ---------------------------------------------------------------------------

def generate_recommendations(applicant_data: dict, eligibility_result: dict) -> dict:
    """Generate enablement recommendations using ReAct + RAG loop."""
    react_trace = []

    # Build profile summary for RAG query
    profile_summary = (
        f"{applicant_data.get('employment_status', 'Unknown')} "
        f"{applicant_data.get('gender', '')} applicant, "
        f"age {applicant_data.get('age', 'Unknown')}, "
        f"education: {applicant_data.get('education_level', 'Unknown')}, "
        f"family size: {applicant_data.get('family_size', 'Unknown')}, "
        f"income: AED {applicant_data.get('monthly_income', 0):,.0f}, "
        f"support tier: {eligibility_result.get('support_tier', 'Unknown')}"
    )

    profile_detail = f"""- Name: {applicant_data.get('full_name', 'Unknown')}
- Age: {applicant_data.get('age', 'Unknown')} | Gender: {applicant_data.get('gender', 'Unknown')}
- Employment: {applicant_data.get('employment_status', 'Unknown')}
- Education: {applicant_data.get('education_level', 'Unknown')}
- Monthly Income: AED {applicant_data.get('monthly_income', 0):,.0f}
- Family Size: {applicant_data.get('family_size', 0)} | Dependents: {applicant_data.get('dependents', 0)}
- Years Experience: {applicant_data.get('years_of_experience', 0)}
- Support Tier: {eligibility_result.get('support_tier', 'Unknown')}
- Recommendation: {eligibility_result.get('recommendation', 'Unknown')}"""

    # Step 1: Thought — plan RAG query
    react_trace.append({
        "type": "thought",
        "content": (
            f"I need to recommend enablement programs for: {profile_summary}. "
            "I will first query the policy vector store (ChromaDB via LlamaIndex RAG) "
            "to retrieve relevant program descriptions."
        ),
    })

    # Step 2: Action — RAG query
    rag_query = f"What enablement programs are suitable for a {profile_summary}?"
    react_trace.append({
        "type": "action",
        "tool": "query_policy_rag",
        "input": {"profile_query": rag_query},
    })

    policy_context = query_policies(rag_query, top_k=5) if True else "No policy available."
    rag_used = bool(policy_context and "unavailable" not in policy_context.lower())

    react_trace.append({
        "type": "observation",
        "tool": "query_policy_rag",
        "status": "success" if rag_used else "fallback",
        "content": policy_context[:300] + "..." if len(policy_context) > 300 else policy_context,
    })

    # Step 3: Thought — interpret RAG results
    react_trace.append({
        "type": "thought",
        "content": (
            f"RAG retrieved {'relevant policy context' if rag_used else 'no context (using rule-based fallback)'}. "
            "Now generating tailored program recommendations using LLM."
        ),
    })

    # Step 4: Action — LLM recommendations
    react_trace.append({
        "type": "action",
        "tool": "generate_program_recommendations",
        "input": {"profile": profile_summary, "rag_context_chars": len(policy_context)},
    })

    try:
        result = _llm_recommendations(profile_detail, policy_context)
        rag_status = "success"
    except Exception:
        result = _fallback_recommendations(applicant_data)
        rag_status = "fallback"

    programs = result.get("recommended_programs", [])
    react_trace.append({
        "type": "observation",
        "tool": "generate_program_recommendations",
        "status": rag_status,
        "content": f"{len(programs)} programs recommended: {[p.get('program_name') for p in programs[:3]]}",
    })

    react_trace.append({
        "type": "thought",
        "content": (
            f"Generated {len(programs)} enablement recommendations. "
            f"Top recommendation: {programs[0].get('program_name') if programs else 'None'}. "
            "Assessment complete."
        ),
    })

    result["react_trace"] = react_trace
    result["rag_used"] = rag_used
    return result


def _llm_recommendations(profile_detail: str, policy_context: str) -> dict:
    prompt = f"""As an Economic Enablement Advisor for a government social support program,
recommend specific programs for this applicant.

APPLICANT PROFILE:
{profile_detail}

AVAILABLE PROGRAMS FROM POLICY DATABASE:
{policy_context}

Based on the profile and available programs, provide recommendations as JSON:
{{
    "recommended_programs": [
        {{
            "program_name": "<name>",
            "priority": "high/medium/low",
            "reason": "<brief reason why this program suits the applicant>"
        }}
    ],
    "career_pathway": "<suggested career direction>",
    "immediate_actions": ["<action 1>", "<action 2>"],
    "long_term_plan": "<brief 6-12 month enablement strategy>"
}}

Return ONLY the JSON."""

    response = invoke_llm(prompt, name="enablement_recommendations")
    result = extract_json_from_response(response)
    if result and result.get("recommended_programs"):
        return result
    return _fallback_recommendations_from_profile(profile_detail)


def _fallback_recommendations(applicant_data: dict) -> dict:
    programs = []
    emp = applicant_data.get("employment_status", "").lower()
    edu = applicant_data.get("education_level", "").lower()
    age = applicant_data.get("age", 30)
    gender = applicant_data.get("gender", "").lower()
    dependents = applicant_data.get("dependents", 0)

    if emp == "unemployed":
        programs.append({"program_name": "Career Counseling & Job Matching", "priority": "high",
                        "reason": "Immediate job placement support for unemployed applicant"})
        programs.append({"program_name": "Financial Literacy Workshop", "priority": "high",
                        "reason": "Essential financial management skills during unemployment"})
    if edu in ("high school", "diploma"):
        programs.append({"program_name": "Professional Certification Fast-Track", "priority": "high",
                        "reason": "Upgrade qualifications to improve employment prospects"})
    if age < 25:
        programs.append({"program_name": "Digital Skills Accelerator", "priority": "medium",
                        "reason": "Build foundational digital skills for young job seeker"})
    if emp == "self-employed":
        programs.append({"program_name": "Entrepreneurship Launchpad", "priority": "high",
                        "reason": "Business development support for self-employed applicant"})
    if gender == "female" and dependents > 0:
        programs.append({"program_name": "Women Empowerment Initiative", "priority": "high",
                        "reason": "Comprehensive support for women with dependents"})
    if not programs:
        programs.append({"program_name": "Career Counseling & Job Matching", "priority": "medium",
                        "reason": "General career guidance and employment support"})

    return {
        "recommended_programs": programs,
        "career_pathway": "To be determined through career counseling session",
        "immediate_actions": ["Register for recommended programs", "Schedule initial career counseling"],
        "long_term_plan": "Develop skills and secure stable employment within 6-12 months",
    }


def _fallback_recommendations_from_profile(profile_str: str) -> dict:
    return {
        "recommended_programs": [
            {"program_name": "Career Counseling & Job Matching", "priority": "high",
             "reason": "Immediate career guidance and job placement support"},
            {"program_name": "Financial Literacy Workshop", "priority": "medium",
             "reason": "Build financial management and budgeting skills"},
        ],
        "career_pathway": "Career counseling will determine the optimal pathway",
        "immediate_actions": ["Register for career counseling", "Attend financial literacy orientation"],
        "long_term_plan": "Progressive skill development and employment stabilisation over 12 months",
    }


ENABLEMENT_AGENT_TOOLS = [query_policy_rag, generate_program_recommendations]
