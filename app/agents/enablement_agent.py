"""
Enablement Recommender Agent
Responsible for recommending appropriate economic enablement programs
based on applicant profile using RAG over policy documents.
"""
from langchain_core.tools import tool
from app.services.vector_store import query_policies
from app.services.llm_service import invoke_llm, extract_json_from_response


@tool
def recommend_enablement_programs(applicant_data: dict, eligibility_result: dict) -> dict:
    """Recommend economic enablement programs based on applicant profile.

    Args:
        applicant_data: The applicant's validated data
        eligibility_result: The eligibility assessment results

    Returns:
        List of recommended programs with reasoning
    """
    return generate_recommendations(applicant_data, eligibility_result)


def generate_recommendations(applicant_data: dict, eligibility_result: dict) -> dict:
    """Generate enablement recommendations using RAG + LLM."""

    # Build query for RAG
    profile_summary = (
        f"{applicant_data.get('employment_status', 'Unknown')} "
        f"{applicant_data.get('gender', 'Unknown')} applicant, "
        f"age {applicant_data.get('age', 'Unknown')}, "
        f"education: {applicant_data.get('education_level', 'Unknown')}, "
        f"family size: {applicant_data.get('family_size', 'Unknown')}, "
        f"income: AED {applicant_data.get('monthly_income', 0):,.0f}"
    )

    # Query vector store for relevant programs
    rag_query = f"What enablement programs are suitable for a {profile_summary}?"
    try:
        policy_context = query_policies(rag_query, top_k=5)
    except Exception:
        policy_context = "No policy documents available."

    # Use LLM to generate tailored recommendations
    prompt = f"""As an Economic Enablement Advisor for a government social support program,
recommend specific programs for this applicant.

APPLICANT PROFILE:
- Name: {applicant_data.get('full_name', 'Unknown')}
- Age: {applicant_data.get('age', 'Unknown')}
- Gender: {applicant_data.get('gender', 'Unknown')}
- Employment Status: {applicant_data.get('employment_status', 'Unknown')}
- Education: {applicant_data.get('education_level', 'Unknown')}
- Monthly Income: AED {applicant_data.get('monthly_income', 0):,.0f}
- Family Size: {applicant_data.get('family_size', 0)}
- Dependents: {applicant_data.get('dependents', 0)}
- Years of Experience: {applicant_data.get('years_of_experience', 0)}
- Support Tier: {eligibility_result.get('support_tier', 'Unknown')}

AVAILABLE PROGRAMS FROM POLICY:
{policy_context}

Based on the applicant's profile and available programs, provide your recommendation as JSON:
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
    "long_term_plan": "<brief long-term enablement strategy>"
}}

Return ONLY the JSON."""

    try:
        response = invoke_llm(prompt)
        result = extract_json_from_response(response)
        if result:
            return result
    except Exception:
        pass

    # Fallback rule-based recommendations
    return _fallback_recommendations(applicant_data)


def _fallback_recommendations(applicant_data: dict) -> dict:
    """Generate rule-based recommendations as fallback."""
    programs = []
    emp_status = applicant_data.get("employment_status", "").lower()
    education = applicant_data.get("education_level", "").lower()
    age = applicant_data.get("age", 30)
    gender = applicant_data.get("gender", "").lower()

    if emp_status == "unemployed":
        programs.append({"program_name": "Career Counseling & Job Matching", "priority": "high",
                        "reason": "Immediate job placement support for unemployed applicant"})
        programs.append({"program_name": "Financial Literacy Workshop", "priority": "high",
                        "reason": "Essential financial management during unemployment"})

    if education in ["high school", "diploma"]:
        programs.append({"program_name": "Professional Certification Fast-Track", "priority": "high",
                        "reason": "Upgrade qualifications to improve employment prospects"})

    if age < 25:
        programs.append({"program_name": "Digital Skills Accelerator", "priority": "medium",
                        "reason": "Build foundational digital skills for young job seeker"})

    if emp_status == "self-employed":
        programs.append({"program_name": "Entrepreneurship Launchpad", "priority": "high",
                        "reason": "Business development support for self-employed applicant"})

    if gender == "female" and applicant_data.get("dependents", 0) > 0:
        programs.append({"program_name": "Women Empowerment Initiative", "priority": "high",
                        "reason": "Comprehensive support for women with dependents"})

    if not programs:
        programs.append({"program_name": "Career Counseling & Job Matching", "priority": "medium",
                        "reason": "General career guidance and employment support"})

    return {
        "recommended_programs": programs,
        "career_pathway": "To be determined through career counseling",
        "immediate_actions": ["Register for recommended programs", "Schedule career counseling session"],
        "long_term_plan": "Develop skills and secure stable employment within 6-12 months",
    }


ENABLEMENT_AGENT_TOOLS = [recommend_enablement_programs]

ENABLEMENT_AGENT_PROMPT = """You are an Economic Enablement Recommender Agent for a government social support program.

Your role is to:
1. Analyze the applicant's profile, skills, and circumstances
2. Query available enablement programs from the policy database
3. Recommend the most suitable programs for the applicant
4. Provide a career pathway suggestion
5. Outline immediate actions and long-term development plan

Available program categories:
- Upskilling (digital literacy, certifications)
- Training (vocational, entrepreneurship)
- Job Matching (resume building, direct placement)
- Career Counseling (assessment, gap analysis)

Always recommend programs that match the applicant's profile and maximize their path to self-sufficiency."""
