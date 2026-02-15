"""
Eligibility Assessment Agent
Responsible for scoring applicants on eligibility criteria and
providing approve/soft-decline recommendations using ML + LLM reasoning.
"""
from langchain_core.tools import tool
from app.services.ml_classifier import predict_eligibility
from app.services.llm_service import invoke_llm, extract_json_from_response


@tool
def assess_eligibility(applicant_data: dict) -> dict:
    """Assess applicant eligibility using ML classifier and rule-based scoring.

    Args:
        applicant_data: Validated applicant data with all financial information

    Returns:
        Eligibility assessment with scores and recommendation
    """
    return run_eligibility_assessment(applicant_data)


def run_eligibility_assessment(applicant_data: dict) -> dict:
    """Run complete eligibility assessment."""
    # Get ML-based prediction and component scores
    ml_result = predict_eligibility(applicant_data)

    eligibility_score = ml_result.get("eligibility_score", 0)
    is_eligible = ml_result.get("overall_eligible", False)
    confidence = ml_result.get("confidence", 0)

    # Determine recommendation
    if eligibility_score >= 80:
        recommendation = "APPROVE"
        support_tier = "Tier 1 - Emergency Support"
    elif eligibility_score >= 55:
        recommendation = "APPROVE"
        support_tier = "Tier 2 - Standard Support"
    elif eligibility_score >= 40:
        recommendation = "SOFT_DECLINE"
        support_tier = "Tier 3 - Enablement Only"
    else:
        recommendation = "SOFT_DECLINE"
        support_tier = "Tier 4 - Self-Sufficient"

    # Generate LLM reasoning
    reasoning = _generate_reasoning(applicant_data, ml_result, recommendation, support_tier)

    return {
        "recommendation": recommendation,
        "support_tier": support_tier,
        "confidence_score": confidence,
        "eligibility_score": eligibility_score,
        "income_score": ml_result.get("income_score", 0),
        "employment_score": ml_result.get("employment_score", 0),
        "family_score": ml_result.get("family_score", 0),
        "wealth_score": ml_result.get("wealth_score", 0),
        "demographic_score": ml_result.get("demographic_score", 0),
        "reasoning": reasoning,
    }


def _generate_reasoning(applicant_data: dict, scores: dict, recommendation: str, tier: str) -> str:
    """Use LLM to generate human-readable reasoning for the decision."""
    prompt = f"""As an eligibility assessment specialist for a government social support program,
provide a clear, professional reasoning for the following assessment decision.

APPLICANT PROFILE:
- Name: {applicant_data.get('full_name', 'Unknown')}
- Age: {applicant_data.get('age', 'Unknown')}
- Employment: {applicant_data.get('employment_status', 'Unknown')}
- Monthly Income: AED {applicant_data.get('monthly_income', 0):,.0f}
- Family Size: {applicant_data.get('family_size', 0)}
- Dependents: {applicant_data.get('dependents', 0)}
- Education: {applicant_data.get('education_level', 'Unknown')}
- Total Assets: AED {applicant_data.get('total_assets', 0):,.0f}
- Total Liabilities: AED {applicant_data.get('total_liabilities', 0):,.0f}

ASSESSMENT SCORES:
- Overall Eligibility: {scores.get('eligibility_score', 0):.1f}/100
- Income Score: {scores.get('income_score', 0):.1f}/100 (weight: 30%)
- Employment Score: {scores.get('employment_score', 0):.1f}/100 (weight: 25%)
- Wealth Score: {scores.get('wealth_score', 0):.1f}/100 (weight: 20%)
- Family Score: {scores.get('family_score', 0):.1f}/100 (weight: 15%)
- Demographic Score: {scores.get('demographic_score', 0):.1f}/100 (weight: 10%)

DECISION: {recommendation}
SUPPORT TIER: {tier}

Provide a 3-4 sentence professional reasoning for this decision.
Explain which factors most influenced the decision and why.
Be empathetic but objective."""

    try:
        return invoke_llm(prompt)
    except Exception:
        return (
            f"Based on the comprehensive assessment, the applicant receives an overall "
            f"eligibility score of {scores.get('eligibility_score', 0):.1f}/100. "
            f"The recommendation is {recommendation} ({tier}). "
            f"Key factors: income adequacy ({scores.get('income_score', 0):.0f}/100), "
            f"employment status ({scores.get('employment_score', 0):.0f}/100), "
            f"and wealth position ({scores.get('wealth_score', 0):.0f}/100)."
        )


ELIGIBILITY_AGENT_TOOLS = [assess_eligibility]

ELIGIBILITY_AGENT_PROMPT = """You are an Eligibility Assessment Agent for a government social support program.

Your role is to:
1. Evaluate applicant eligibility based on validated data
2. Score applicants across 5 dimensions: income, employment, family, wealth, demographics
3. Provide a clear APPROVE or SOFT_DECLINE recommendation
4. Assign the appropriate support tier
5. Generate transparent reasoning for the decision

Decision Thresholds:
- Score 80+: APPROVE (Tier 1 - Emergency Support)
- Score 55-79: APPROVE (Tier 2 - Standard Support)
- Score 40-54: SOFT_DECLINE (Tier 3 - Enablement Only)
- Score below 40: SOFT_DECLINE (Tier 4 - Self-Sufficient)

Always be fair, transparent, and objective in your assessments."""
