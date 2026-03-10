"""
Eligibility Assessment Agent — ReAct Reasoning + ML

ReAct loop:
  Thought → "I need to compute component scores using the ML classifier"
  Action  → run_ml_classifier(applicant_data)
  Obs     → {score breakdown from GradientBoosting}
  Thought → "Scores computed. Now determine tier and generate reasoning"
  Action  → determine_support_tier(eligibility_score)
  Obs     → {tier, recommendation}
  Thought → "Ready to generate human-readable reasoning"
  Action  → generate_reasoning(profile, scores)
  Obs     → {reasoning text}
  Final   → {complete eligibility result}

ML Algorithm: GradientBoostingClassifier (sklearn)
  - Chosen over RandomForest: better accuracy on tabular imbalanced data
  - Chosen over LogisticRegression: captures non-linear feature interactions
  - Chosen over SVM: faster inference, probability calibration built-in
  - 14 engineered features: income, family, employment, wealth, demographic
"""
import logging
from langchain_core.tools import tool
from app.services.ml_classifier import predict_eligibility
from app.services.llm_service import invoke_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def run_ml_classifier(applicant_data: dict) -> dict:
    """Run the GradientBoosting ML classifier to predict eligibility and compute component scores."""
    return predict_eligibility(applicant_data)


@tool
def determine_support_tier(eligibility_score: float) -> dict:
    """Determine the support tier and recommendation based on the eligibility score."""
    if eligibility_score >= 80:
        return {"recommendation": "APPROVE", "support_tier": "Tier 1 - Emergency Support",
                "description": "High-need case: immediate financial and social support required"}
    elif eligibility_score >= 55:
        return {"recommendation": "APPROVE", "support_tier": "Tier 2 - Standard Support",
                "description": "Moderate need: standard social support package"}
    elif eligibility_score >= 40:
        return {"recommendation": "SOFT_DECLINE", "support_tier": "Tier 3 - Enablement Only",
                "description": "Low-moderate need: economic enablement programs recommended"}
    else:
        return {"recommendation": "SOFT_DECLINE", "support_tier": "Tier 4 - Self-Sufficient",
                "description": "Applicant appears self-sufficient; monitor for future changes"}


# ---------------------------------------------------------------------------
# ReAct loop for eligibility
# ---------------------------------------------------------------------------

def run_eligibility_assessment(applicant_data: dict) -> dict:
    """Run eligibility assessment using ReAct reasoning loop."""
    react_trace = []

    # Step 1: Thought — prepare for ML scoring
    react_trace.append({
        "type": "thought",
        "content": (
            f"I need to assess eligibility for {applicant_data.get('full_name', 'applicant')}. "
            f"Employment: {applicant_data.get('employment_status')}, "
            f"Income: AED {applicant_data.get('monthly_income', 0):,.0f}, "
            f"Family size: {applicant_data.get('family_size', 1)}. "
            "I will first run the GradientBoosting ML classifier to get component scores."
        ),
    })

    # Step 2: Action — run ML classifier
    react_trace.append({
        "type": "action",
        "tool": "run_ml_classifier",
        "input": {"employment": applicant_data.get("employment_status"),
                  "income": applicant_data.get("monthly_income"),
                  "family_size": applicant_data.get("family_size")},
    })

    ml_result = predict_eligibility(applicant_data)
    eligibility_score = ml_result.get("eligibility_score", 0)

    react_trace.append({
        "type": "observation",
        "tool": "run_ml_classifier",
        "status": "success",
        "content": {
            "eligibility_score": eligibility_score,
            "income_score": ml_result.get("income_score"),
            "employment_score": ml_result.get("employment_score"),
            "wealth_score": ml_result.get("wealth_score"),
            "family_score": ml_result.get("family_score"),
            "demographic_score": ml_result.get("demographic_score"),
            "ml_prediction": ml_result.get("overall_eligible"),
            "confidence": ml_result.get("confidence"),
        },
    })

    # Step 3: Thought — interpret scores
    react_trace.append({
        "type": "thought",
        "content": (
            f"ML classifier returned eligibility score {eligibility_score:.1f}/100. "
            f"Dominant factors: Income ({ml_result.get('income_score', 0):.0f}), "
            f"Employment ({ml_result.get('employment_score', 0):.0f}), "
            f"Wealth ({ml_result.get('wealth_score', 0):.0f}). "
            "Now I will determine the support tier and recommendation."
        ),
    })

    # Step 4: Action — determine tier
    react_trace.append({
        "type": "action",
        "tool": "determine_support_tier",
        "input": {"eligibility_score": eligibility_score},
    })

    tier_result = determine_support_tier.invoke({"eligibility_score": eligibility_score})
    recommendation = tier_result["recommendation"]
    support_tier = tier_result["support_tier"]

    react_trace.append({
        "type": "observation",
        "tool": "determine_support_tier",
        "status": "success",
        "content": tier_result,
    })

    # Step 5: Thought — generate reasoning
    react_trace.append({
        "type": "thought",
        "content": (
            f"Tier determined: {support_tier} ({recommendation}). "
            "Now generating human-readable reasoning using LLM to explain the decision transparently."
        ),
    })

    # Step 6: Action — generate reasoning
    reasoning = _generate_reasoning(applicant_data, ml_result, recommendation, support_tier)
    react_trace.append({
        "type": "observation",
        "tool": "generate_reasoning",
        "status": "success",
        "content": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning,
    })

    react_trace.append({
        "type": "thought",
        "content": f"Assessment complete. Final recommendation: {recommendation} ({support_tier}).",
    })

    return {
        "recommendation": recommendation,
        "support_tier": support_tier,
        "confidence_score": ml_result.get("confidence", 0),
        "eligibility_score": eligibility_score,
        "income_score": ml_result.get("income_score", 0),
        "employment_score": ml_result.get("employment_score", 0),
        "family_score": ml_result.get("family_score", 0),
        "wealth_score": ml_result.get("wealth_score", 0),
        "demographic_score": ml_result.get("demographic_score", 0),
        "reasoning": reasoning,
        "react_trace": react_trace,  # consumed by orchestrator node
    }


def _generate_reasoning(applicant_data: dict, scores: dict, recommendation: str, tier: str) -> str:
    prompt = f"""As an eligibility assessment specialist for a government social support program,
provide a clear, professional 3-4 sentence reasoning for this decision.

APPLICANT PROFILE:
- Name: {applicant_data.get('full_name', 'Unknown')}
- Age: {applicant_data.get('age', 'Unknown')} | Employment: {applicant_data.get('employment_status', 'Unknown')}
- Monthly Income: AED {applicant_data.get('monthly_income', 0):,.0f}
- Family Size: {applicant_data.get('family_size', 0)} | Dependents: {applicant_data.get('dependents', 0)}
- Education: {applicant_data.get('education_level', 'Unknown')}
- Total Assets: AED {applicant_data.get('total_assets', 0):,.0f}
- Total Liabilities: AED {applicant_data.get('total_liabilities', 0):,.0f}

ASSESSMENT SCORES (0-100, higher = greater need):
- Overall Eligibility: {scores.get('eligibility_score', 0):.1f}/100
- Income: {scores.get('income_score', 0):.1f}/100 (weight 30%)
- Employment: {scores.get('employment_score', 0):.1f}/100 (weight 25%)
- Wealth: {scores.get('wealth_score', 0):.1f}/100 (weight 20%)
- Family: {scores.get('family_score', 0):.1f}/100 (weight 15%)
- Demographic: {scores.get('demographic_score', 0):.1f}/100 (weight 10%)

DECISION: {recommendation} — {tier}

Explain which factors most influenced the decision. Be empathetic but objective."""

    try:
        return invoke_llm(prompt, name="eligibility_reasoning")
    except Exception:
        return (
            f"Based on comprehensive assessment, eligibility score is {scores.get('eligibility_score', 0):.1f}/100. "
            f"Decision: {recommendation} ({tier}). "
            f"Primary drivers: income adequacy ({scores.get('income_score', 0):.0f}/100), "
            f"employment status ({scores.get('employment_score', 0):.0f}/100)."
        )


ELIGIBILITY_AGENT_TOOLS = [run_ml_classifier, determine_support_tier]
