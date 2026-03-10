"""
Tests for app/services/ml_classifier.py

Verifies that:
- GradientBoosting model trains successfully on synthetic data
- Component scores are in [0, 100]
- Tier thresholds produce correct recommendations
- Feature engineering produces the right shape
"""
import pytest
import numpy as np

# ── Fixtures ──────────────────────────────────────────────────────────────────

HIGH_NEED_APPLICANT = {
    "full_name": "Test Applicant",
    "employment_status": "Unemployed",
    "monthly_income": 500.0,
    "family_size": 6,
    "dependents": 4,
    "age": 28,
    "gender": "Female",
    "marital_status": "Divorced",
    "education_level": "High School",
    "total_assets": 5000.0,
    "total_liabilities": 3000.0,
    "years_of_experience": 1.0,
}

LOW_NEED_APPLICANT = {
    "full_name": "Test Applicant 2",
    "employment_status": "Employed",
    "monthly_income": 25000.0,
    "family_size": 2,
    "dependents": 0,
    "age": 35,
    "gender": "Male",
    "marital_status": "Married",
    "education_level": "Master",
    "total_assets": 500000.0,
    "total_liabilities": 50000.0,
    "years_of_experience": 10.0,
}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_component_scores_are_in_range():
    from app.services.ml_classifier import _compute_component_scores
    for applicant in (HIGH_NEED_APPLICANT, LOW_NEED_APPLICANT):
        scores = _compute_component_scores(applicant)
        for key in ("income_score", "employment_score", "family_score", "wealth_score", "demographic_score", "eligibility_score"):
            assert 0 <= scores[key] <= 100, f"{key} out of range: {scores[key]}"


def test_high_need_scores_higher_than_low_need():
    from app.services.ml_classifier import _compute_component_scores
    high = _compute_component_scores(HIGH_NEED_APPLICANT)
    low = _compute_component_scores(LOW_NEED_APPLICANT)
    assert high["eligibility_score"] > low["eligibility_score"], (
        f"High-need score {high['eligibility_score']} should exceed low-need {low['eligibility_score']}"
    )


def test_unemployed_gets_max_employment_score():
    from app.services.ml_classifier import _compute_component_scores
    scores = _compute_component_scores(HIGH_NEED_APPLICANT)
    assert scores["employment_score"] == 90, "Unemployed should score 90 on employment"


def test_employed_gets_low_employment_score():
    from app.services.ml_classifier import _compute_component_scores
    scores = _compute_component_scores(LOW_NEED_APPLICANT)
    assert scores["employment_score"] == 20, "Employed should score 20 on employment"


def test_model_trains_and_predicts():
    from app.services.ml_classifier import train_model, predict_eligibility
    training_data = []
    for i in range(50):
        training_data.append({
            **HIGH_NEED_APPLICANT,
            "emirates_id": f"784-1990-{i:07d}-1",
            "eligible": True,
        })
        training_data.append({
            **LOW_NEED_APPLICANT,
            "emirates_id": f"784-1985-{i:07d}-2",
            "eligible": False,
        })
    train_model(training_data)
    result = predict_eligibility(HIGH_NEED_APPLICANT)
    assert "eligibility_score" in result
    assert 0 <= result["eligibility_score"] <= 100
    assert "income_score" in result
    assert result.get("confidence", 0) > 0


def test_feature_vector_shape():
    from app.services.ml_classifier import _build_features
    features = _build_features(HIGH_NEED_APPLICANT)
    assert features.shape == (1, 14), f"Expected (1, 14), got {features.shape}"


def test_tier_thresholds():
    from app.agents.eligibility_agent import determine_support_tier
    assert determine_support_tier.invoke({"eligibility_score": 85})["recommendation"] == "APPROVE"
    assert determine_support_tier.invoke({"eligibility_score": 60})["recommendation"] == "APPROVE"
    assert determine_support_tier.invoke({"eligibility_score": 45})["recommendation"] == "SOFT_DECLINE"
    assert determine_support_tier.invoke({"eligibility_score": 30})["recommendation"] == "SOFT_DECLINE"
    assert "Tier 1" in determine_support_tier.invoke({"eligibility_score": 85})["support_tier"]
    assert "Tier 2" in determine_support_tier.invoke({"eligibility_score": 60})["support_tier"]
