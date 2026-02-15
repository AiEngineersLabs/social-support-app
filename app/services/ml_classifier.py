import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eligibility_model.pkl"

_model = None


def _build_features(applicant_data: dict) -> np.ndarray:
    """Build feature vector from applicant data."""
    monthly_income = float(applicant_data.get("monthly_income", 0))
    family_size = int(applicant_data.get("family_size", 1))
    dependents = int(applicant_data.get("dependents", 0))
    years_of_experience = float(applicant_data.get("years_of_experience", 0))
    total_assets = float(applicant_data.get("total_assets", 0))
    total_liabilities = float(applicant_data.get("total_liabilities", 0))
    age = int(applicant_data.get("age", 30))

    # Derived features
    income_per_capita = monthly_income / max(family_size, 1)
    debt_to_income = total_liabilities / max(monthly_income * 12, 1)
    net_worth = total_assets - total_liabilities
    asset_to_liability = total_assets / max(total_liabilities, 1)

    # Employment encoding
    emp_status = applicant_data.get("employment_status", "unemployed").lower()
    emp_encoded = {"employed": 3, "self-employed": 2, "part-time": 1}.get(emp_status, 0)

    # Education encoding
    edu_level = applicant_data.get("education_level", "high school").lower()
    edu_encoded = {
        "phd": 5, "doctorate": 5,
        "master": 4, "masters": 4,
        "bachelor": 3, "bachelors": 3,
        "diploma": 2,
        "high school": 1,
    }.get(edu_level, 1)

    # Marital status encoding
    marital = applicant_data.get("marital_status", "single").lower()
    marital_encoded = {"married": 2, "divorced": 1, "widowed": 1}.get(marital, 0)

    features = [
        monthly_income,
        income_per_capita,
        family_size,
        dependents,
        years_of_experience,
        total_assets,
        total_liabilities,
        net_worth,
        debt_to_income,
        asset_to_liability,
        age,
        emp_encoded,
        edu_encoded,
        marital_encoded,
    ]
    return np.array(features).reshape(1, -1)


def train_model(training_data: list[dict]):
    """Train the eligibility classifier on synthetic data."""
    X_list = []
    y_list = []
    for record in training_data:
        features = _build_features(record)
        X_list.append(features.flatten())
        y_list.append(record["eligible"])

    X = np.array(X_list)
    y = np.array(y_list)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )),
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)

    global _model
    _model = pipeline
    return pipeline


def load_model():
    global _model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_eligibility(applicant_data: dict) -> dict:
    """Predict eligibility and return detailed scores."""
    global _model
    if _model is None:
        load_model()
    if _model is None:
        return _rule_based_assessment(applicant_data)

    features = _build_features(applicant_data)
    prediction = _model.predict(features)[0]
    probabilities = _model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    # Component scores (normalized 0-100)
    scores = _compute_component_scores(applicant_data)
    scores["overall_eligible"] = bool(prediction)
    scores["confidence"] = round(confidence * 100, 1)

    return scores


def _compute_component_scores(data: dict) -> dict:
    """Compute individual component scores for transparency."""
    monthly_income = float(data.get("monthly_income", 0))
    family_size = int(data.get("family_size", 1))
    dependents = int(data.get("dependents", 0))
    years_exp = float(data.get("years_of_experience", 0))
    total_assets = float(data.get("total_assets", 0))
    total_liabilities = float(data.get("total_liabilities", 0))
    age = int(data.get("age", 30))

    # Income score (lower income = higher need = higher score)
    income_per_capita = monthly_income / max(family_size, 1)
    if income_per_capita < 2000:
        income_score = 90
    elif income_per_capita < 5000:
        income_score = 70
    elif income_per_capita < 10000:
        income_score = 40
    else:
        income_score = 15

    # Employment score (unemployed = higher need)
    emp = data.get("employment_status", "unemployed").lower()
    emp_scores = {"unemployed": 90, "part-time": 65, "self-employed": 40, "employed": 20}
    employment_score = emp_scores.get(emp, 50)

    # Family score (larger family = more need)
    family_score = min(90, 30 + (family_size * 8) + (dependents * 5))

    # Wealth score (lower net worth = higher need)
    net_worth = total_assets - total_liabilities
    if net_worth < 10000:
        wealth_score = 90
    elif net_worth < 50000:
        wealth_score = 65
    elif net_worth < 200000:
        wealth_score = 35
    else:
        wealth_score = 10

    # Demographic score
    demo_score = 50
    if age < 25 or age > 55:
        demo_score += 15
    if dependents >= 3:
        demo_score += 10
    demo_score = min(demo_score, 100)

    # Overall eligibility score (weighted average)
    eligibility_score = (
        income_score * 0.30
        + employment_score * 0.25
        + family_score * 0.15
        + wealth_score * 0.20
        + demo_score * 0.10
    )

    return {
        "income_score": round(income_score, 1),
        "employment_score": round(employment_score, 1),
        "family_score": round(family_score, 1),
        "wealth_score": round(wealth_score, 1),
        "demographic_score": round(demo_score, 1),
        "eligibility_score": round(eligibility_score, 1),
    }


def _rule_based_assessment(data: dict) -> dict:
    """Fallback rule-based assessment if ML model is not available."""
    scores = _compute_component_scores(data)
    scores["overall_eligible"] = scores["eligibility_score"] >= 55
    scores["confidence"] = 75.0
    return scores
