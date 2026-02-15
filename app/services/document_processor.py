import json
import pandas as pd
from pathlib import Path
from app.services.llm_service import invoke_llm, extract_json_from_response


def extract_bank_statement(file_path: str) -> dict:
    """Extract financial data from a bank statement (CSV/Excel)."""
    path = Path(file_path)
    if path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        return _extract_bank_statement_via_llm(file_path)

    data = df.to_string(index=False, max_rows=50)
    prompt = f"""Analyze this bank statement data and extract the following in JSON format:
{{
    "average_monthly_balance": <float>,
    "total_credits_last_3_months": <float>,
    "total_debits_last_3_months": <float>,
    "salary_detected": <bool>,
    "estimated_monthly_income": <float>,
    "irregular_large_transactions": <int>,
    "summary": "<brief summary>"
}}

Bank statement data:
{data}

Return ONLY the JSON, no other text."""

    response = invoke_llm(prompt)
    return extract_json_from_response(response)


def _extract_bank_statement_via_llm(file_path: str) -> dict:
    with open(file_path, "r", errors="ignore") as f:
        content = f.read()[:3000]
    prompt = f"""Extract financial information from this bank statement text as JSON:
{{
    "average_monthly_balance": <float>,
    "total_credits_last_3_months": <float>,
    "total_debits_last_3_months": <float>,
    "salary_detected": <bool>,
    "estimated_monthly_income": <float>,
    "irregular_large_transactions": <int>,
    "summary": "<brief summary>"
}}

Text:
{content}

Return ONLY the JSON."""
    response = invoke_llm(prompt)
    return extract_json_from_response(response)


def extract_emirates_id(file_path: str) -> dict:
    """Extract data from Emirates ID image or text file."""
    path = Path(file_path)
    if path.suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        return _extract_emirates_id_from_image(file_path)
    else:
        with open(file_path, "r", errors="ignore") as f:
            content = f.read()[:2000]
        return _extract_emirates_id_from_text(content)


def _extract_emirates_id_from_image(file_path: str) -> dict:
    """Use LLM vision to extract Emirates ID data from image."""
    try:
        from app.services.llm_service import invoke_llm_with_image
        prompt = """Extract all information from this Emirates ID card image as JSON:
{
    "id_number": "<string>",
    "full_name": "<string>",
    "nationality": "<string>",
    "date_of_birth": "<string>",
    "gender": "<string>",
    "expiry_date": "<string>",
    "card_number": "<string>"
}
Return ONLY the JSON."""
        response = invoke_llm_with_image(prompt, file_path)
        return extract_json_from_response(response)
    except Exception:
        return {"error": "Vision extraction not available", "id_number": "", "full_name": ""}


def _extract_emirates_id_from_text(content: str) -> dict:
    prompt = f"""Extract Emirates ID information from this text as JSON:
{{
    "id_number": "<string>",
    "full_name": "<string>",
    "nationality": "<string>",
    "date_of_birth": "<string>",
    "gender": "<string>",
    "expiry_date": "<string>"
}}

Text:
{content}

Return ONLY the JSON."""
    response = invoke_llm(prompt)
    return extract_json_from_response(response)


def extract_resume(file_path: str) -> dict:
    """Extract career information from resume."""
    with open(file_path, "r", errors="ignore") as f:
        content = f.read()[:4000]

    prompt = f"""Analyze this resume and extract the following as JSON:
{{
    "years_of_experience": <float>,
    "highest_education": "<string>",
    "skills": ["<skill1>", "<skill2>"],
    "last_job_title": "<string>",
    "employment_gaps": <bool>,
    "industry": "<string>",
    "certifications": ["<cert1>"],
    "summary": "<brief summary>"
}}

Resume:
{content}

Return ONLY the JSON."""
    response = invoke_llm(prompt)
    return extract_json_from_response(response)


def extract_assets_liabilities(file_path: str) -> dict:
    """Extract asset and liability data from Excel file."""
    path = Path(file_path)
    if path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        with open(file_path, "r", errors="ignore") as f:
            content = f.read()[:3000]
        df = None

    if df is not None:
        data = df.to_string(index=False)
    else:
        data = content

    prompt = f"""Analyze this assets and liabilities statement and extract as JSON:
{{
    "total_assets": <float>,
    "total_liabilities": <float>,
    "net_worth": <float>,
    "asset_breakdown": {{"property": <float>, "vehicles": <float>, "savings": <float>, "investments": <float>, "other": <float>}},
    "liability_breakdown": {{"mortgage": <float>, "loans": <float>, "credit_cards": <float>, "other": <float>}},
    "debt_to_asset_ratio": <float>,
    "summary": "<brief summary>"
}}

Data:
{data}

Return ONLY the JSON."""
    response = invoke_llm(prompt)
    return extract_json_from_response(response)


def extract_credit_report(file_path: str) -> dict:
    """Extract credit information from credit report."""
    with open(file_path, "r", errors="ignore") as f:
        content = f.read()[:4000]

    prompt = f"""Analyze this credit report and extract as JSON:
{{
    "credit_score": <int>,
    "total_open_accounts": <int>,
    "total_closed_accounts": <int>,
    "payment_history_rating": "<good/fair/poor>",
    "outstanding_debt": <float>,
    "credit_utilization_pct": <float>,
    "defaults_or_late_payments": <int>,
    "summary": "<brief summary>"
}}

Credit Report:
{content}

Return ONLY the JSON."""
    response = invoke_llm(prompt)
    return extract_json_from_response(response)


EXTRACTORS = {
    "bank_statement": extract_bank_statement,
    "emirates_id": extract_emirates_id,
    "resume": extract_resume,
    "assets_liabilities": extract_assets_liabilities,
    "credit_report": extract_credit_report,
}


def process_document(doc_type: str, file_path: str) -> dict:
    """Process a document based on its type."""
    extractor = EXTRACTORS.get(doc_type)
    if not extractor:
        return {"error": f"Unknown document type: {doc_type}"}
    return extractor(file_path)
