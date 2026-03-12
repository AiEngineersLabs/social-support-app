"""
Document Processor Service — Multimodal extraction for all document types.

Supported input formats:
  Bank Statement      : CSV, XLSX, PDF
  Emirates ID         : TXT, PNG, JPG, JPEG, PDF
  Resume / CV         : TXT, PDF
  Assets & Liabilities: XLSX, CSV, PDF, TXT
  Credit Report       : TXT, PDF

Processing pipeline:
  Tabular  (CSV/XLSX) → pandas → LLM structuring
  Text     (TXT)      → raw read → LLM structuring
  PDF                 → pdfplumber text extraction → LLM structuring
  Image    (PNG/JPG)  → llava vision OCR (primary) / pytesseract (fallback)
"""
import json
import logging
from pathlib import Path

import pandas as pd

from app.services.llm_service import invoke_llm, invoke_llm_with_image, extract_json_from_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF helper (pdfplumber — primary PDF extraction)
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(file_path: str, max_chars: int = 4000) -> str:
    """Extract plain text from a PDF using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                if sum(len(t) for t in text_parts) > max_chars:
                    break
        return "\n".join(text_parts)[:max_chars]
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s — trying pymupdf", file_path, exc)

    # Fallback: pymupdf (fitz)
    try:
        import fitz  # pymupdf
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                break
        return text[:max_chars]
    except Exception as exc2:
        logger.warning("pymupdf also failed for %s: %s", file_path, exc2)
        return ""


# ---------------------------------------------------------------------------
# Bank Statement
# ---------------------------------------------------------------------------

def extract_bank_statement(file_path: str) -> dict:
    """Extract financial data from a bank statement (CSV, XLSX, PDF)."""
    path = Path(file_path)
    data_str = ""

    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
        data_str = df.to_string(index=False, max_rows=50)
    elif path.suffix == ".csv":
        df = pd.read_csv(file_path)
        data_str = df.to_string(index=False, max_rows=50)
    elif path.suffix == ".pdf":
        data_str = _extract_text_from_pdf(file_path)
    else:
        try:
            with open(file_path, "r", errors="ignore") as f:
                data_str = f.read(3000)
        except Exception:
            data_str = ""

    if not data_str.strip():
        return {"error": "Could not extract content from bank statement", "estimated_monthly_income": 0}

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
{data_str}

Return ONLY the JSON, no other text."""

    response = invoke_llm(prompt, name="bank_statement_extraction")
    return extract_json_from_response(response)


# ---------------------------------------------------------------------------
# Emirates ID
# ---------------------------------------------------------------------------

def extract_emirates_id(file_path: str) -> dict:
    """Extract data from Emirates ID — image (vision OCR) or text/PDF."""
    path = Path(file_path)

    if path.suffix in (".png", ".jpg", ".jpeg", ".webp"):
        return _extract_emirates_id_from_image(file_path)

    # Text or PDF
    if path.suffix == ".pdf":
        content = _extract_text_from_pdf(file_path, max_chars=2000)
    else:
        try:
            with open(file_path, "r", errors="ignore") as f:
                content = f.read(2000)
        except Exception:
            content = ""

    return _extract_emirates_id_from_text(content)


def _extract_emirates_id_from_image(file_path: str) -> dict:
    """Extract Emirates ID data from image using tesseract (primary) + llava (fallback).

    Tesseract is faster and more reliable for ID cards with clear printed text.
    If tesseract produces usable text, we pass it to the text LLM for structuring.
    Otherwise, we fall back to llava vision model.
    """
    structured_prompt = """Extract all information from this Emirates ID card as JSON:
{{
    "id_number": "<string>",
    "full_name": "<string>",
    "nationality": "<string>",
    "date_of_birth": "<string>",
    "gender": "<string>",
    "expiry_date": "<string>",
    "card_number": "<string>"
}}
Return ONLY the JSON."""

    # Strategy 1: Tesseract OCR → text LLM (fast, reliable for printed text)
    try:
        import pytesseract
        from PIL import Image as PILImage
        img = PILImage.open(file_path)
        ocr_text = pytesseract.image_to_string(img)
        if ocr_text and len(ocr_text.strip()) > 10:
            logger.info("Tesseract extracted %d chars from Emirates ID image", len(ocr_text.strip()))
            text_prompt = (
                f"{structured_prompt}\n\nOCR text from the card:\n{ocr_text[:2000]}\n\n"
                "Extract the requested fields from the OCR text above and return ONLY JSON."
            )
            response = invoke_llm(text_prompt, name="emirates_id_tesseract_ocr")
            result = extract_json_from_response(response)
            if result and not result.get("error"):
                return result
    except Exception as tess_err:
        logger.warning("Tesseract OCR failed for Emirates ID: %s", tess_err)

    # Strategy 2: llava vision model (handles poor quality / non-standard layouts)
    try:
        response = invoke_llm_with_image(structured_prompt, file_path)
        result = extract_json_from_response(response)
        if result and not result.get("error"):
            return result
    except Exception as llava_err:
        logger.warning("llava vision OCR failed for Emirates ID: %s", llava_err)

    return {"error": "Could not extract Emirates ID details from image"}


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
    response = invoke_llm(prompt, name="emirates_id_text_extraction")
    return extract_json_from_response(response)


# ---------------------------------------------------------------------------
# Resume / CV
# ---------------------------------------------------------------------------

def extract_resume(file_path: str) -> dict:
    """Extract career information from resume (TXT or PDF)."""
    path = Path(file_path)

    if path.suffix == ".pdf":
        content = _extract_text_from_pdf(file_path, max_chars=4000)
    else:
        try:
            with open(file_path, "r", errors="ignore") as f:
                content = f.read(4000)
        except Exception:
            content = ""

    if not content.strip():
        return {"error": "Could not extract resume content", "years_of_experience": 0}

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
    response = invoke_llm(prompt, name="resume_extraction")
    return extract_json_from_response(response)


# ---------------------------------------------------------------------------
# Assets & Liabilities
# ---------------------------------------------------------------------------

def extract_assets_liabilities(file_path: str) -> dict:
    """Extract asset and liability data from Excel, CSV, PDF, or TXT."""
    path = Path(file_path)
    data_str = ""

    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
        data_str = df.to_string(index=False)
    elif path.suffix == ".csv":
        df = pd.read_csv(file_path)
        data_str = df.to_string(index=False)
    elif path.suffix == ".pdf":
        data_str = _extract_text_from_pdf(file_path, max_chars=3000)
    else:
        try:
            with open(file_path, "r", errors="ignore") as f:
                data_str = f.read(3000)
        except Exception:
            data_str = ""

    if not data_str.strip():
        return {"error": "Could not extract assets/liabilities content", "total_assets": 0, "total_liabilities": 0}

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
{data_str}

Return ONLY the JSON."""
    response = invoke_llm(prompt, name="assets_liabilities_extraction")
    return extract_json_from_response(response)


# ---------------------------------------------------------------------------
# Credit Report
# ---------------------------------------------------------------------------

def extract_credit_report(file_path: str) -> dict:
    """Extract credit information from credit report (TXT or PDF)."""
    path = Path(file_path)

    if path.suffix == ".pdf":
        content = _extract_text_from_pdf(file_path, max_chars=4000)
    else:
        try:
            with open(file_path, "r", errors="ignore") as f:
                content = f.read(4000)
        except Exception:
            content = ""

    if not content.strip():
        return {"error": "Could not extract credit report content", "credit_score": 0}

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
    response = invoke_llm(prompt, name="credit_report_extraction")
    return extract_json_from_response(response)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

EXTRACTORS = {
    "bank_statement": extract_bank_statement,
    "emirates_id": extract_emirates_id,
    "resume": extract_resume,
    "assets_liabilities": extract_assets_liabilities,
    "credit_report": extract_credit_report,
}


def process_document(doc_type: str, file_path: str) -> dict:
    """Process a document based on its type. Supports multimodal input."""
    extractor = EXTRACTORS.get(doc_type)
    if not extractor:
        return {"error": f"Unknown document type: {doc_type}"}
    return extractor(file_path)
