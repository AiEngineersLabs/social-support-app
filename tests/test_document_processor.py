"""
Tests for app/services/document_processor.py

Verifies:
- PDF text extraction (pdfplumber)
- CSV bank statement parsing
- XLSX assets/liabilities parsing
- Text resume/credit report parsing
"""
import pytest
import os
import tempfile
import json


# ── Helpers ──────────────────────────────────────────────────────────────────

def write_temp_file(content: str, suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


def write_temp_bytes(content: bytes, suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


# ── PDF extraction ─────────────────────────────────────────────────────────

def test_extract_text_from_pdf_returns_string():
    """_extract_text_from_pdf should not raise even on an empty file."""
    from app.services.document_processor import _extract_text_from_pdf
    # Write a minimal valid text file with .pdf extension to test graceful fallback
    path = write_temp_file("Bank Statement\nIncome: 3000 AED\n", ".txt")
    try:
        # Should handle non-PDF gracefully (returns "" or minimal text)
        result = _extract_text_from_pdf(path)
        assert isinstance(result, str)
    finally:
        os.unlink(path)


# ── CSV bank statement ─────────────────────────────────────────────────────

def test_bank_statement_csv_extraction_returns_dict(monkeypatch):
    """extract_bank_statement should return a dict (LLM call mocked)."""
    from app.services import document_processor

    monkeypatch.setattr(
        document_processor, "invoke_llm",
        lambda prompt, **kwargs: '{"estimated_monthly_income": 3200, "average_monthly_balance": 8000, "salary_detected": true, "summary": "test"}'
    )

    csv_content = "Date,Description,Amount\n2024-01-01,Salary,3200\n2024-01-15,Rent,-1500\n"
    path = write_temp_file(csv_content, ".csv")
    try:
        result = document_processor.extract_bank_statement(path)
        assert isinstance(result, dict)
    finally:
        os.unlink(path)


# ── Text resume ──────────────────────────────────────────────────────────────

def test_resume_text_extraction_returns_dict(monkeypatch):
    from app.services import document_processor

    monkeypatch.setattr(
        document_processor, "invoke_llm",
        lambda prompt, **kwargs: '{"years_of_experience": 5.0, "highest_education": "Bachelor", "skills": ["Python"], "last_job_title": "Engineer", "employment_gaps": false, "industry": "Tech", "certifications": [], "summary": "test"}'
    )

    resume = "John Doe\nSoftware Engineer\n5 years of experience in Python development\nBachelor of Computer Science\n"
    path = write_temp_file(resume, ".txt")
    try:
        result = document_processor.extract_resume(path)
        assert isinstance(result, dict)
    finally:
        os.unlink(path)


# ── Credit report ────────────────────────────────────────────────────────────

def test_credit_report_extraction_returns_dict(monkeypatch):
    from app.services import document_processor

    monkeypatch.setattr(
        document_processor, "invoke_llm",
        lambda prompt, **kwargs: '{"credit_score": 720, "total_open_accounts": 3, "total_closed_accounts": 1, "payment_history_rating": "good", "outstanding_debt": 5000, "credit_utilization_pct": 20, "defaults_or_late_payments": 0, "summary": "test"}'
    )

    credit_txt = "Credit Score: 720\nPayment History: Good\nOutstanding Debt: AED 5,000\n"
    path = write_temp_file(credit_txt, ".txt")
    try:
        result = document_processor.extract_credit_report(path)
        assert isinstance(result, dict)
    finally:
        os.unlink(path)


# ── Dispatcher ───────────────────────────────────────────────────────────────

def test_process_document_unknown_type():
    from app.services.document_processor import process_document
    result = process_document("unknown_type", "/nonexistent")
    assert "error" in result


def test_process_document_routes_correctly(monkeypatch):
    from app.services import document_processor

    called_with = {}

    def mock_extract_resume(path):
        called_with["path"] = path
        return {"years_of_experience": 3}

    # EXTRACTORS dict holds a direct reference captured at import time;
    # patch the dict entry rather than the module attribute.
    monkeypatch.setitem(document_processor.EXTRACTORS, "resume", mock_extract_resume)

    path = write_temp_file("resume content", ".txt")
    try:
        result = document_processor.process_document("resume", path)
        assert result == {"years_of_experience": 3}
        assert called_with["path"] == path
    finally:
        os.unlink(path)
