"""
Document Processing Agent
Responsible for extracting structured data from uploaded documents:
- Bank statements (CSV/Excel)
- Emirates ID (image/text)
- Resume (text)
- Assets & Liabilities (Excel)
- Credit Report (text)
"""
from langchain_core.tools import tool
from app.services.document_processor import process_document


@tool
def extract_document_data(doc_type: str, file_path: str) -> dict:
    """Extract structured data from a document.

    Args:
        doc_type: Type of document (bank_statement, emirates_id, resume, assets_liabilities, credit_report)
        file_path: Path to the document file

    Returns:
        Extracted data as a dictionary
    """
    return process_document(doc_type, file_path)


def process_all_documents(documents: list[dict]) -> dict:
    """Process all uploaded documents for an applicant.

    Args:
        documents: List of {"doc_type": str, "file_path": str}

    Returns:
        Dictionary mapping doc_type to extracted data
    """
    results = {}
    for doc in documents:
        doc_type = doc["doc_type"]
        file_path = doc["file_path"]
        try:
            extracted = process_document(doc_type, file_path)
            results[doc_type] = {"status": "success", "data": extracted}
        except Exception as e:
            results[doc_type] = {"status": "error", "error": str(e)}
    return results


DOCUMENT_AGENT_TOOLS = [extract_document_data]

DOCUMENT_AGENT_PROMPT = """You are a Document Processing Agent for a government social support application system.

Your role is to extract and structure data from applicant documents including:
- Bank statements: Extract income patterns, balances, transaction summaries
- Emirates ID: Extract identity information (name, ID number, nationality, DOB)
- Resume: Extract employment history, skills, education
- Assets/Liabilities: Extract financial position details
- Credit Reports: Extract credit score, payment history, outstanding debt

For each document:
1. Identify the document type
2. Extract all relevant fields
3. Return structured JSON data
4. Flag any inconsistencies or missing information

Always return data in a structured, consistent format."""
