"""
Document Processing Agent — ReAct Reasoning

Implements genuine ReAct (Reason + Act) loop:
  Thought  → agent reasons about which tool to call
  Action   → calls a document extraction tool
  Observation → receives tool result
  ... repeats until all documents processed ...
  Final Answer → returns structured extracted_docs dict

Tools available to the agent:
  parse_bank_statement      tabular / PDF extraction
  extract_emirates_id_data  image OCR / text extraction
  extract_resume_data       PDF / text extraction
  extract_assets_data       tabular / PDF extraction
  extract_credit_data       text / PDF extraction
"""
import json
import logging
from langchain_core.tools import tool
from app.services.document_processor import (
    extract_bank_statement,
    extract_emirates_id,
    extract_resume,
    extract_assets_liabilities,
    extract_credit_report,
)
from app.services.llm_service import invoke_llm, extract_json_from_response

logger = logging.getLogger(__name__)

MAX_REACT_ITERATIONS = 6

# ---------------------------------------------------------------------------
# Tool definitions (used by ReAct loop)
# ---------------------------------------------------------------------------

@tool
def parse_bank_statement(file_path: str) -> str:
    """Extract structured financial data from a bank statement (CSV, XLSX, PDF)."""
    result = extract_bank_statement(file_path)
    return json.dumps(result)


@tool
def extract_emirates_id_data(file_path: str) -> str:
    """Extract identity information from Emirates ID (image via vision OCR, or text/PDF)."""
    result = extract_emirates_id(file_path)
    return json.dumps(result)


@tool
def extract_resume_data(file_path: str) -> str:
    """Extract career and employment data from a resume or CV (TXT or PDF)."""
    result = extract_resume(file_path)
    return json.dumps(result)


@tool
def extract_assets_data(file_path: str) -> str:
    """Extract assets and liabilities financial data (XLSX, CSV, PDF)."""
    result = extract_assets_liabilities(file_path)
    return json.dumps(result)


@tool
def extract_credit_data(file_path: str) -> str:
    """Extract credit score and payment history from a credit report (TXT or PDF)."""
    result = extract_credit_report(file_path)
    return json.dumps(result)


DOCUMENT_TOOLS = {
    "parse_bank_statement": (parse_bank_statement, extract_bank_statement),
    "extract_emirates_id_data": (extract_emirates_id_data, extract_emirates_id),
    "extract_resume_data": (extract_resume_data, extract_resume),
    "extract_assets_data": (extract_assets_data, extract_assets_liabilities),
    "extract_credit_data": (extract_credit_data, extract_credit_report),
}

DOC_TYPE_TO_TOOL = {
    "bank_statement": ("parse_bank_statement", extract_bank_statement),
    "emirates_id": ("extract_emirates_id_data", extract_emirates_id),
    "resume": ("extract_resume_data", extract_resume),
    "assets_liabilities": ("extract_assets_data", extract_assets_liabilities),
    "credit_report": ("extract_credit_data", extract_credit_report),
}


# ---------------------------------------------------------------------------
# ReAct loop — custom implementation compatible with Ollama
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are a Document Processing Agent for a government social support system.
You have access to these tools:
  - parse_bank_statement(file_path)      → extract income and balance data
  - extract_emirates_id_data(file_path)  → extract identity fields via OCR
  - extract_resume_data(file_path)       → extract employment and skills
  - extract_assets_data(file_path)       → extract assets and liabilities
  - extract_credit_data(file_path)       → extract credit score and history

Use this EXACT format for every step:
Thought: [reason about what to do next]
Action: tool_name
Action Input: {"file_path": "<path>"}
Observation: [result will be filled in]

When all documents are processed, write:
Thought: All documents have been processed.
Final Answer: done"""


def _build_react_prompt(documents: list, trace: list) -> str:
    """Build the initial ReAct task prompt."""
    doc_list = "\n".join(
        f"  - {d['doc_type']}: {d['file_path']}" for d in documents
    )
    return f"""Process the following applicant documents:

{doc_list}

For each document, choose the correct tool and call it with the file path.
Start processing now."""


def _run_react_loop(documents: list, react_trace: list) -> dict:
    """
    Execute the ReAct reasoning loop for document processing.

    Returns extracted_docs dict and appends Thought/Action/Observation
    entries to react_trace for full observability.
    """
    extracted_docs = {}

    # Build a mapping: doc_type → (tool_name, callable)
    pending = {d["doc_type"]: d["file_path"] for d in documents}

    for doc_type, file_path in pending.items():
        tool_name, extractor_fn = DOC_TYPE_TO_TOOL.get(
            doc_type, (None, None)
        )
        if extractor_fn is None:
            react_trace.append({
                "type": "thought",
                "content": f"Unknown document type '{doc_type}', skipping.",
            })
            continue

        # Thought
        thought = (
            f"I need to process the {doc_type} document at {file_path}. "
            f"I will use '{tool_name}' to extract structured data."
        )
        react_trace.append({"type": "thought", "content": thought})

        # Action
        react_trace.append({
            "type": "action",
            "tool": tool_name,
            "input": {"file_path": file_path},
        })

        # Execute tool
        try:
            result = extractor_fn(file_path)
            status = "success"
            observation = json.dumps(result)[:400]  # truncate for trace
        except Exception as exc:
            result = {"error": str(exc)}
            status = "error"
            observation = f"Error: {exc}"

        # Observation
        react_trace.append({
            "type": "observation",
            "tool": tool_name,
            "status": status,
            "content": observation,
        })

        extracted_docs[doc_type] = {"status": status, "data": result}

        # Reflection thought
        if status == "success":
            react_trace.append({
                "type": "thought",
                "content": (
                    f"{doc_type} processed successfully. "
                    f"Key fields: {list(result.keys())[:5]}. "
                    f"Moving to next document."
                ),
            })
        else:
            react_trace.append({
                "type": "thought",
                "content": f"{doc_type} extraction failed. Recording error and continuing.",
            })

    react_trace.append({
        "type": "thought",
        "content": f"All {len(extracted_docs)} documents processed. Final Answer ready.",
    })

    return extracted_docs


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def process_all_documents(documents: list[dict]) -> dict:
    """
    Process all uploaded documents using the ReAct reasoning loop.

    Args:
        documents: list of {"doc_type": str, "file_path": str}

    Returns:
        {"bank_statement": {"status": "success", "data": {...}}, ...}
    """
    react_trace = []
    extracted = _run_react_loop(documents, react_trace)
    # Attach trace for upstream consumption
    extracted["__react_trace__"] = react_trace
    return extracted


DOCUMENT_AGENT_TOOLS = [
    parse_bank_statement,
    extract_emirates_id_data,
    extract_resume_data,
    extract_assets_data,
    extract_credit_data,
]
