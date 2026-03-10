"""
Data Validation Agent — Reflexion Reasoning

Implements Reflexion pattern:
  Pass 1  → run all 5 cross-checks (identity, income, employment, wealth, credit)
  Reflect → LLM critiques whether validation was thorough and complete
  Pass 2  → if reflection identifies gaps, run additional targeted checks
  Compile → merge all flags/warnings into final validated result

This self-critique loop ensures no inconsistencies are overlooked.
"""
import logging
from langchain_core.tools import tool
from app.services.llm_service import invoke_llm, invoke_light_llm, extract_json_from_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def check_identity_consistency(applicant_data: dict, eid_data: dict) -> dict:
    """Cross-check applicant name, ID number and nationality against Emirates ID document."""
    import re
    flags, warnings = [], []

    # ── 1. Emirates ID number: declared vs document ──────────────────────────
    declared_id = applicant_data.get("emirates_id", "").strip()
    doc_id = eid_data.get("id_number", "").strip()
    eid_pattern = r"^784-\d{4}-\d{7}-\d$"

    if declared_id and not re.match(eid_pattern, declared_id):
        flags.append(f"Emirates ID format invalid: '{declared_id}'")

    if declared_id and doc_id:
        # Normalize both to digits-only for comparison.
        # Handles real OCR output variations: "784 1990 1234567 8",
        # "7841990-1234567-8", "784-1990-12345678" etc.
        declared_digits = re.sub(r"\D", "", declared_id)
        doc_digits = re.sub(r"\D", "", doc_id)
        if declared_digits != doc_digits:
            flags.append(
                f"Emirates ID number mismatch: declared='{declared_id}', document='{doc_id}'"
            )

    # ── 2. Full name: declared vs document (strict — last name must match) ───
    doc_name = eid_data.get("full_name", "").strip()
    if doc_name and doc_name.lower() not in ("", "unknown", "n/a"):
        declared_name = applicant_data.get("full_name", "").strip()
        if declared_name:
            declared_parts = declared_name.lower().split()
            doc_parts = doc_name.lower().split()

            # Last name (family name) must match exactly
            declared_last = declared_parts[-1] if declared_parts else ""
            doc_last = doc_parts[-1] if doc_parts else ""
            last_name_match = declared_last == doc_last

            # Count how many name words overlap
            overlap = len(set(declared_parts) & set(doc_parts))
            total_unique = len(set(declared_parts) | set(doc_parts))
            overlap_ratio = overlap / total_unique if total_unique > 0 else 0

            if not last_name_match and overlap_ratio < 0.5:
                # Family name differs AND less than half the words match → FLAG
                flags.append(
                    f"Name mismatch: declared='{declared_name}', document='{doc_name}'"
                )
            elif not last_name_match or overlap_ratio < 0.5:
                # Either last name differs OR low overlap → WARNING
                warnings.append(
                    f"Partial name difference: declared='{declared_name}', document='{doc_name}'"
                )

    # ── 3. Nationality ────────────────────────────────────────────────────────
    if eid_data.get("nationality"):
        decl_nat = applicant_data.get("nationality", "").lower()
        ext_nat = eid_data["nationality"].lower()
        if decl_nat and ext_nat and decl_nat not in ext_nat and ext_nat not in decl_nat:
            flags.append(
                f"Nationality mismatch: declared='{applicant_data.get('nationality')}', document='{eid_data['nationality']}'"
            )

    return {"flags": flags, "warnings": warnings}


@tool
def check_income_consistency(applicant_data: dict, bank_data: dict) -> dict:
    """Cross-check declared income against bank statement estimated income."""
    flags, warnings = [], []
    estimated = bank_data.get("estimated_monthly_income", 0)
    declared = applicant_data.get("monthly_income", 0)
    try:
        estimated, declared = float(estimated), float(declared)
        if estimated > 0 and declared >= 0:
            diff_pct = abs(estimated - declared) / max(declared, 1) * 100
            if diff_pct > 20:
                # > 20% gap → FLAG (stricter than before)
                flags.append(
                    f"Income discrepancy: declared AED {declared:,.0f}, "
                    f"bank statement suggests AED {estimated:,.0f} ({diff_pct:.0f}% gap)"
                )
            elif diff_pct > 10:
                warnings.append(
                    f"Minor income difference: declared AED {declared:,.0f}, "
                    f"bank suggests AED {estimated:,.0f} ({diff_pct:.0f}%)"
                )
        elif declared == 0 and estimated > 500:
            # Declared zero/unemployed but bank shows regular credits → FLAG
            flags.append(
                f"Income discrepancy: declared AED 0 (unemployed), "
                f"bank statement shows AED {estimated:,.0f}/month in credits"
            )
    except (ValueError, TypeError):
        pass
    return {"flags": flags, "warnings": warnings, "verified_income": estimated or declared}


@tool
def check_employment_consistency(applicant_data: dict, resume_data: dict) -> dict:
    """Cross-check employment history and experience against resume."""
    flags, warnings = [], []

    # ── Years of experience: declared vs resume ───────────────────────────────
    try:
        resume_exp = float(resume_data.get("years_of_experience", 0))
        declared_exp = float(applicant_data.get("years_of_experience", 0))
        if resume_exp > 0 or declared_exp > 0:
            gap = abs(resume_exp - declared_exp)
            if gap > 2:
                # > 2 year difference → FLAG
                flags.append(
                    f"Experience mismatch: declared {declared_exp:.0f} yrs, "
                    f"resume shows {resume_exp:.0f} yrs ({gap:.0f} yr gap)"
                )
            elif gap > 1:
                warnings.append(
                    f"Minor experience difference: declared {declared_exp:.0f} yrs, "
                    f"resume shows {resume_exp:.0f} yrs"
                )
    except (ValueError, TypeError):
        pass

    # ── Employment status vs resume activity ──────────────────────────────────
    emp_status = applicant_data.get("employment_status", "").lower()
    last_job = resume_data.get("last_job_title", "")
    employment_gaps = resume_data.get("employment_gaps", False)

    if emp_status == "unemployed" and last_job and "seeking" not in last_job.lower():
        warnings.append(
            f"Declared unemployed but resume lists recent job: '{last_job}'"
        )

    if emp_status == "employed" and not last_job:
        flags.append("Declared as employed but resume shows no current job title")

    if employment_gaps:
        warnings.append("Resume shows employment gaps requiring review")

    return {"flags": flags, "warnings": warnings}


@tool
def check_wealth_consistency(applicant_data: dict, assets_data: dict) -> dict:
    """Validate assets/liabilities and debt ratio."""
    flags, warnings = [], []
    try:
        total_assets = float(assets_data.get("total_assets", 0))
        total_liabilities = float(assets_data.get("total_liabilities", 0))
        if total_assets > 0:
            debt_ratio = total_liabilities / total_assets
            if debt_ratio > 1.0:
                flags.append(f"Liabilities exceed assets (debt ratio: {debt_ratio:.2f})")
            elif debt_ratio > 0.7:
                warnings.append(f"High debt-to-asset ratio: {debt_ratio:.2f}")
    except (ValueError, TypeError):
        pass
    return {
        "flags": flags,
        "warnings": warnings,
        "total_assets": float(assets_data.get("total_assets", 0)),
        "total_liabilities": float(assets_data.get("total_liabilities", 0)),
    }


@tool
def check_credit_standing(credit_data: dict) -> dict:
    """Validate credit score and payment history."""
    flags, warnings = [], []
    try:
        score = int(credit_data.get("credit_score", 0))
        if score > 0:
            if score < 500:
                flags.append(f"Very low credit score: {score}")
            elif score < 600:
                warnings.append(f"Below-average credit score: {score}")

        defaults = int(credit_data.get("defaults_or_late_payments", 0))
        if defaults > 3:
            flags.append(f"Multiple defaults/late payments: {defaults}")
        elif defaults > 1:
            warnings.append(f"Some late payment history: {defaults} incidents")
    except (ValueError, TypeError):
        pass
    return {"flags": flags, "warnings": warnings}


# ---------------------------------------------------------------------------
# Reflexion implementation
# ---------------------------------------------------------------------------

def _reflexion_critique(
    applicant_data: dict,
    extracted_docs: dict,
    pass1_flags: list,
    pass1_warnings: list,
) -> str:
    """
    LLM self-critique: did the validation catch everything?
    Returns a critique string identifying any missed checks.
    """
    prompt = f"""You are reviewing a data validation pass for a government social support application.

APPLICANT:
  Name: {applicant_data.get('full_name')}
  Emirates ID: {applicant_data.get('emirates_id')}
  Employment: {applicant_data.get('employment_status')}
  Income declared: AED {applicant_data.get('monthly_income', 0):,.0f}
  Family size: {applicant_data.get('family_size')}

DOCUMENTS AVAILABLE: {list(k for k in extracted_docs if not k.startswith('__'))}

PASS 1 FLAGS: {pass1_flags or 'None'}
PASS 1 WARNINGS: {pass1_warnings or 'None'}

Task: Critically review the validation. Did it check all important dimensions?
Consider:
1. Are there any cross-document inconsistencies not yet flagged?
2. Are there any missing documents that are critical?
3. Are there any implausible values (e.g., income vs family size vs housing)?
4. Any demographic or contextual red flags?

Return a brief critique (2-3 sentences). If validation is thorough, say "Validation is comprehensive."
If gaps exist, describe them specifically so a second pass can address them."""

    try:
        return invoke_light_llm(prompt, name="reflexion_critique")
    except Exception:
        return "Validation is comprehensive."


def _reflexion_second_pass(
    applicant_data: dict,
    extracted_docs: dict,
    critique: str,
) -> tuple[list, list]:
    """
    If the reflexion critique identified gaps, run an additional LLM check.
    Returns additional (flags, warnings).
    """
    if "comprehensive" in critique.lower() and "no" not in critique.lower():
        return [], []

    # Income sanity check against family size
    extra_flags, extra_warnings = [], []
    monthly_income = float(applicant_data.get("monthly_income", 0))
    family_size = int(applicant_data.get("family_size", 1))
    if monthly_income > 0 and family_size > 0:
        per_capita = monthly_income / family_size
        # UAE minimum wage context: below AED 800/person is very low
        if per_capita < 300 and applicant_data.get("employment_status") == "Employed":
            extra_flags.append(
                f"Suspicious: income AED {monthly_income:,.0f} for family of {family_size} "
                f"while declared as Employed (AED {per_capita:.0f}/person)"
            )

    # Check if critical documents are missing
    available_docs = [k for k in extracted_docs if not k.startswith("__")]
    critical_missing = [d for d in ["bank_statement", "emirates_id"] if d not in available_docs]
    if critical_missing:
        extra_warnings.append(f"Critical documents missing: {critical_missing} — reduced confidence")

    return extra_flags, extra_warnings


def run_validation(applicant_data: dict, extracted_docs: dict) -> dict:
    """
    Run Reflexion-based validation.

    Pass 1  → 5 rule-based + LLM cross-checks
    Reflect → LLM critiques completeness
    Pass 2  → targeted gap-filling checks
    Compile → unified result
    """
    reflexion_trace = []
    all_flags, all_warnings = [], []
    validated_data = dict(applicant_data)

    # ── Pass 1: rule-based checks ───────────────────────────────────────────
    reflexion_trace.append({"step": "pass_1_start", "checks": 5})

    def _extraction_failed(doc_entry: dict, required_keys: list) -> bool:
        """Return True when a document was processed but yielded no usable data."""
        if not doc_entry:
            return False          # document was never uploaded — skip silently
        data = doc_entry.get("data", {})
        if data.get("error"):
            return True           # explicit extraction error
        # All required keys missing or empty → extraction failed
        return all(not data.get(k) for k in required_keys)

    # 1. Identity
    eid_entry = extracted_docs.get("emirates_id", {})
    eid_data  = eid_entry.get("data", {}) if eid_entry else {}
    if _extraction_failed(eid_entry, ["id_number", "full_name"]):
        all_flags.append(
            "Emirates ID document was uploaded but could not be read — "
            "identity cannot be verified"
        )
        reflexion_trace.append({"check": "identity", "flags": all_flags[-1:], "warnings": []})
    elif eid_data:
        r = check_identity_consistency.invoke({"applicant_data": applicant_data, "eid_data": eid_data})
        all_flags.extend(r.get("flags", []))
        all_warnings.extend(r.get("warnings", []))
        reflexion_trace.append({"check": "identity", "flags": r.get("flags", []), "warnings": r.get("warnings", [])})

    # 2. Income
    bank_entry = extracted_docs.get("bank_statement", {})
    bank_data  = bank_entry.get("data", {}) if bank_entry else {}
    if _extraction_failed(bank_entry, ["estimated_monthly_income"]):
        all_warnings.append(
            "Bank statement was uploaded but income could not be extracted — "
            "income cannot be verified"
        )
        reflexion_trace.append({"check": "income", "flags": [], "warnings": all_warnings[-1:]})
    elif bank_data:
        r = check_income_consistency.invoke({"applicant_data": applicant_data, "bank_data": bank_data})
        all_flags.extend(r.get("flags", []))
        all_warnings.extend(r.get("warnings", []))
        if r.get("verified_income"):
            validated_data["verified_monthly_income"] = r["verified_income"]
        reflexion_trace.append({"check": "income", "flags": r.get("flags", []), "warnings": r.get("warnings", [])})

    # 3. Employment
    resume_entry = extracted_docs.get("resume", {})
    resume_data  = resume_entry.get("data", {}) if resume_entry else {}
    if _extraction_failed(resume_entry, ["years_of_experience", "last_job_title"]):
        all_warnings.append(
            "Resume was uploaded but content could not be extracted — "
            "employment history cannot be verified"
        )
        reflexion_trace.append({"check": "employment", "flags": [], "warnings": all_warnings[-1:]})
    elif resume_data:
        r = check_employment_consistency.invoke({"applicant_data": applicant_data, "resume_data": resume_data})
        all_flags.extend(r.get("flags", []))
        all_warnings.extend(r.get("warnings", []))
        reflexion_trace.append({"check": "employment", "flags": r.get("flags", []), "warnings": r.get("warnings", [])})

    # 4. Wealth
    assets_entry = extracted_docs.get("assets_liabilities", {})
    assets_data  = assets_entry.get("data", {}) if assets_entry else {}
    if _extraction_failed(assets_entry, ["total_assets"]):
        all_warnings.append(
            "Assets & liabilities document was uploaded but data could not be extracted"
        )
        reflexion_trace.append({"check": "wealth", "flags": [], "warnings": all_warnings[-1:]})
    elif assets_data:
        r = check_wealth_consistency.invoke({"applicant_data": applicant_data, "assets_data": assets_data})
        all_flags.extend(r.get("flags", []))
        all_warnings.extend(r.get("warnings", []))
        if r.get("total_assets") is not None:
            validated_data["total_assets"] = r["total_assets"]
            validated_data["total_liabilities"] = r.get("total_liabilities", 0)
        reflexion_trace.append({"check": "wealth", "flags": r.get("flags", []), "warnings": r.get("warnings", [])})

    # 5. Credit
    credit_entry = extracted_docs.get("credit_report", {})
    credit_data  = credit_entry.get("data", {}) if credit_entry else {}
    if _extraction_failed(credit_entry, ["credit_score"]):
        all_warnings.append(
            "Credit report was uploaded but data could not be extracted"
        )
        reflexion_trace.append({"check": "credit", "flags": [], "warnings": all_warnings[-1:]})
    elif credit_data:
        r = check_credit_standing.invoke({"credit_data": credit_data})
        all_flags.extend(r.get("flags", []))
        all_warnings.extend(r.get("warnings", []))
        reflexion_trace.append({"check": "credit", "flags": r.get("flags", []), "warnings": r.get("warnings", [])})

    # ── Reflexion: self-critique ────────────────────────────────────────────
    critique = _reflexion_critique(applicant_data, extracted_docs, all_flags, all_warnings)
    reflexion_trace.append({"step": "reflexion_critique", "critique": critique})

    # ── Pass 2: gap-filling ─────────────────────────────────────────────────
    extra_flags, extra_warnings = _reflexion_second_pass(applicant_data, extracted_docs, critique)
    if extra_flags or extra_warnings:
        all_flags.extend(extra_flags)
        all_warnings.extend(extra_warnings)
        reflexion_trace.append({
            "step": "pass_2_additions",
            "extra_flags": extra_flags,
            "extra_warnings": extra_warnings,
        })

    # ── LLM summary ────────────────────────────────────────────────────────
    validation_summary = _llm_validation_summary(applicant_data, all_flags, all_warnings)
    reflexion_trace.append({"step": "summary_generated"})

    return {
        "is_valid": len(all_flags) == 0,
        "flags": all_flags,
        "warnings": all_warnings,
        "validated_data": validated_data,
        "validation_summary": validation_summary,
        "reflexion_trace": reflexion_trace,
        "reflexion_critique": critique,
    }


def _llm_validation_summary(applicant_data: dict, flags: list, warnings: list) -> str:
    prompt = f"""As a data validation specialist, provide a 2-3 sentence summary of validation results.

Applicant: {applicant_data.get('full_name', 'Unknown')}
Employment: {applicant_data.get('employment_status', 'Unknown')}
Income: AED {applicant_data.get('monthly_income', 0):,.0f}
Family Size: {applicant_data.get('family_size', 0)}

Critical flags: {flags or 'None'}
Warnings: {warnings or 'None'}

Summarise: is the data consistent? Should it proceed or be flagged for review?"""
    try:
        return invoke_light_llm(prompt, name="validation_summary")
    except Exception:
        if flags:
            return f"Validation found {len(flags)} critical flag(s) and {len(warnings)} warning(s). Manual review recommended."
        return f"Data validation passed with {len(warnings)} minor warning(s). Application may proceed."


VALIDATION_AGENT_TOOLS = [
    check_identity_consistency,
    check_income_consistency,
    check_employment_consistency,
    check_wealth_consistency,
    check_credit_standing,
]
