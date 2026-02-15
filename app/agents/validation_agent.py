"""
Data Validation Agent
Responsible for cross-checking extracted data for inconsistencies,
flagging discrepancies, and ensuring data quality.
"""
from langchain_core.tools import tool
from app.services.llm_service import invoke_llm, extract_json_from_response


@tool
def validate_applicant_data(applicant_data: dict, extracted_docs: dict) -> dict:
    """Validate and cross-reference applicant data with extracted document data.

    Args:
        applicant_data: The applicant's form data
        extracted_docs: Extracted data from all documents

    Returns:
        Validation results with flags and warnings
    """
    return run_validation(applicant_data, extracted_docs)


def run_validation(applicant_data: dict, extracted_docs: dict) -> dict:
    """Run all validation checks on applicant data."""
    flags = []
    warnings = []
    validated_data = dict(applicant_data)

    # 1. Identity cross-check
    eid_data = extracted_docs.get("emirates_id", {}).get("data", {})
    if eid_data:
        if eid_data.get("full_name") and eid_data["full_name"].lower() != applicant_data.get("full_name", "").lower():
            flags.append(f"Name mismatch: Form says '{applicant_data.get('full_name')}', Emirates ID says '{eid_data['full_name']}'")
        if eid_data.get("nationality") and eid_data["nationality"].lower() != applicant_data.get("nationality", "").lower():
            warnings.append(f"Nationality mismatch: Form says '{applicant_data.get('nationality')}', Emirates ID says '{eid_data['nationality']}'")

    # 2. Income cross-check with bank statement
    bank_data = extracted_docs.get("bank_statement", {}).get("data", {})
    if bank_data:
        estimated_income = bank_data.get("estimated_monthly_income", 0)
        declared_income = applicant_data.get("monthly_income", 0)
        if estimated_income and declared_income:
            try:
                estimated_income = float(estimated_income)
                declared_income = float(declared_income)
                if declared_income > 0 and abs(estimated_income - declared_income) / declared_income > 0.3:
                    flags.append(
                        f"Income discrepancy: Declared AED {declared_income:,.0f}, "
                        f"Bank statement suggests AED {estimated_income:,.0f} "
                        f"({abs(estimated_income - declared_income)/declared_income*100:.0f}% difference)"
                    )
                # Use bank statement figure as more reliable
                validated_data["verified_monthly_income"] = estimated_income
            except (ValueError, TypeError):
                pass

    # 3. Employment cross-check with resume
    resume_data = extracted_docs.get("resume", {}).get("data", {})
    if resume_data:
        resume_exp = resume_data.get("years_of_experience", 0)
        declared_exp = applicant_data.get("years_of_experience", 0)
        try:
            resume_exp = float(resume_exp)
            declared_exp = float(declared_exp)
            if declared_exp > 0 and abs(resume_exp - declared_exp) > 3:
                warnings.append(
                    f"Experience discrepancy: Declared {declared_exp} years, "
                    f"Resume indicates {resume_exp} years"
                )
        except (ValueError, TypeError):
            pass

        if resume_data.get("employment_gaps"):
            warnings.append("Resume shows employment gaps that may need review")

    # 4. Wealth cross-check
    assets_data = extracted_docs.get("assets_liabilities", {}).get("data", {})
    if assets_data:
        try:
            total_assets = float(assets_data.get("total_assets", 0))
            total_liabilities = float(assets_data.get("total_liabilities", 0))
            validated_data["total_assets"] = total_assets
            validated_data["total_liabilities"] = total_liabilities

            debt_ratio = total_liabilities / max(total_assets, 1)
            if debt_ratio > 1.0:
                flags.append(f"Liabilities exceed assets (debt ratio: {debt_ratio:.2f})")
            elif debt_ratio > 0.7:
                warnings.append(f"High debt-to-asset ratio: {debt_ratio:.2f}")
        except (ValueError, TypeError):
            pass

    # 5. Credit report cross-check
    credit_data = extracted_docs.get("credit_report", {}).get("data", {})
    if credit_data:
        try:
            credit_score = int(credit_data.get("credit_score", 0))
            if credit_score > 0 and credit_score < 500:
                flags.append(f"Very low credit score: {credit_score}")
            elif credit_score < 600:
                warnings.append(f"Below average credit score: {credit_score}")

            defaults = int(credit_data.get("defaults_or_late_payments", 0))
            if defaults > 3:
                flags.append(f"Multiple defaults/late payments detected: {defaults}")
        except (ValueError, TypeError):
            pass

    # Use LLM for nuanced validation summary
    validation_summary = _llm_validation_summary(applicant_data, extracted_docs, flags, warnings)

    return {
        "is_valid": len(flags) == 0,
        "flags": flags,
        "warnings": warnings,
        "validated_data": validated_data,
        "validation_summary": validation_summary,
    }


def _llm_validation_summary(applicant_data: dict, extracted_docs: dict, flags: list, warnings: list) -> str:
    """Use LLM to produce a human-readable validation summary."""
    prompt = f"""As a data validation specialist for a government social support program,
provide a brief validation summary for this applicant.

Applicant: {applicant_data.get('full_name', 'Unknown')}
Emirates ID: {applicant_data.get('emirates_id', 'Unknown')}
Employment: {applicant_data.get('employment_status', 'Unknown')}
Monthly Income: AED {applicant_data.get('monthly_income', 0):,.0f}
Family Size: {applicant_data.get('family_size', 0)}

Data Flags (critical issues): {flags if flags else 'None'}
Warnings (minor issues): {warnings if warnings else 'None'}

Provide a 2-3 sentence summary of data validation results.
Note if data is consistent or if issues were found."""

    try:
        return invoke_llm(prompt)
    except Exception:
        if flags:
            return f"Validation found {len(flags)} critical flag(s) and {len(warnings)} warning(s). Manual review recommended."
        return f"Data validation passed with {len(warnings)} minor warning(s)."


VALIDATION_AGENT_TOOLS = [validate_applicant_data]

VALIDATION_AGENT_PROMPT = """You are a Data Validation Agent for a government social support application system.

Your role is to:
1. Cross-reference data from the application form with extracted document data
2. Identify inconsistencies between declared information and document evidence
3. Flag critical discrepancies (income, identity, employment)
4. Produce a validation summary

Always be thorough but fair - minor discrepancies may be due to timing differences or rounding."""
