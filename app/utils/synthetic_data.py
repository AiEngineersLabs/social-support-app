"""
Synthetic data generator for training the ML eligibility classifier
and creating sample documents for testing the application workflow.
"""
import random
import json
import os
import pandas as pd
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

NATIONALITIES = ["UAE", "India", "Pakistan", "Philippines", "Egypt", "Jordan", "Syria", "Bangladesh"]
EDUCATION_LEVELS = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
EMPLOYMENT_STATUSES = ["Employed", "Unemployed", "Part-Time", "Self-Employed"]
GENDERS = ["Male", "Female"]
MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed"]
INDUSTRIES = ["Retail", "Construction", "Hospitality", "Healthcare", "IT", "Education", "Transport", "Manufacturing"]
SKILLS_POOL = [
    "Customer Service", "Microsoft Office", "Driving", "Cooking", "Accounting",
    "Welding", "Plumbing", "Nursing", "Teaching", "Data Entry", "Sales",
    "Carpentry", "Tailoring", "Security", "Cleaning", "Arabic", "English",
    "Project Management", "Communication", "Teamwork",
]

FIRST_NAMES_MALE = ["Ahmed", "Mohammed", "Ali", "Omar", "Hassan", "Khalid", "Saeed", "Rashid", "Yousef", "Ibrahim"]
FIRST_NAMES_FEMALE = ["Fatima", "Aisha", "Maryam", "Noura", "Hala", "Layla", "Sara", "Amina", "Rania", "Huda"]
LAST_NAMES = ["Al Maktoum", "Khan", "Ahmed", "Ali", "Hassan", "Al Nahyan", "Sharma", "Santos", "Reyes", "Ibrahim"]


def generate_emirates_id():
    return f"784-{random.randint(1950,2005)}-{random.randint(1000000,9999999)}-{random.randint(1,9)}"


def generate_applicant_profile() -> dict:
    gender = random.choice(GENDERS)
    if gender == "Male":
        first_name = random.choice(FIRST_NAMES_MALE)
    else:
        first_name = random.choice(FIRST_NAMES_FEMALE)
    last_name = random.choice(LAST_NAMES)

    age = random.randint(18, 65)
    emp_status = random.choice(EMPLOYMENT_STATUSES)
    education = random.choice(EDUCATION_LEVELS)
    family_size = random.randint(1, 8)
    dependents = random.randint(0, max(0, family_size - 1))
    marital = random.choice(MARITAL_STATUSES)

    # Income based on employment
    if emp_status == "Unemployed":
        monthly_income = random.uniform(0, 500)
    elif emp_status == "Part-Time":
        monthly_income = random.uniform(1000, 4000)
    elif emp_status == "Self-Employed":
        monthly_income = random.uniform(2000, 15000)
    else:
        monthly_income = random.uniform(3000, 25000)

    years_exp = random.uniform(0, min(age - 18, 30))

    # Assets & liabilities
    total_assets = random.uniform(0, 500000)
    total_liabilities = random.uniform(0, total_assets * 1.5)

    profile = {
        "full_name": f"{first_name} {last_name}",
        "emirates_id": generate_emirates_id(),
        "age": age,
        "gender": gender,
        "nationality": random.choice(NATIONALITIES),
        "marital_status": marital,
        "family_size": family_size,
        "dependents": dependents,
        "education_level": education,
        "employment_status": emp_status,
        "years_of_experience": round(years_exp, 1),
        "monthly_income": round(monthly_income, 2),
        "total_assets": round(total_assets, 2),
        "total_liabilities": round(total_liabilities, 2),
    }

    # Determine eligibility (for training labels)
    income_per_cap = monthly_income / max(family_size, 1)
    net_worth = total_assets - total_liabilities
    need_score = 0
    if income_per_cap < 2000:
        need_score += 3
    elif income_per_cap < 5000:
        need_score += 2
    elif income_per_cap < 10000:
        need_score += 1

    if emp_status == "Unemployed":
        need_score += 3
    elif emp_status == "Part-Time":
        need_score += 2

    if family_size >= 5:
        need_score += 2
    elif family_size >= 3:
        need_score += 1

    if net_worth < 10000:
        need_score += 3
    elif net_worth < 50000:
        need_score += 2

    if dependents >= 3:
        need_score += 1

    profile["eligible"] = 1 if need_score >= 5 else 0
    return profile


def generate_training_data(n: int = 500) -> list[dict]:
    """Generate n synthetic applicant profiles for ML training."""
    return [generate_applicant_profile() for _ in range(n)]


def generate_sample_bank_statement(applicant: dict) -> str:
    """Generate a sample bank statement CSV for an applicant."""
    months = ["2024-10", "2024-11", "2024-12"]
    rows = []
    for month in months:
        for day in range(1, 29, random.randint(2, 5)):
            date = f"{month}-{day:02d}"
            if random.random() < 0.3:
                # Credit (salary or transfer)
                amount = round(random.uniform(applicant["monthly_income"] * 0.8, applicant["monthly_income"] * 1.2), 2)
                rows.append({"Date": date, "Description": random.choice(["Salary Credit", "Transfer In", "Cash Deposit"]),
                            "Credit": amount, "Debit": 0, "Balance": 0})
            else:
                # Debit
                amount = round(random.uniform(50, 2000), 2)
                rows.append({"Date": date, "Description": random.choice(["Rent Payment", "Grocery", "Utility Bill", "ATM Withdrawal", "Online Purchase"]),
                            "Credit": 0, "Debit": amount, "Balance": 0})

    # Calculate running balance
    balance = round(random.uniform(1000, 10000), 2)
    for row in rows:
        balance += row["Credit"] - row["Debit"]
        row["Balance"] = round(balance, 2)

    df = pd.DataFrame(rows)
    path = str(BASE_DIR / "data" / "synthetic" / f"bank_statement_{applicant['emirates_id'].replace('-', '_')}.csv")
    df.to_csv(path, index=False)
    return path


def generate_sample_resume(applicant: dict) -> str:
    """Generate a sample resume text file for an applicant."""
    skills = random.sample(SKILLS_POOL, min(5, len(SKILLS_POOL)))
    industry = random.choice(INDUSTRIES)

    content = f"""CURRICULUM VITAE

Name: {applicant['full_name']}
Emirates ID: {applicant['emirates_id']}
Age: {applicant['age']}
Nationality: {applicant['nationality']}
Contact: +971-{random.randint(50,59)}-{random.randint(1000000,9999999)}
Email: {applicant['full_name'].lower().replace(' ', '.')}@email.com

EDUCATION
{applicant['education_level']} - Graduated {2024 - random.randint(1, 10)}

WORK EXPERIENCE
{f"Current: {industry} sector - {applicant['employment_status']}" if applicant['employment_status'] != 'Unemployed' else "Currently seeking employment"}
Total Experience: {applicant['years_of_experience']} years
Industry: {industry}

SKILLS
{chr(10).join(f"- {s}" for s in skills)}

REFERENCES
Available upon request
"""
    path = str(BASE_DIR / "data" / "synthetic" / f"resume_{applicant['emirates_id'].replace('-', '_')}.txt")
    with open(path, "w") as f:
        f.write(content)
    return path


def generate_sample_assets_liabilities(applicant: dict) -> str:
    """Generate a sample assets/liabilities Excel file."""
    assets = {
        "Category": ["Property", "Vehicles", "Savings Account", "Investments", "Other"],
        "Value (AED)": [
            round(applicant["total_assets"] * random.uniform(0.3, 0.5), 2),
            round(applicant["total_assets"] * random.uniform(0.05, 0.15), 2),
            round(applicant["total_assets"] * random.uniform(0.1, 0.3), 2),
            round(applicant["total_assets"] * random.uniform(0.05, 0.2), 2),
            round(applicant["total_assets"] * random.uniform(0, 0.1), 2),
        ]
    }
    liabilities = {
        "Category": ["Mortgage", "Personal Loans", "Credit Cards", "Vehicle Loan", "Other"],
        "Value (AED)": [
            round(applicant["total_liabilities"] * random.uniform(0.3, 0.5), 2),
            round(applicant["total_liabilities"] * random.uniform(0.1, 0.3), 2),
            round(applicant["total_liabilities"] * random.uniform(0.05, 0.15), 2),
            round(applicant["total_liabilities"] * random.uniform(0.05, 0.2), 2),
            round(applicant["total_liabilities"] * random.uniform(0, 0.1), 2),
        ]
    }

    path = str(BASE_DIR / "data" / "synthetic" / f"assets_{applicant['emirates_id'].replace('-', '_')}.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(assets).to_excel(writer, sheet_name="Assets", index=False)
        pd.DataFrame(liabilities).to_excel(writer, sheet_name="Liabilities", index=False)
    return path


def generate_sample_credit_report(applicant: dict) -> str:
    """Generate a sample credit report text file."""
    income = applicant["monthly_income"]
    debt = applicant["total_liabilities"]

    if income > 10000 and debt < 50000:
        score = random.randint(700, 850)
        rating = "Good"
    elif income > 5000:
        score = random.randint(550, 700)
        rating = "Fair"
    else:
        score = random.randint(300, 550)
        rating = "Poor"

    content = f"""CREDIT REPORT
=============
Report Date: 2024-12-15
Subject: {applicant['full_name']}
Emirates ID: {applicant['emirates_id']}

CREDIT SCORE: {score}
PAYMENT HISTORY: {rating}

ACCOUNT SUMMARY:
Open Accounts: {random.randint(1, 5)}
Closed Accounts: {random.randint(0, 3)}
Total Outstanding Debt: AED {round(debt * random.uniform(0.3, 0.7), 2):,.2f}
Credit Utilization: {random.randint(10, 90)}%
Late Payments (last 12 months): {random.randint(0, 5)}
Defaults: {random.randint(0, 2)}

INQUIRY HISTORY:
Recent Inquiries: {random.randint(0, 4)}
"""
    path = str(BASE_DIR / "data" / "synthetic" / f"credit_{applicant['emirates_id'].replace('-', '_')}.txt")
    with open(path, "w") as f:
        f.write(content)
    return path


def generate_sample_emirates_id_text(applicant: dict) -> str:
    """Generate a sample Emirates ID text representation."""
    content = f"""UNITED ARAB EMIRATES
IDENTITY CARD

Name: {applicant['full_name']}
ID Number: {applicant['emirates_id']}
Nationality: {applicant['nationality']}
Date of Birth: {2024 - applicant['age']}-{random.randint(1,12):02d}-{random.randint(1,28):02d}
Gender: {applicant['gender']}
Expiry Date: 2029-{random.randint(1,12):02d}-{random.randint(1,28):02d}
Card Number: {random.randint(100000000, 999999999)}
"""
    path = str(BASE_DIR / "data" / "synthetic" / f"eid_{applicant['emirates_id'].replace('-', '_')}.txt")
    with open(path, "w") as f:
        f.write(content)
    return path


def generate_full_sample_set(n_applicants: int = 5) -> list[dict]:
    """Generate a full set of sample applicants with all documents."""
    applicants_with_docs = []
    for _ in range(n_applicants):
        applicant = generate_applicant_profile()
        docs = {
            "bank_statement": generate_sample_bank_statement(applicant),
            "emirates_id": generate_sample_emirates_id_text(applicant),
            "resume": generate_sample_resume(applicant),
            "assets_liabilities": generate_sample_assets_liabilities(applicant),
            "credit_report": generate_sample_credit_report(applicant),
        }
        applicant["documents"] = docs
        applicants_with_docs.append(applicant)

    return applicants_with_docs


if __name__ == "__main__":
    # Generate training data
    training_data = generate_training_data(500)
    training_path = str(BASE_DIR / "data" / "synthetic" / "training_data.json")
    with open(training_path, "w") as f:
        json.dump(training_data, f, indent=2)
    print(f"Generated {len(training_data)} training records -> {training_path}")

    # Generate full sample set with documents
    samples = generate_full_sample_set(5)
    samples_path = str(BASE_DIR / "data" / "synthetic" / "sample_applicants.json")
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2, default=str)
    print(f"Generated {len(samples)} sample applicants with documents -> {samples_path}")
