import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class Applicant(Base):
    __tablename__ = "applicants"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    emirates_id = Column(String(20), unique=True, nullable=False)
    age = Column(Integer)
    gender = Column(String(10))
    nationality = Column(String(100))
    marital_status = Column(String(20))
    family_size = Column(Integer)
    dependents = Column(Integer, default=0)
    education_level = Column(String(100))
    employment_status = Column(String(50))
    current_employer = Column(String(255), nullable=True)
    years_of_experience = Column(Float, default=0)
    monthly_income = Column(Float, default=0)
    total_assets = Column(Float, default=0)
    total_liabilities = Column(Float, default=0)
    address = Column(Text, nullable=True)
    phone = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    documents = relationship("Document", back_populates="applicant")
    decision = relationship("Decision", back_populates="applicant", uselist=False)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(Integer, ForeignKey("applicants.id"), nullable=False)
    doc_type = Column(String(50), nullable=False)  # bank_statement, emirates_id, resume, assets_liabilities, credit_report
    file_path = Column(String(500), nullable=False)
    extracted_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    applicant = relationship("Applicant", back_populates="documents")


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(Integer, ForeignKey("applicants.id"), unique=True, nullable=False)
    recommendation = Column(String(20), nullable=False)  # APPROVE / SOFT_DECLINE
    confidence_score = Column(Float)
    eligibility_score = Column(Float)
    income_score = Column(Float)
    employment_score = Column(Float)
    family_score = Column(Float)
    wealth_score = Column(Float)
    demographic_score = Column(Float)
    reasoning = Column(Text)
    enablement_recommendations = Column(JSON, nullable=True)
    agent_trace = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    applicant = relationship("Applicant", back_populates="decision")
