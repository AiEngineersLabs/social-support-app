from pydantic import BaseModel
from typing import Optional


class ApplicantCreate(BaseModel):
    full_name: str
    emirates_id: str
    age: int
    gender: str
    nationality: str
    marital_status: str
    family_size: int
    dependents: int = 0
    education_level: str
    employment_status: str
    current_employer: Optional[str] = None
    years_of_experience: float = 0
    monthly_income: float = 0
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class ApplicantResponse(BaseModel):
    id: int
    full_name: str
    emirates_id: str

    model_config = {"from_attributes": True}


class DecisionResponse(BaseModel):
    recommendation: str
    confidence_score: float
    eligibility_score: float
    income_score: float
    employment_score: float
    family_score: float
    wealth_score: float
    demographic_score: float
    reasoning: str
    enablement_recommendations: Optional[list] = None

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    applicant_id: int
    message: str


class ChatResponse(BaseModel):
    response: str
    agent_used: Optional[str] = None
