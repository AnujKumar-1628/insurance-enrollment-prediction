from pydantic import BaseModel, Field


class EmployeeRecord(BaseModel):
    age: int = Field(..., ge=16, le=100)
    gender: str
    marital_status: str
    salary: float = Field(..., ge=0)
    employment_type: str
    region: str
    has_dependents: bool
    tenure_years: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    enrollment_probability: float
    enrolled_prediction: int


class BatchPredictionRequest(BaseModel):
    records: list[EmployeeRecord]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
