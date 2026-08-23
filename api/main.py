from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from sklearn.pipeline import Pipeline

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EmployeeRecord,
    PredictionResponse,
)
from src.models.predict import load_model, predict_enrollment
from src.utils.config import CONFIG


model: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once when the API starts."""

    global model
    model = load_model(CONFIG.model.final_model_path)
    yield


app = FastAPI(
    title="Insurance Enrollment Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


def get_model() -> Pipeline:
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    return model


def prediction_row_to_response(row) -> PredictionResponse:
    return PredictionResponse(
        enrollment_probability=float(row["enrollment_probability"]),
        enrolled_prediction=int(row["enrolled_prediction"]),
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(CONFIG.model.final_model_path),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(record: EmployeeRecord):
    predictions = predict_enrollment(
        record.model_dump(),
        model=get_model(),
        threshold=CONFIG.model.threshold,
    )
    return prediction_row_to_response(predictions.iloc[0])


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    records = [record.model_dump() for record in request.records]
    predictions = predict_enrollment(
        records,
        model=get_model(),
        threshold=CONFIG.model.threshold,
    )

    return BatchPredictionResponse(
        predictions=[
            prediction_row_to_response(row)
            for _, row in predictions.iterrows()
        ]
    )
