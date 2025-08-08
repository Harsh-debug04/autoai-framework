import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

# --- App and Model Loading ---

app = FastAPI(title="AutoAI API", description="API for serving ML models trained with AutoAI.")

# --- Pydantic Models for Input Validation ---

class PredictionRequest(BaseModel):
    data: dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "age": 35,
                    "salary": 70000,
                    "country_UK": 0,
                    "country_USA": 1
                }
            }
        }

class PredictionResponse(BaseModel):
    prediction: Any


# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the AutoAI API!"}


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(request: PredictionRequest):
    """
    Accepts input data and returns a model prediction.
    """
    # --- Load model on demand ---
    try:
        API_DIR = Path(__file__).parent
        MODEL_PATH = API_DIR.parent.parent / "trained_model.joblib"
        payload = joblib.load(MODEL_PATH)
        model = payload['model']
        model_columns = payload['columns']
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"Model file not found at {MODEL_PATH}. Please train and save a model first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model could not be loaded: {e}")

    # --- Make prediction ---
    try:
        input_df = pd.DataFrame([request.data])
        input_df_reordered = input_df.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(input_df_reordered)[0]
        return {"prediction": prediction.item() if hasattr(prediction, 'item') else prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")
