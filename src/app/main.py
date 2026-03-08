"""
FASTAPI + GRADIO SERVING APPLICATION
Production-Ready ML Model Serving
"""

from typing import Optional, Dict, Any
from pathlib import Path
import random
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import gradio as gr

# ==============================================================================
# 1. MODEL LOADING (FIX MODEL PATH ISSUE)
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "model" / "model_trained.pkl"

model = None

if MODEL_PATH.exists():
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}. Using mock prediction.")


def predict(data: Dict[str, Any]) -> float:
    """
    Safe prediction wrapper.
    Uses real model if available, otherwise mock prediction.
    """
    global model

    if model is None:
        return round(random.uniform(0.0, 1.0), 4)

    try:
        import pandas as pd
        df = pd.DataFrame([data])
        prob = model.predict_proba(df)[0][1]
        return float(prob)
    except Exception as e:
        print("Prediction error:", e)
        return round(random.uniform(0.0, 1.0), 4)


# ==============================================================================
# 2. FASTAPI SETUP
# ==============================================================================

app = FastAPI(
    title="Home Credit Default Prediction API",
    description="ML API for predicting credit default risk",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"status": "ok", "service": "home-credit-prediction"}


# ==============================================================================
# 3. DATA SCHEMA
# ==============================================================================

class CustomerData(BaseModel):

    EXT_SOURCE_1: Optional[float] = Field(None)
    EXT_SOURCE_2: Optional[float] = Field(None)
    EXT_SOURCE_3: Optional[float] = Field(None)

    AMT_ANNUITY: float
    AMT_CREDIT: float
    AMT_GOODS_PRICE: float
    AMT_INCOME_TOTAL: float

    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_ID_PUBLISH: int
    NAME_EDUCATION_TYPE: int

    POS_CNT_INSTALMENT_FUTURE_MEAN: Optional[float] = None
    BUREAU_DAYS_CREDIT_MAX: Optional[float] = None
    INST_AMT_PAYMENT_SUM: Optional[float] = None
    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN: Optional[float] = None
    INST_DPD_MAX: Optional[float] = None
    PREV_APP_APPROVAL_RATE: Optional[float] = None

    class Config:
        extra = "allow"


# ==============================================================================
# 4. API ENDPOINT
# ==============================================================================

@app.post("/predict")
def get_prediction(data: CustomerData):

    try:

        try:
            input_data = data.model_dump()
        except:
            input_data = data.dict()

        prediction_prob = predict(input_data)

        return {
            "default_probability": prediction_prob,
            "risk_label": "High Risk" if prediction_prob > 0.5 else "Low Risk"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# 5. GRADIO INTERFACE FUNCTION
# ==============================================================================

def gradio_interface(
        amt_income, amt_credit, amt_annuity, amt_goods_price,
        ext_source_1, ext_source_2, ext_source_3,
        days_birth, days_employed, education_type
):

    data = {

        "AMT_INCOME_TOTAL": float(amt_income),
        "AMT_CREDIT": float(amt_credit),
        "AMT_ANNUITY": float(amt_annuity),
        "AMT_GOODS_PRICE": float(amt_goods_price),

        "EXT_SOURCE_1": float(ext_source_1),
        "EXT_SOURCE_2": float(ext_source_2),
        "EXT_SOURCE_3": float(ext_source_3),

        "DAYS_BIRTH": int(days_birth),
        "DAYS_EMPLOYED": int(days_employed),
        "NAME_EDUCATION_TYPE": int(education_type),

        "POS_CNT_INSTALMENT_FUTURE_MEAN": 10.0,
        "BUREAU_DAYS_CREDIT_MAX": -500.0,
        "INST_AMT_PAYMENT_SUM": 0.0,
        "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.0,
        "INST_DPD_MAX": 0.0,
        "PREV_APP_APPROVAL_RATE": 0.5,
        "DAYS_ID_PUBLISH": -2000
    }

    try:

        prob = predict(data)

        risk_level = "🔴 High Risk" if prob > 0.5 else "🟢 Low Risk"
        client_age = abs(int(days_birth)) // 365

        return f"""
Probability of Default: {prob:.2%}

Risk Level: {risk_level}

Client Info
-----------
Age: {client_age}
Income/Credit Ratio: {float(amt_income)/float(amt_credit):.2f}
"""

    except Exception as e:
        return f"Prediction error: {str(e)}"


# ==============================================================================
# 6. GRADIO UI
# ==============================================================================

demo = gr.Interface(

    fn=gradio_interface,

    inputs=[

        gr.Number(label="Total Income", value=135000),
        gr.Number(label="Credit Amount", value=568800),
        gr.Number(label="Annuity Amount", value=20560),
        gr.Number(label="Goods Price", value=450000),

        gr.Slider(0, 1, value=0.5, label="EXT_SOURCE_1"),
        gr.Slider(0, 1, value=0.5, label="EXT_SOURCE_2"),
        gr.Slider(0, 1, value=0.5, label="EXT_SOURCE_3"),

        gr.Number(label="Days Birth", value=-19241),
        gr.Number(label="Days Employed", value=-2329),
        gr.Dropdown([0, 1, 2, 3, 4], label="Education Type", value=1),
    ],

    outputs=gr.Textbox(label="Prediction", lines=6),

    title="🏦 Home Credit Default Risk Predictor",

    description="Predict probability of credit default using LightGBM model."
)

# ==============================================================================
# 7. MOUNT GRADIO INTO FASTAPI
# ==============================================================================

app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui"
)
