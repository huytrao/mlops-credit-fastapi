"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Home Credit Default Risk
prediction model. It combines FastAPI for robust API endpoints and Gradio for an 
interactive web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation.
- Gradio: User-friendly web UI for manual testing and demonstrations.
- Pydantic: Data validation based on the Top 20 LightGBM important features.
"""

from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import gradio as gr

# IMPORT YOUR INFERENCE LOGIC
# Ensure src/serving/inference.py exists and has a predict() function
try:
    from src.serving.inference import predict
except ImportError:
    # Fallback for testing if the module isn't present
    print("WARNING: 'src.serving.inference' not found. Using mock prediction for demo.")
    import random
    def predict(data: Dict[str, Any]) -> float:
        # Mock logic: Return a random probability
        return round(random.uniform(0.0, 1.0), 4)

# ==============================================================================
# 1. FASTAPI SETUP
# ==============================================================================

app = FastAPI(
    title="Home Credit Default Prediction API",
    description="ML API for predicting credit default risk based on LightGBM Top 20 Features.",
    version="1.0.0"
)

@app.get("/")
def root():
    """Health check endpoint for monitoring and load balancers."""
    return {"status": "ok", "service": "home-credit-prediction"}

# ==============================================================================
# 2. DATA SCHEMA (PYDANTIC)
# ==============================================================================

class CustomerData(BaseModel):
    """
    Input schema optimized for the Top 20 LightGBM Features.
    Contains Raw features for calculation and key Aggregated features.
    """
    
    # --- GROUP 1: EXTERNAL SOURCES (Highest Importance) ---
    # These drive features like EXT_SOURCES_MEAN/MIN/MAX
    EXT_SOURCE_1: Optional[float] = Field(None, description="Normalized score from external data source 1", example=0.7526)
    EXT_SOURCE_2: Optional[float] = Field(None, description="Normalized score from external data source 2", example=0.7896)
    EXT_SOURCE_3: Optional[float] = Field(None, description="Normalized score from external data source 3 (Rank #4)", example=0.1595)

    # --- GROUP 2: FINANCIALS ---
    # Used to calculate CREDIT_TERM (#2) and ANNUITY_INCOME_PERCENT (#18)
    AMT_ANNUITY: float = Field(..., description="Loan annuity (monthly payment) (Rank #5)", example=20560.5)
    AMT_CREDIT: float = Field(..., description="Credit amount of the loan", example=568800.0)
    AMT_GOODS_PRICE: float = Field(..., description="Price of goods for which the loan is given (Rank #19)", example=450000.0)
    AMT_INCOME_TOTAL: float = Field(..., description="Income of the client", example=135000.0)

    # --- GROUP 3: DEMOGRAPHICS ---
    # DAYS_BIRTH is used to calculate AGE_YEARS (#17)
    DAYS_BIRTH: int = Field(..., description="Client's age in days (negative value)", example=-19241)
    DAYS_EMPLOYED: int = Field(..., description="Days employed (negative value) (Rank #16)", example=-2329)
    DAYS_ID_PUBLISH: int = Field(..., description="Days since ID was published (Rank #13)", example=-812)
    NAME_EDUCATION_TYPE: int = Field(..., description="Level of highest education (Label Encoded) (Rank #15)", example=1)

    # --- GROUP 4: HISTORY AGGREGATES (Pre-calculated) ---
    # These are complex aggregated features. If the backend doesn't calculate them,
    # the client must provide them.
    
    # Rank #7: POS - Mean of future installments
    POS_CNT_INSTALMENT_FUTURE_MEAN: Optional[float] = Field(None, example=12.0)
    
    # Rank #8: Bureau - Max days credit
    BUREAU_DAYS_CREDIT_MAX: Optional[float] = Field(None, example=-100.0)
    
    # Rank #10: Installments - Sum of payments
    INST_AMT_PAYMENT_SUM: Optional[float] = Field(None, example=500000.0)
    
    # Rank #12: Credit Card - Mean ATM drawings
    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN: Optional[float] = Field(None, example=0.0)
    
    # Rank #14: Installments - Max Days Past Due
    INST_DPD_MAX: Optional[float] = Field(None, example=0.0)
    
    # Rank #20: Previous Apps - Approval rate
    PREV_APP_APPROVAL_RATE: Optional[float] = Field(None, example=0.85)

    # Allow extra fields (e.g., SK_ID_CURR) without validation error
    class Config:
        extra = "allow"

# ==============================================================================
# 3. API ENDPOINTS
# ==============================================================================

@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint.
    1. Validates input using Pydantic.
    2. Converts data to dictionary.
    3. Calls the ML inference pipeline.
    """
    try:
        # Convert Pydantic model to dict
        input_data = data.dict()
        
        # Call inference logic
        prediction_prob = predict(input_data)
        
        return {
            "default_probability": prediction_prob,
            "risk_label": "High Risk" if prediction_prob > 0.5 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# 4. GRADIO INTERFACE LOGIC
# ==============================================================================

def gradio_interface(
    amt_income, amt_credit, amt_annuity, amt_goods_price,
    ext_source_1, ext_source_2, ext_source_3,
    days_birth, days_employed, education_type
):
    """
    Handler function for the Gradio UI.
    Maps UI inputs to the dictionary format expected by the model.
    """
    # 1. Prepare the dictionary matching CustomerData fields
    data = {
        # Raw inputs from UI
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
        
        # Default/Dummy values for Aggregated Features not exposed in UI
        # (To prevent model errors during simple manual testing)
        "POS_CNT_INSTALMENT_FUTURE_MEAN": 10.0,
        "BUREAU_DAYS_CREDIT_MAX": -500.0,
        "INST_AMT_PAYMENT_SUM": 0.0,
        "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.0,
        "INST_DPD_MAX": 0.0,
        "PREV_APP_APPROVAL_RATE": 0.5,
        "DAYS_ID_PUBLISH": -2000, 
    }
    
    # 2. Call the inference function
    try:
        prob = predict(data)
        
        # 3. Format the output for the user
        risk_level = "🔴 High Risk" if prob > 0.5 else "🟢 Safe / Low Risk"
        client_age = abs(int(days_birth)) // 365
        
        return f"""
        Prediction Result:
        ------------------
        Probability of Default: {prob:.2%}
        Risk Assessment: {risk_level}
        
        Client Profile:
        - Age: ~{client_age} years
        - Income/Credit Ratio: {float(amt_income)/float(amt_credit):.2f}
        """
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# ==============================================================================
# 5. GRADIO UI LAYOUT
# ==============================================================================

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Group 1: Financials
        gr.Number(label="Total Income (AMT_INCOME_TOTAL)", value=135000),
        gr.Number(label="Credit Amount (AMT_CREDIT)", value=568800),
        gr.Number(label="Annuity Amount (AMT_ANNUITY - Rank #5)", value=20560),
        gr.Number(label="Goods Price (AMT_GOODS_PRICE - Rank #19)", value=450000),
        
        # Group 2: External Sources (Crucial for LightGBM)
        gr.Slider(0, 1, value=0.5, label="EXT_SOURCE_1 (Rank #11)"),
        gr.Slider(0, 1, value=0.5, label="EXT_SOURCE_2 (Rank #1)"),
        gr.Slider(0, 1, value=0.5, label="EXT_SOURCE_3 (Rank #4)"),
        
        # Group 3: Personal/Demographics
        gr.Number(label="Days Birth (Negative Value) (Rank #9)", value=-19241),
        gr.Number(label="Days Employed (Negative Value) (Rank #16)", value=-2329),
        gr.Dropdown([0, 1, 2, 3, 4], label="Education Type (Encoded) (Rank #15)", value=1),
    ],
    outputs=gr.Textbox(label="Model Decision", lines=5),
    title="🏆 Home Credit - Default Risk Predictor",
    description="""
    **Real-time prediction based on the Top 20 LightGBM Features.**
    
    This interface exposes the most critical features identified by the model:
    1. **External Sources:** Normalized credit scores from external agencies.
    2. **Financial Ratios:** Annuity, Credit amount, and Goods price.
    3. **Demographics:** Age and Employment history.
    """,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# ==============================================================================
# 6. APP MOUNTING
# ==============================================================================

# Mount the Gradio app onto FastAPI at the /ui path
app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui"
)

# To run: uvicorn main:app --reload