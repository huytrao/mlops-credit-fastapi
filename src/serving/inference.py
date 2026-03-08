"""
INFERENCE PIPELINE - Production ML Model Serving for Home Credit Default Risk
=============================================================================

This module provides the core inference functionality for the Home Credit Default Risk model.
It ensures that serving-time feature transformations matches training-time logic,
handling feature engineering and schema alignment critical for LightGBM models.

Key Responsibilities:
1. Load the trained LightGBM model (Joblib/Pickle format)
2. Apply Feature Engineering (calculate EXT_SOURCES_MEAN, CREDIT_TERM, etc.)
3. Handle missing features (Padding/Imputation) for the LightGBM schema
4. Convert model probability to business-friendly output

CRITICAL PATTERN: Feature Consistency & Padding
- Calculates derived features (Mean, Min, Max) on the fly
- Aligns columns with the model's expected input (handling 100+ missing cols)
- Returns a probability score (0.0 to 1.0) rather than a binary class

Production Deployment:
- MODEL_DIR points to the directory containing the .pkl model file
- Designed to be robust against missing optional inputs
"""

import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb  # Required for model object handling

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path should point to where your model file is located.
# In a Docker container, this is usually mapped to /app/model/
MODEL_PATH = "model/model_trained.pkl"

# Global model variable
model = None
model_features = None

def _load_model():
    """
    Loads the model and extracts feature names once at startup.
    """
    global model, model_features
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            
            # Extract expected feature names from the model for alignment
            try:
                # For XGbost model
                model_features = model.feature_name()
            except:
                try:
                    # For Sklearn API
                    model_features = model.feature_name_
                except:
                    print("⚠️ Warning: Could not extract feature names from model.")
                    model_features = None
            
            if model_features:
                print(f"ℹ️ Model expects {len(model_features)} features.")
        else:
            print(f"❌ Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")

# Initialize model load on module import
_load_model()


# === FEATURE ENGINEERING LOGIC ===
def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering logic identical to training.
    
    This function derives new features from the raw inputs.
    LightGBM relies heavily on these calculated features (e.g., EXT_SOURCES_MEAN).
    
    Args:
        df: DataFrame with raw input columns
        
    Returns:
        DataFrame with added engineering columns
    """
    # 1. External Sources Statistics (Top Important Features)
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    
    # Ensure columns exist to avoid errors
    for col in ext_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    # Calculate row-wise statistics
    df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCES_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCES_MAX'] = df[ext_cols].max(axis=1)
    
    # 2. Financial Ratios
    # Credit Term: How long is the loan? (Annuity / Credit)
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        # Avoid division by zero
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)
        
    # Annuity to Income: Percentage of income going to loan
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)

    # 3. Demographics
    # Age in years (from Days Birth)
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = df['DAYS_BIRTH'] / -365.25
        
    return df

def _align_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align input DataFrame with the model's expected feature list.
    
    CRITICAL: LightGBM requires exact column order and count.
    This function pads missing columns with NaN/0.
    """
    if model_features is None:
        return df
        
    # Create a new DataFrame with the exact columns expected by the model
    # Initialize with NaN (LightGBM handles NaN well)
    df_aligned = pd.DataFrame(index=df.index, columns=model_features)
    
    # Fill in the data we have
    common_cols = [c for c in df.columns if c in model_features]
    df_aligned[common_cols] = df[common_cols]
    
    # Missing columns remain as NaN (Imputation happens inside LightGBM if configured, or treated as missing)
    # Alternatively, fill with 0 if that was the training strategy:
    # df_aligned = df_aligned.fillna(0)
    
    return df_aligned

def predict(input_dict: dict) -> float:
    """
    Main prediction function for Home Credit Default Risk.
    
    Pipeline:
    1. Convert input dict to DataFrame
    2. Engineer Features (calculate derived stats)
    3. Align Features (pad missing columns)
    4. Predict Probability
    
    Args:
        input_dict: Dictionary containing raw customer data
                   
    Returns:
        float: Probability of default (0.0 to 1.0)
    """
    # Check if model is loaded
    if model is None:
        # Fallback for testing UI without model file
        print("⚠️ Model not loaded. Returning dummy prediction.")
        return 0.5
    
    try:
        # === STEP 1: Convert Input to DataFrame ===
        df = pd.DataFrame([input_dict])
        
        # === STEP 2: Feature Engineering ===
        df_engineered = _engineer_features(df)
        
        # === STEP 3: Feature Alignment ===
        # Ensure we have all 100+ columns the model expects
        df_final = _align_features(df_engineered)
        
        # === STEP 4: Prediction ===
        # Get probability of class 1 (Default)
        # Handle different predict APIs (sklearn vs booster)
        if hasattr(model, "predict_proba"):
            # Sklearn API: returns [[prob_0, prob_1]]
            pred_prob = model.predict_proba(df_final)[0][1]
        else:
            # Booster API: returns [prob] or [raw_score] depending on params
            # Assuming standard prediction returns probability
            pred_prob = model.predict(df_final)[0]
            
        return float(pred_prob)
        
    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
        raise Exception(f"Model prediction failed: {str(e)}")