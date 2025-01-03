# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:56:50 2025

@author: alima
"""

import os
from pydantic import BaseModel
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import joblib

# Step 1: Load models
project_root = r'C:\Users\alima\code\e2e_loan_approval_predictor'
models = {
    "logistics_regression": joblib.load(os.path.join(project_root, 'data', 'processed', 'logistics_regression.pkl')),
    "random_forest": joblib.load(os.path.join(project_root, 'data', 'processed', 'random_forest.pkl')),
    "xgboost": joblib.load(os.path.join(project_root, 'data', 'processed', 'xgboost.pkl'))
}

model_name_mapping = {
    "Logistic Regression": "logistics_regression",
    "Random Forest": "random_forest",
    "XG Boost": "xgboost"
}

# Step 2: Initialize FastAPI
app = FastAPI()

## Define input schema
class PredictionInput(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int
    
## Load JSON from a file
file_path = os.path.join(project_root, 'data', 'processed', 'encoded_json.json')

with open(file_path, 'r') as file:
    categories_json = json.load(file)
    
education_mapping, self_employed_mapping = categories_json['education'], categories_json['loan_status']

## Function processing user request
def preprocess_input(data: PredictionInput):
    """
    Preprocess user input to prepare it for the ML model.

    Args:
        data (PredictionInput): User input validated by FastAPI.

    Returns:
        pd.DataFrame: Preprocessed data in a DataFrame format.
    """
    # Convert input to DataFrame
    input_dict = {
        "no_of_dependents": data.no_of_dependents,
        "education": education_mapping.get(data.education, -1),  # Encode 'education'
        "self_employed": self_employed_mapping.get(data.self_employed, -1),  # Encode 'self_employed'
        "income_annum": data.income_annum,
        "loan_amount": data.loan_amount,
        "loan_term": data.loan_term,
        "cibil_score": data.cibil_score,
        "residential_assets_value": data.residential_assets_value,
        "commercial_assets_value": data.commercial_assets_value,
        "luxury_assets_value": data.luxury_assets_value,
        "bank_asset_value": data.bank_asset_value,
    }
    return pd.DataFrame([input_dict])

# Step 3: FastAPI app development
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Handles all unexpected exceptions globally.
    """
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {exc}"}
    )


@app.get("/health/")
def health_check():
    return {"status": "healthy"}


# Prediction endpoint with model selection
@app.post("/predict/")
def predict(input_data: PredictionInput, model: str = Query(..., enum = ["Logistic Regression", "XG Boost"])):
    """
    Predict loan approval using the selected model.
    
    Args:
        input_data (PredictionInput): User input.
        model (str): Selected model ("log_reg", "svm", "xgboost").
    
    Returns:
        dict: Prediction result.
    """
    try:
        # Map full name to model name
        model_key = model_name_mapping.get(model)
        if model_key is None:
            raise HTTPException(status_code = 400, detail = "Invalid model selected.")
        
        # Get the selected model
        selected_model = models[model_key]
        
        # Preprocess input
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = selected_model.predict(processed_data)
        
        return {
            "model": model,
            "prediction": int(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail = str(e))
