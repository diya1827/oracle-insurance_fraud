from fastapi import FastAPI
from pydantic import BaseModel
import cloudpickle
import json
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = cloudpickle.load(f)

# Load label mappings
with open("claim_rejection_reason_mapping.json", "r") as f:
    claim_mapping = json.load(f)
with open("payment_method_mapping.json", "r") as f:
    payment_mapping = json.load(f)
with open("prior_authorization_mapping.json", "r") as f:
    priorauth_mapping = json.load(f)
with open("fraud_investigation_flag_mapping.json", "r") as f:
    fiflag_mapping = json.load(f)
with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

# Invert Fraud Investigation Flag mapping (so 0 → "No", 1 → "Yes")
fiflag_inverse_mapping = {int(v): k for k, v in fiflag_mapping.items()}

# Input schema
class InputData(BaseModel):
    claim_rejection_reason: str
    icd10_severity_score: int
    payment_method: str
    length_of_stay: str        # e.g., "6-10"
    days_taken_to_claim: str   # e.g., "11-15"
    prior_authorization: str   # e.g., "Yes" or "No"

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    try:
        # Encode using saved mappings
        encoded_claim = claim_mapping.get(data.claim_rejection_reason.strip())
        encoded_payment = payment_mapping.get(data.payment_method.strip())
        encoded_priorauth = priorauth_mapping.get(data.prior_authorization.strip())

        if None in [encoded_claim, encoded_payment, encoded_priorauth]:
            return {
                "error": "Invalid input values. Please use consistent categories from training data.",
                "details": {
                    "claim_rejection_reason": encoded_claim,
                    "payment_method": encoded_payment,
                    "prior_authorization": encoded_priorauth
                }
            }

        # Create feature dict and one-hot encode range-based fields
        input_dict = {
            'Claim Rejection Reason': encoded_claim,
            'ICD-10 Severity Score': data.icd10_severity_score,
            'Payment Method': encoded_payment,
            'Prior Authorization': encoded_priorauth,
            f'days_taken_to_claim_{data.days_taken_to_claim}': 1,
            f'Length of Stay_{data.length_of_stay}': 1
        }

        # Fill missing features with 0
        final_input = {col: input_dict.get(col, 0) for col in model_columns}

        # Convert to DataFrame
        input_df = pd.DataFrame([final_input])

        # Make prediction
        prediction = model.predict(input_df)[0]
        readable_prediction = fiflag_inverse_mapping.get(int(prediction), "Unknown")

        return {"prediction": readable_prediction}

    except Exception as e:
        return {"error": str(e)}
