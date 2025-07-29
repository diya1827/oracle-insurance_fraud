from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
import cloudpickle
import json
import pandas as pd

app = FastAPI()

# ✅ Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace "*" with specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "hint": "Check the request body for exact field names and types."
        }
    )

@app.get("/")
def root():
    return {"status": "ok", "message": "FastAPI is running. Visit /docs for usage."}

# Load model and mappings
with open("model.pkl", "rb") as f:
    model = cloudpickle.load(f)

with open("claim_rejection_reason_mapping.json") as f:
    claim_mapping = json.load(f)
with open("payment_method_mapping.json") as f:
    payment_mapping = json.load(f)
with open("prior_authorization_mapping.json") as f:
    priorauth_mapping = json.load(f)
with open("fraud_investigation_flag_mapping.json") as f:
    fiflag_mapping = json.load(f)
with open("model_columns.json") as f:
    model_columns = json.load(f)

fiflag_inverse_mapping = {int(v): k for k, v in fiflag_mapping.items()}

# Define input schema
class InputData(BaseModel):
    claim_rejection_reason: str
    icd10_severity_score: int
    payment_method: str
    length_of_stay: str
    days_taken_to_claim: str
    prior_authorization: str

@app.post("/predict")
def predict(data: InputData):
    try:
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

        input_dict = {
            'Claim Rejection Reason': encoded_claim,
            'ICD-10 Severity Score': data.icd10_severity_score,
            'Payment Method': encoded_payment,
            'Prior Authorization': encoded_priorauth,
            f'days_taken_to_claim_{data.days_taken_to_claim}': 1,
            f'Length of Stay_{data.length_of_stay}': 1
        }

        final_input = {col: input_dict.get(col, 0) for col in model_columns}
        input_df = pd.DataFrame([final_input])

        prediction = model.predict(input_df)[0]
        readable_prediction = fiflag_inverse_mapping.get(int(prediction), "Unknown")

        return {"prediction": readable_prediction}
    except Exception as e:
        return {"error": str(e)}
