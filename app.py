from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()

# =========================
# Static + Templates
# =========================
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =========================
# Load Model Safely
# =========================
try:
    package = joblib.load("kproto_model.pkl")

    model = package["model"]
    categorical_cols = package["categorical_cols"]
    cat_idx = package["cat_idx"]
    columns = package["columns"]

except Exception as e:
    print("MODEL LOAD ERROR:", e)
    model = None

# =========================
# Cluster Labels
# =========================
cluster_labels = {
    0: "High Income - Low Risk",
    1: "Low Income - High Risk",
    2: "Young Professionals",
    3: "Moderate Income Users"
}

def credit_card_recommendation(age: int, income: float, credit_score: float, segment: str):
    score = 0
    reasons = []

    # Credit score signal (dominant)
    if credit_score >= 750:
        score += 3
        reasons.append("Strong credit score (≥ 750).")
    elif credit_score >= 700:
        score += 2
        reasons.append("Good credit score (700–749).")
    elif credit_score >= 650:
        score += 1
        reasons.append("Fair credit score (650–699) — needs tighter limits.")
    else:
        score -= 3
        reasons.append("Low credit score (< 650) — higher default risk.")

    # Income signal
    if income >= 80000:
        score += 2
        reasons.append("High income supports repayments.")
    elif income >= 40000:
        score += 1
        reasons.append("Moderate income supports basic limits.")
    else:
        score -= 1
        reasons.append("Low income may strain repayments.")

    # Age signal (very light)
    if age < 21:
        score -= 1
        reasons.append("Very young applicant — thin/limited credit history is common.")
    elif age >= 25:
        score += 1
        reasons.append("Age suggests more stable credit behavior on average.")

    # Segment hint from clustering (not a decision by itself)
    seg_lower = (segment or "").lower()
    if "low risk" in seg_lower:
        score += 1
        reasons.append("Segment indicates lower risk behavior.")
    elif "high risk" in seg_lower:
        score -= 1
        reasons.append("Segment indicates higher risk behavior.")

    if score >= 4:
        decision = "Approve"
        limit_hint = "Recommend a standard limit; consider a higher starting limit if income is stable."
    elif score >= 1:
        decision = "Review"
        limit_hint = "Recommend a low starting limit, verify income, and monitor utilization for 60–90 days."
    else:
        decision = "Decline"
        limit_hint = "Recommend decline for now; offer secured card / credit-builder product instead."

    explanation = (
        f"Decision: {decision}. This recommendation is based on credit score, income, and segment risk signals. "
        f"{limit_hint}"
    )
    return decision, explanation, reasons

# =========================
# Home Route
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request},
    )

# =========================
# Prediction Route (FORM)
# =========================
@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    age: int = Form(...),
    income: float = Form(...),
    credit_score: float = Form(...),
    gender: str = Form(...),
    city: str = Form(...),
    occupation: str = Form(...)
):
    try:
        # 🔥 If model not loaded
        if model is None:
            return templates.TemplateResponse(
                request,
                "index.html",
                {
                    "request": request,
                    "error": "Model not loaded. Check kproto_model.pkl"
                },
            )

        # =========================
        # Create Input
        # =========================
        input_dict = {
            "Age": age,
            "Income": income,
            "Credit Score": credit_score,
            "Gender": gender,
            "City": city,
            "Occupation": occupation,

            # Default values for remaining columns
            "Existing Customer": "Yes",
            "State": "Unknown",
            "Employment Profile": "Unknown",
            "Is_Max_Loan": "0",
            "Is_Max_Profile": "0",
            "Is_Max_LTV": "0",
            "Is_Max_Loan_Amount": "0",
            "Is_Max_Profile_Score": "0",
            "Is_Min_LTV": "0",
            "Is_Min_Credit_Score": "0",
            "Is_Max_Credit_Score": "0"
        }

        # =========================
        # DataFrame Processing
        # =========================
        # Build a full row with safe defaults:
        # - categorical columns default to "Unknown"
        # - numeric columns default to 0
        row = {}
        for col in columns:
            row[col] = "Unknown" if col in categorical_cols else 0
        row.update(input_dict)

        df = pd.DataFrame([row], columns=columns)

        # Convert categorical
        for col in categorical_cols:
            df[col] = df[col].astype(str)

        # =========================
        # Prediction
        # =========================
        cluster = model.predict(df, categorical=cat_idx)[0]
        segment = cluster_labels.get(cluster, "Unknown Segment")
        decision, decision_explanation, decision_reasons = credit_card_recommendation(
            age=age,
            income=income,
            credit_score=credit_score,
            segment=segment,
        )

        # =========================
        # Return result
        # =========================
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "cluster": cluster,
                "segment": segment,
                "decision": decision,
                "decision_explanation": decision_explanation,
                "decision_reasons": decision_reasons,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "error": str(e)
            },
        )