import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_package(path: str = "kproto_model.pkl"):
    package = joblib.load(path)
    return (
        package["model"],
        list(package["columns"]),
        list(package["categorical_cols"]),
        list(package["cat_idx"]),
        package.get("category_values", {}),
        package.get("thresholds", {}),
        package.get("raw_scaler", {}),
        package.get("train_metrics", {}),
    )


def build_row(columns: list[str], categorical_cols: set[str], values: dict):
    row = {}
    for col in columns:
        row[col] = "Unknown" if col in categorical_cols else 0
    row.update(values)
    return row


def scale_numeric_inputs(raw_values: dict, scaler_params: dict[str, dict]) -> dict[str, float]:
    scaled = {}
    for col, value in raw_values.items():
        params = scaler_params.get(col, {})
        if value is None:
            scaled[col] = 0.0
        elif params.get("std") in (None, 0):
            scaled[col] = float(value)
        else:
            scaled[col] = float((value - params["mean"]) / params["std"])
    return scaled


def compute_derived_flags(scaled_values: dict[str, float], thresholds: dict) -> dict[str, int]:
    return {
        "Is_Max_Loan": int(scaled_values.get("Loan Amount", 0) == thresholds.get("Loan Amount", {}).get("max", 0)),
        "Is_Max_Profile": int(scaled_values.get("Profile Score", 0) == thresholds.get("Profile Score", {}).get("max", 0)),
        "Is_Max_LTV": int(scaled_values.get("LTV Ratio", 0) == thresholds.get("LTV Ratio", {}).get("max", 0)),
        "Is_Max_Loan_Amount": int(scaled_values.get("Loan Amount", 0) == thresholds.get("Loan Amount", {}).get("max", 0)),
        "Is_Max_Profile_Score": int(scaled_values.get("Profile Score", 0) == thresholds.get("Profile Score", {}).get("max", 0)),
        "Is_Min_LTV": int(scaled_values.get("LTV Ratio", 0) == thresholds.get("LTV Ratio", {}).get("min", 0)),
        "Is_Min_Credit_Score": int(scaled_values.get("Credit Score", 0) == thresholds.get("Credit Score", {}).get("min", 0)),
        "Is_Max_Credit_Score": int(scaled_values.get("Credit Score", 0) == thresholds.get("Credit Score", {}).get("max", 0)),
    }


CLUSTER_LABELS = {
    0: "High Income - Low Risk",
    1: "Low Income - High Risk",
    2: "Young Professionals",
    3: "Moderate Income Users",
}

CLUSTER_DESCRIPTIONS = {
    0: (
        "This cluster represents customers with higher income, stronger credit profiles,"
        " and lower default risk. They are more likely to qualify for standard or premium"
        " credit products with stable repayment behavior."
    ),
    1: (
        "This cluster includes lower-income customers with weaker credit indicators and"
        " higher risk signals. Approval should be careful and products should focus on"
        " credit-building or secured-card options."
    ),
    2: (
        "This cluster reflects younger professionals who are early in their credit journey."
        " They typically have moderate income and fair-to-good credit scores, so products"
        " that support credit building and gradual limit increases are appropriate."
    ),
    3: (
        "This cluster contains moderate-income users with balanced risk profiles. They are"
        " generally stable borrowers who may qualify for standard credit offerings with"
        " monitoring of utilization and repayment behavior."
    ),
}

def credit_card_recommendation(age: int, income: float, credit_score: float, segment: str):
    score = 0
    reasons = []

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

    if income >= 80000:
        score += 2
        reasons.append("High income supports repayments.")
    elif income >= 40000:
        score += 1
        reasons.append("Moderate income supports basic limits.")
    else:
        score -= 1
        reasons.append("Low income may strain repayments.")

    if age < 21:
        score -= 1
        reasons.append("Very young applicant — thin/limited credit history is common.")
    elif age >= 25:
        score += 1
        reasons.append("Age suggests more stable credit behavior on average.")

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
        f"Decision: {decision}. Based on credit score, income, and segment risk signals. {limit_hint}"
    )
    return decision, explanation, reasons


def generate_llm_style_explanation(
    cluster: int,
    segment: str,
    segment_description: str,
    input_values: dict,
    decision: str,
    decision_explanation: str,
    decision_reasons: list[str],
) -> str:
    age = input_values.get("Age")
    income = input_values.get("Income")
    credit_score = input_values.get("Credit Score")
    gender = input_values.get("Gender")
    occupation = input_values.get("Occupation")
    city = input_values.get("City")

    return (
        f"### Cluster Explanation\n"
        f"The model assigned this profile to cluster {cluster} — **{segment}**. "
        f"This cluster is best described as: {segment_description} \n\n"
        f"### Why this profile fits the cluster\n"
        f"- Age: {age}\n"
        f"- Income: ${income:,.2f}\n"
        f"- Credit Score: {credit_score}\n"
        f"- Gender: {gender}\n"
        f"- Occupation: {occupation}\n"
        f"- City: {city}\n\n"
        f"These attributes suggest a profile that aligns with the cluster because the model is "
        f"looking for a mix of credit strength, income level, and behavioral indicators. "
        f"In particular, the credit score and income are the strongest signals in this case.\n\n"
        f"### Recommendation rationale\n"
        f"{decision_explanation} \n\n"
        f"### Supporting signals from the profile\n"
        + "\n".join(f"- {reason}" for reason in decision_reasons)
        + "\n\n"
        f"### Practical summary\n"
        f"In plain terms, this customer is being placed in a cluster that represents a customer"
        f" segment with the above income and credit characteristics. The suggested credit action"
        f" is `{decision}` and should be applied with the cluster's risk profile in mind."
    )


st.set_page_config(page_title="Credit Risk Modelling", page_icon="📊", layout="centered")
st.title("Credit Risk Modelling")
st.caption("Customer segmentation (K-Prototypes)")

try:
    (
        model,
        columns,
        categorical_cols_list,
        cat_idx,
        category_values,
        thresholds,
        raw_scaler,
        train_metrics,
    ) = load_package()
    categorical_cols = set(categorical_cols_list)
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

if train_metrics:
    st.subheader("Model evaluation")
    k = train_metrics.get("k")
    mse = train_metrics.get("mse")
    rmse = train_metrics.get("rmse")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Selected k", k)
        st.metric("MSE-like cost per row", f"{mse:.4f}" if mse is not None else "N/A")
    with r2:
        st.metric("RMSE-like cost per row", f"{rmse:.4f}" if rmse is not None else "N/A")
    st.write(
        "These values describe the K-Prototypes training cost per sample and are the nearest"
        " equivalent to MSE/RMSE for this unsupervised segmentation model."
    )

with st.form("predict"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=44, step=1)
        income = st.number_input("Income", min_value=0.0, value=76000.0, step=100.0)
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=583.0, step=1.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=105000.0, step=1000.0)
        loan_tenure = st.number_input("Loan Tenure (months)", min_value=1, max_value=360, value=133, step=1)
    with c2:
        gender = st.selectbox("Gender", category_values.get("Gender", ["Male", "Female", "Other"]))
        existing_customer = st.selectbox(
            "Existing Customer",
            category_values.get("Existing Customer", ["No", "Yes"]),
        )
        state = st.selectbox(
            "State",
            category_values.get("State", ["Unknown"]),
        )
        city = st.text_input("City", value="Mumbai")
        employment_profile = st.selectbox(
            "Employment Profile",
            category_values.get(
                "Employment Profile",
                ["Salaried", "Self-Employed", "Freelancer", "Student", "Unemployed"],
            ),
        )
    profile_score = st.number_input("Profile Score", min_value=0.0, max_value=100.0, value=77.0, step=1.0)
    ltv_ratio = st.number_input("LTV Ratio", min_value=0.0, max_value=150.0, value=72.0, step=0.1)
    occupation = st.selectbox(
        "Occupation",
        category_values.get(
            "Occupation",
            ["Engineer", "Doctor", "Teacher", "Business Owner", "Student", "Other"],
        ),
    )

    submitted = st.form_submit_button("Predict segment")

if submitted:
    raw_inputs = {
        "Age": int(age),
        "Income": float(income),
        "Credit Score": float(credit_score),
        "Loan Amount": float(loan_amount),
        "Loan Tenure": int(loan_tenure),
        "LTV Ratio": float(ltv_ratio),
        "Profile Score": float(profile_score),
    }
    scaled_inputs = scale_numeric_inputs(raw_inputs, raw_scaler)
    derived_flags = compute_derived_flags(scaled_inputs, thresholds)

    input_values = {
        **scaled_inputs,
        "Gender": str(gender),
        "City": str(city),
        "Occupation": str(occupation),
        "Existing Customer": str(existing_customer),
        "State": str(state),
        "Employment Profile": str(employment_profile),
        **derived_flags,
    }

    row = build_row(columns, categorical_cols, input_values)
    df = pd.DataFrame([row], columns=columns)

    for col in categorical_cols_list:
        df[col] = df[col].astype(str)

    try:
        cluster = int(model.predict(df, categorical=cat_idx)[0])
        segment = CLUSTER_LABELS.get(cluster, "Unknown Segment")
        segment_description = CLUSTER_DESCRIPTIONS.get(
            cluster,
            "No detailed description available for this cluster."
        )
        decision, decision_explanation, decision_reasons = credit_card_recommendation(
            age=int(age),
            income=float(income),
            credit_score=float(credit_score),
            segment=segment,
        )

        st.success("Prediction complete")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Cluster", cluster)
            st.write(f"**Segment:** {segment}")
        with c2:
            st.metric("Recommendation", decision)

        st.info(segment_description)
        st.write(decision_explanation)
        st.write("**Key reasons**")
        for r in decision_reasons:
            st.write(f"- {r}")

        llm_explanation = generate_llm_style_explanation(
            cluster,
            segment,
            segment_description,
            raw_inputs,
            decision,
            decision_explanation,
            decision_reasons,
        )
        st.markdown("---")
        st.markdown(llm_explanation)

        with st.expander("Input row used"):
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Prediction error: {e}")

