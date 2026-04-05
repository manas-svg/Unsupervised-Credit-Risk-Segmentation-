import os
import time
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
import joblib

# =========================
# 1. Load Dataset
# =========================
DATA_PATH = os.environ.get("DATA_PATH", "final_dataset.csv")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "kproto_model.pkl")
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH", "credit_data.csv")

raw_scaler = {}
if os.path.exists(RAW_DATA_PATH):
    raw_df = pd.read_csv(RAW_DATA_PATH)
    raw_numeric_cols = [
        "Age",
        "Income",
        "Credit Score",
        "Loan Amount",
        "Loan Tenure",
        "LTV Ratio",
        "Profile Score",
    ]
    raw_scaler = {
        col: {
            "mean": float(raw_df[col].mean()),
            "std": float(raw_df[col].std()),
        }
        for col in raw_numeric_cols
    }


df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError(f"Dataset at {DATA_PATH!r} is empty")

# =========================
# 2. Define Categorical Columns
# =========================
categorical_cols = [
    'Gender', 'Existing Customer', 'State', 'City', 
    'Employment Profile', 'Occupation', 'Is_Max_Loan', 
    'Is_Max_Profile', 'Is_Max_LTV', 'Is_Max_Loan_Amount', 
    'Is_Max_Profile_Score', 'Is_Min_LTV', 
    'Is_Min_Credit_Score', 'Is_Max_Credit_Score'
]

# Validate columns exist
missing_cat = [c for c in categorical_cols if c not in df.columns]
if missing_cat:
    raise KeyError(f"Missing categorical columns in dataset: {missing_cat}")

# Infer numeric columns as everything else
numeric_cols = [c for c in df.columns if c not in categorical_cols]

# =========================
# 3. Clean + Optimize types
# =========================
# Coerce numeric columns to float (invalid -> NaN)
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values
for col in numeric_cols:
    df[col] = df[col].fillna(0)

for col in categorical_cols:
    df[col] = df[col].astype(str).fillna("Unknown")

# Memory optimization
df[numeric_cols] = df[numeric_cols].astype("float32")

# =========================
# 4. Get Categorical Indices (required by kmodes)
# =========================
cat_idx = [df.columns.get_loc(col) for col in categorical_cols]

# =========================
# 5. Model selection (pick k by lowest cost)
# =========================
K_MIN = int(os.environ.get("K_MIN", "2"))
K_MAX = int(os.environ.get("K_MAX", "6"))  # inclusive
N_INIT = int(os.environ.get("N_INIT", "2"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))
N_JOBS = int(os.environ.get("N_JOBS", "4"))
SAMPLE_ROWS = int(os.environ.get("SAMPLE_ROWS", "15000"))

if K_MIN < 2 or K_MAX < K_MIN:
    raise ValueError(f"Invalid K range: {K_MIN}..{K_MAX}")

# =========================
# 6. Train Model
# =========================
if SAMPLE_ROWS > 0 and len(df) > SAMPLE_ROWS:
    df_train = df.sample(n=SAMPLE_ROWS, random_state=RANDOM_STATE)
    df_train = df_train.reset_index(drop=True)
    print(f"Training K-Prototypes model on sample: {len(df_train):,}/{len(df):,} rows...")
else:
    df_train = df
    print(f"Training K-Prototypes model on {len(df_train):,} rows...")
print(f"Categorical cols: {len(categorical_cols)} | Numeric cols: {len(numeric_cols)}")

best = None
history = []

for k in range(K_MIN, K_MAX + 1):
    start = time.time()
    kproto = KPrototypes(
        n_clusters=k,
        init="Cao",
        n_init=N_INIT,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    clusters = kproto.fit_predict(df_train, categorical=cat_idx)
    cost = float(kproto.cost_)
    elapsed = time.time() - start

    counts = np.bincount(clusters, minlength=k)
    min_cluster = int(counts.min()) if len(counts) else 0
    max_cluster = int(counts.max()) if len(counts) else 0

    mse = cost / len(df_train)
    rmse = float(np.sqrt(mse))
    history.append(
        {
            "k": k,
            "cost": cost,
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "elapsed_s": round(elapsed, 3),
            "min_cluster_size": min_cluster,
            "max_cluster_size": max_cluster,
        }
    )

    print(
        f"k={k:<2} cost={cost:,.2f}  elapsed={elapsed:,.2f}s  cluster_size(min,max)=({min_cluster},{max_cluster})"
    )

    if best is None or cost < best["cost"]:
        best = {
            "k": k,
            "cost": cost,
            "mse": mse,
            "rmse": rmse,
            "model": kproto,
        }

print("Training completed!")
print(f"Selected k={best['k']} with cost={best['cost']:,.2f}")
print(f"Training MSE-like cost per row={best['mse']:.4f}, RMSE-like cost per row={best['rmse']:.4f}")

# =========================
# 7. Save Model + Metadata
# =========================
thresholds = {
    "Loan Amount": {"max": float(df["Loan Amount"].max())},
    "Profile Score": {"max": float(df["Profile Score"].max())},
    "LTV Ratio": {
        "min": float(df["LTV Ratio"].min()),
        "max": float(df["LTV Ratio"].max()),
    },
    "Credit Score": {
        "min": float(df["Credit Score"].min()),
        "max": float(df["Credit Score"].max()),
    },
}
category_values = {
    col: sorted(df[col].dropna().astype(str).unique().tolist())
    for col in [
        "Gender",
        "Existing Customer",
        "State",
        "Employment Profile",
        "Occupation",
    ]
}
model_package = {
    "model": best["model"],
    "categorical_cols": categorical_cols,
    "cat_idx": cat_idx,
    "columns": df.columns.tolist(),
    "numeric_cols": numeric_cols,
    "k_selection": history,
    "trained_rows": int(len(df_train)),
    "train_metrics": {
        "k": best["k"],
        "cost": best["cost"],
        "mse": best["mse"],
        "rmse": best["rmse"],
        "rows": int(len(df_train)),
    },
    "category_values": category_values,
    "thresholds": thresholds,
    "raw_scaler": raw_scaler,
}

joblib.dump(model_package, OUTPUT_PATH)

print(f"Model saved as {OUTPUT_PATH}")