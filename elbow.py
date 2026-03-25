import gc
import pandas as pd
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv("final_dataset.csv")
float_cols = df.select_dtypes(include=['float64']).columns
df[float_cols] = df[float_cols].astype('float32')

categorical_cols = [
    'Gender', 'Existing Customer', 'State', 'City', 
    'Employment Profile', 'Occupation', 'Is_Max_Loan', 'Is_Max_Profile', 
    'Is_Max_LTV', 'Is_Max_Loan_Amount', 'Is_Max_Profile_Score', 
    'Is_Min_LTV', 'Is_Min_Credit_Score', 'Is_Max_Credit_Score'
]

for col in categorical_cols:
    df[col] = df[col].astype(str)

cat_idx = [df.columns.get_loc(col) for col in categorical_cols]

df_sample = df.sample(frac=0.2, random_state=42)

costs = []
K_range = range(2, 9)

print("Starting Elbow Method on 4 cores (Sampled Data)...")
for k in tqdm(K_range, desc="Calculating Elbow"):
    kproto = KPrototypes(n_clusters=k, init='Cao', n_jobs=4, random_state=42)
    kproto.fit(df_sample, categorical=cat_idx)
    costs.append(kproto.cost_)
    print(f"k={k} complete. Cost: {kproto.cost_:.2f}")
    del kproto
    gc.collect()

plt.plot(K_range, costs, 'bx-')
plt.xlabel('k')
plt.ylabel('Cost')
plt.title('Elbow Method to find Optimal k')
plt.savefig('elbow_method_result.png', dpi=300, bbox_inches='tight')
plt.show()