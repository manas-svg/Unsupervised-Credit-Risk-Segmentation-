# 🧠 Unsupervised Credit Risk Segmentation

## 📌 Overview
This project focuses on **segmenting borrowers based on credit risk** using an **unsupervised learning approach**. By leveraging the **K-Prototypes algorithm**, we effectively handle mixed-type data (numerical + categorical) to uncover meaningful borrower segments.

The goal is to assist financial institutions in:
- Understanding borrower behavior
- Identifying risk profiles
- Making data-driven lending decisions

---

## 🚀 Key Features
- 🔍 Segmentation of **270,000+ loan records**
- 🧩 Handles **mixed data types** using K-Prototypes
- 📊 Identifies **5 distinct borrower segments**
- ⚡ Improved processing speed by **~50%** via feature engineering & optimal K selection
- 📈 Provides actionable insights into **20+ features**

---

## 🛠️ Tech Stack
- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Kmodes (for K-Prototypes)
- Matplotlib & Seaborn (Visualization)

---

## 📂 Dataset
- Large-scale dataset with **270K+ loan records**
- Includes:
  - Numerical features (e.g., income, loan amount)
  - Categorical features (e.g., loan purpose, employment type)

>kaggle dataset link .

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handled missing values
- Encoded categorical features
- Scaled numerical variables
- Feature selection & engineering

### 2. Model Used
- **K-Prototypes Clustering**
  - Suitable for mixed data types
  - Combines K-Means (numerical) + K-Modes (categorical)

### 3. Optimal Cluster Selection
- Used cost-based evaluation to determine optimal **K = 5**

### 4. Segmentation Output
- Generated **5 borrower segments**
- Each segment represents a unique risk profile

---

## 📊 Results & Insights
- Successfully identified distinct borrower groups such as:
  - Low-risk stable borrowers
  - High-risk defaulters
  - Moderate-risk fluctuating profiles
- Enabled better understanding of:
  - Loan repayment behavior
  - Income vs loan patterns
  - Category-based risk trends

---

## ⚡ Performance Optimization
- Reduced computation time by **~50%**
- Achieved through:
  - Efficient feature engineering
  - Optimal cluster selection
  - Reduced dimensional noise

---

## 📸 Sample Visualizations
*(Add plots here if available)*
- Cluster distribution
- Feature importance
- Segment comparison charts

---

## 🔮 Future Work
- 🌐 Deploy the model using **FastAPI**
- 📊 Build an interactive dashboard using **Streamlit**
- 🔄 Automate real-time prediction pipeline
- 📈 Integrate with larger financial datasets

---

## 🧑‍💻 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/credit-risk-segmentation.git

# Navigate to project folder
cd credit-risk-segmentation

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
