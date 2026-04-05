# Credit Risk Modelling - K-Prototypes Clustering

## Understanding Credit Risk Through Data

Have you ever wondered how banks decide whether to approve a loan or credit card application? This project takes you through that entire journey. We built an intelligent customer segmentation system using K-Prototypes clustering that automatically categorizes customers into different risk profiles and recommends approval decisions.

Think of it this way: instead of reviewing each customer one by one, we teach the machine to recognize patterns in customer data and group similar customers together. Customers in the same group tend to behave similarly, so we can make better decisions faster.

---

## What This Project Does

This is a real-world machine learning application that does three main things:

1. **Segments Customers**: Using customer data like age, income, credit score, and employment details, the model groups customers into distinct profiles. For example: "High Income - Low Risk", "Young with Good Debt Management", etc.

2. **Explains Decisions**: When someone applies, the system doesn't just say "approved" or "rejected". It explains why they fell into that group and what characteristics define that group.

3. **Recommends Actions**: Based on the customer's segment, it recommends whether to Approve, Review, or Decline the application.

The system collects 13+ pieces of customer information and processes them through a trained K-Prototypes clustering model to deliver these insights instantly.

---

## Core Features

- **Customer Segmentation**: Groups customers into 4 distinct risk profiles based on their financial and demographic characteristics
- **User-Friendly Interface**: Simple form that collects customer information (age, income, credit score, loan amount, employment, location, etc.)
- **Model Performance Metrics**: Shows the training cost metrics so you understand how well the model learned from the data
- **Intelligent Explanations**: Provides detailed, easy-to-understand explanations for each classification
- **Decision Support**: Recommends Approve, Review, or Decline decisions based on cluster membership
- **Two Ways to Use It**: Web interface (Streamlit) or API (FastAPI) - choose what works best for you

---

## Getting Started in 5 Minutes

This section is for anyone who wants to see it working quickly, even if you're new to data science.

**What you need:**
- Python 3.8 or newer
- Git (for downloading the code)
- A command line (Terminal on Mac/Linux, PowerShell on Windows)

**Step 1: Download and Setup (2 minutes)**
```bash
# Download the project
git clone https://github.com/manas-svg/Unsupervised-Credit-Risk-Segmentation-.git
cd Credit-Risk-Modelling

# Create a safe space for project packages (called a virtual environment)
python -m venv venv

# Activate it (tells Python to use this project's packages)
.\venv\Scripts\Activate.ps1  # On Windows
# OR
source venv/bin/activate      # On Mac/Linux

# Install all the tools we need
pip install -r requirements.txt
```

**Step 2: Train the Model (1 minute)**
```bash
python Train_model.py
```
This reads the customer data and teaches the machine learning model to recognize customer patterns.

**Step 3: Run the Application (2 minutes)**
```bash
# Use the web interface (easiest)
streamlit run streamlit_app.py

# OR use the API version
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open your browser and go to `http://localhost:8501` (Streamlit) or `http://127.0.0.1:8000` (FastAPI). You'll see a form where you can enter customer details and get instant predictions.

---

## How the Project is Organized

Your project is organized into different files, each with a specific purpose:

```
Credit-Risk-Modelling/
├── Train_model.py              # Trains the clustering model on customer data
├── streamlit_app.py            # Creates the web interface (user-friendly)
├── app.py                      # Alternative API interface (FastAPI)
├── elbow.py                    # Helps find the right number of customer groups
├── EDA.ipynb                   # Notebook showing data exploration process
├── credit_data.csv             # Original customer data (raw)
├── final_dataset.csv           # Cleaned data ready for training
├── kproto_model.pkl            # The trained model (saved for reuse)
├── requirements.txt            # List of all Python packages needed
├── templates/                  # Folder for website elements
│   └── index.html
├── static/                     # Folder for styling
│   └── style.css
└── README.md                   # This file
```

**Key Files Explained:**

The `Train_model.py` file is the heart of everything. It reads customer data, learns patterns, and saves the trained model so we can use it later without retraining.

The `streamlit_app.py` file creates a simple web interface where users can enter customer information and get predictions instantly. This is what you'll interact with most.

The `credit_data.csv` contains historical customer information. The `final_dataset.csv` is the cleaned version after we've handled missing data and created useful features.

The `EDA.ipynb` is a Jupyter notebook that shows all the data exploration steps. As a data analyst fresher, this is valuable to understand how the data looks before it goes into the model.

---

## Understanding the Machine Learning Model

Let's talk about K-Prototypes. Imagine you're a bank manager with thousands of customer applications. You can't review each one individually. Instead, you identify "typical customers" - the responsible high earners, the young graduates just starting out, the risky borrowers, etc. K-Prototypes does exactly that, but automatically.

**How it works:** The algorithm looks at all customer attributes together (age, income, credit score, etc.) and finds groups of similar customers. It might discover that customers with high income, good credit scores, and stable employment form one group (low risk), while young customers with good income but high debt form another (moderate risk).

**The Numbers:**
- We use 7 numerical features (age, income, credit score, loan amount, loan tenure, LTV ratio, profile score)
- We use 6 categorical features (gender, existing customer status, state, city, employment profile, occupation)
- The model creates 2-6 customer segments based on what it learns
- We measure how good the model is using cost metrics (similar to MSE and RMSE)

**What it produces:**
When you enter a customer's information, the model:
1. Compares them to the patterns it learned
2. Assigns them to the closest customer group
3. Explains what that group is like
4. Recommends a credit decision

---

## Setting Up the Project (Detailed Version)

If you're new to Python projects, here's a complete walkthrough:

**Step 1: Get the Code**
Open your terminal and run:
```bash
git clone https://github.com/manas-svg/Unsupervised-Credit-Risk-Segmentation-.git
cd Credit-Risk-Modelling
```

**Step 2: Create a Virtual Environment**
A virtual environment is like a sandbox - it keeps this project's packages separate from other projects on your computer.
```bash
# Create the virtual environment
python -m venv venv

# Activate it (you'll see (venv) appear in your terminal)
.\venv\Scripts\Activate.ps1      # Windows
source venv/bin/activate          # Mac/Linux
```

**Step 3: Install Dependencies**
The `requirements.txt` file lists everything you need:
```bash
pip install -r requirements.txt
```

**What Gets Installed:**
- `scikit-learn` - The machine learning library containing K-Prototypes
- `pandas` - For reading and manipulating data
- `numpy` - For numerical calculations
- `streamlit` - For creating the web interface
- `fastapi` & `uvicorn` - For the API option
- `joblib` - For saving and loading the model

---

## Deploying Your Application

Once you have the model working locally, you can share it with others online. Here are the easiest options:

**Option 1: Streamlit Cloud (Recommended for Beginners)**

This is the easiest way to share your work. Your application will be live on the internet, and you can send people a link.

1. You've already uploaded your code to GitHub, so that's done!
2. Go to https://streamlit.io/cloud
3. Sign up with your GitHub account
4. Click "New app"
5. Fill in:
   - Repository: `manas-svg/Unsupervised-Credit-Risk-Segmentation-`
   - Branch: `main`
   - Main file: `streamlit_app.py`
6. Click Deploy

In seconds, you'll get a public URL like `https://your-app.streamlit.app` that anyone can visit!

**Option 2: Render (If You Prefer the API)**

For the FastAPI version, use Render (https://render.com):
1. Connect your GitHub repo to Render
2. Create a new Web Service
3. Set up:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Deploy and get a public API

**Option 3: Run Locally Only**

If you just want to use it on your own computer:
```bash
# Streamlit version
streamlit run streamlit_app.py
# Opens at http://localhost:8501

# FastAPI version
python -m uvicorn app:app --reload
# Opens at http://127.0.0.1:8000
```

---

## What You Can Do With This Project

**Customer Input Information**
When someone uses the app, they fill in:

Numerical data: Age, Income, Credit Score, Loan Amount, Loan Tenure (months), LTV Ratio, Profile Score
Categorical data: Gender, Existing Customer Status, State, City, Employment Profile, Occupation

**What You Get Back**
The system returns:
- Which customer group they belong to (Cluster 0-3)
- The name of that group (e.g., "High Income - Low Risk")
- A clear explanation of what that group looks like
- A credit decision recommendation: Approve, Review, or Decline

**Example Workflow:**
1. A young professional applies for a credit card
2. You enter their info into the form
3. The model analyzes their profile
4. It says: "Cluster 1: Young Ambitious Professional"
5. Description: "High income earners under 35 with good credit history and stable employment"
6. Recommendation: Approve

---

## Learning From This Project

As a data analyst, this project teaches you several important concepts:

**Data Preparation:** The EDA.ipynb notebook shows how to explore and clean data before machine learning. You'll see missing values, outliers, and how to create useful features.

**Unsupervised Learning:** Unlike supervised learning where we have right/wrong answers, this project uses clustering. The machine finds patterns without being told what the "correct" answer is.

**Model Training:** The Train_model.py file shows how to fit a model to data, save it, and reuse it later without retraining.

**Putting Models in Production:** Building a model is one thing; making it usable is another. Streamlit makes it easy to create interfaces that non-technical people can use.

---

## Exploring the Code

**To better understand how everything works:**

1. Look at the EDA.ipynb to see data exploration
2. Read Train_model.py to understand the training process
3. Check streamlit_app.py to see how predictions are made and displayed
4. Examine elbow.py to understand how we chose the number of clusters

Each file has comments explaining what the code does.

---

## Troubleshooting Common Issues

**Problem: Model won't train**
- Make sure `credit_data.csv` exists in the main folder
- Check your Python version is 3.8 or newer with `python --version`
- Make sure all packages installed with `pip list`

**Problem: Streamlit app won't load**
- Clear cache with `streamlit cache clear`
- Make sure virtual environment is activated (you should see `(venv)` in terminal)

**Problem: Can't deploy to Streamlit Cloud**
- Ensure your GitHub repository is public
- Double-check the file path is exactly `streamlit_app.py`
- Make sure `requirements.txt` is in the root folder

**Problem: Get an error about missing data**
- The model expects all 13 inputs (7 numeric + 6 categorical)
- Make sure nothing is left blank in the form

---

## Next Steps as a Data Analyst

Once you understand this project, try:
1. Modifying the features - what if you added annual credit limit or bankruptcy history?
2. Changing the number of clusters - how does it affect the results?
3. Creating a different target - instead of credit risk, could you segment by profitability?
4. Adding more data - how would the model improve with more customers?
5. Building your own project - try clustering different datasets

---

## Resources for Learning

- K-Prototypes documentation and tutorials
- Scikit-learn clustering documentation
- Streamlit tutorials for building interfaces
- Data science blogs and courses

---

## Author

Created by Manas - GitHub: [manas-svg](https://github.com/manas-svg)

---

## License

This project uses the MIT License. You're free to use, modify, and distribute this code.
- **Credit Recommendation**: Approve / Review / Decline with reasoning
- **Key Decision Factors**: Why the customer falls into that category
- **Full LLM-Style Explanation**: Multi-section narrative including cluster fit, recommendation rationale, and practical next steps

## Tech Stack

- **Python 3.8+**
- **scikit-learn**: StandardScaler for preprocessing
- **kmodes**: K-Prototypes clustering algorithm
- **pandas**: Data manipulation
- **Streamlit**: Web UI (recommended deployment)
- **FastAPI**: REST API alternative
- **joblib**: Model serialization

## Requirements

See [requirements.txt](requirements.txt) for full dependency list.

Install with:
```
pip install -r requirements.txt
```

## Example Workflow

1. User enters customer details (e.g., Age: 35, Income: $75k, Credit Score: 720)
2. Model scales numeric inputs using stored training statistics
3. Derives binary flags (Is_Max_Loan, Is_Min_Credit_Score, etc.)
4. K-Prototypes assigns to nearest cluster prototype
5. App displays:
   - Cluster assignment
   - Risk profile description
   - Credit card approval recommendation
   - Detailed reasoning with visual formatting

## Notes

- The K-Prototypes model is unsupervised; MSE/RMSE shown are clustering cost metrics per row, not classification accuracy
- Numeric features are z-score normalized using statistics from `credit_data.csv`
- Categorical features are kept as-is (kmodes handles categorical natively)
- The model package includes category value lists and thresholds for downstream applications

## License

Unlicensed (feel free to use and modify).

