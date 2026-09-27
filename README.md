# 📊 Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn based on customer demographics, services, contract details, and billing information.

The project includes data preprocessing, exploratory data analysis, model comparison, hyperparameter tuning, probability threshold optimization, and an interactive Streamlit application.

---

## 🚀 Live Demo

Coming soon.

---

## 🎯 Problem Statement

Customer churn is an important business problem for telecom companies.

The objective of this project is to predict customers who are likely to leave the service so that businesses can identify potential churners and take targeted retention actions.

---

## 📂 Dataset

The project uses the Telco Customer Churn dataset containing information about telecom customers, including:

- Customer demographics
- Tenure
- Phone services
- Internet services
- Online security
- Technical support
- Contract type
- Payment method
- Monthly charges
- Churn status

The dataset contains 7,043 customer records.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🔄 Machine Learning Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Encoding
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Threshold Tuning
   ↓
Model Serialization
   ↓
Streamlit Deployment


🤖 Models Evaluated

Three machine learning models were evaluated:

1. Logistic Regression

Used as an interpretable baseline and selected as the final deployed model.

2. Random Forest

An ensemble tree-based model used for comparison.

3. XGBoost

A gradient boosting model evaluated for additional performance comparison.

📈 Model Performance
Logistic Regression

At the default classification threshold of 0.50:

Metric	Score
Recall	82.0%
Precision	52.3%
ROC-AUC	0.86
🎯 Threshold Optimization

Instead of using the default probability threshold of 0.50, multiple thresholds were evaluated.

Threshold	Recall	Precision	Caught	Missed
0.25	95.4%	40.7%	356	17
0.30	94.1%	42.8%	351	22
0.35	92.5%	45.0%	345	28
0.40	87.7%	46.6%	327	46
0.45	85.5%	49.5%	319	54
0.50	82.0%	52.3%	306	67
0.55	78.6%	55.8%	293	80
0.60	73.7%	57.9%	275	98

The final application uses a probability threshold of 0.40.

At this threshold:

Recall: 87.7%
Precision: 46.6%
Customers caught: 327
Customers missed: 46

The threshold was selected to emphasize identifying potential churners while accepting a higher number of false positives.

🖥️ Streamlit Application

The project includes an interactive Streamlit application where users can enter customer information and receive:

Churn probability
Decision threshold
Risk level
Customer summary
Retention-oriented prediction message

The application loads the trained model, scaler, and threshold from serialized .pkl files.

customer-churn-prediction/
│
├── app.py
├── check_model.py
├── customer_churn_prediction.ipynb
│
├── churn_model.pkl
├── scaler.pkl
├── threshold.pkl
│
├── requirements.txt
├── .gitignore
└── README.md

⚙️ Run Locally
1. Clone the repository
git clone https://github.com/mzaid4696-bot/customer-churn-prediction.git
cd customer-churn-prediction

2. Create a virtual environment
python -m venv .venv

3. Activate the environment
Windows PowerShell:

.venv\Scripts\Activate.ps1

4. Install dependencies
pip install -r requirements.txt

5. Run Streamlit
streamlit run app.py

The application will open in your browser.

💡 Key Business Insights

The analysis found higher churn patterns associated with factors such as:
.Month-to-month contracts
.Higher monthly charges
.Shorter customer tenure
.Electronic check payment
.Certain internet service configurations
Longer contracts and additional support/security services were associated with lower observed churn in the dataset.

🔮 Future Improvements
Add SHAP-based model explanations
Add customer-level feature importance
Add batch CSV prediction
Add churn probability distribution dashboard
Add model monitoring
Add automated retraining pipeline
Add cloud deployment