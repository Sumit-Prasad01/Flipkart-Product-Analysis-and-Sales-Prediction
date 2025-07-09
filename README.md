# 📦 Flipkart Customer Support Escalation Prediction

## This project uses machine learning to predict whether a customer support interaction will be escalated, based on various interaction features such as response time, item price, agent shift, and more. The model aims to help improve customer satisfaction by identifying potentially high-risk cases early.

## 📊 Problem Statement

- Predict whether a customer support ticket will be escalated (binary classification), using features like:

- Channel type (Inbound, Outcall)

- Agent shift and tenure

- Product and support category

- Response time and item price

## ✅ Project Highlights

- Cleaned and preprocessed real-world customer support data

- Feature selection using Random Forest importance

- Imbalance handling with SMOTE and class_weight

- Trained multiple ML models: Logistic Regression, Random Forest, XGBoost

- Model explainability with SHAP

- Final model deployed with Streamlit

🧠 Models Used

### 1. Random Forest (Final Model)

- Tuned using RandomizedSearchCV

- Handles categorical + numerical data

- Balanced class weights

### 2. DecisionTree

### 3. XGBoost

### 4. Logisitic Regression

## ⚖️ Evaluation Metrics

- Accuracy

- Precision

- Recall

- F1-Score

- ROC-AUC Score

- F1-score and recall are prioritized due to class imbalance.


##🧾 Input Features Used

- response_time_minutes

- Item_price

- channel_name_Inbound

- channel_name_Outcall

- Tenure Bucket_*

- Agent Shift_*

## 🚀 Deployment

- The model is deployed using a Streamlit app:

- Takes user input via dropdowns & sliders

- Outputs prediction (Escalated / Not Escalated)

- Displays class probability

## ⚠️ Note on Model Size

- Due to the model's size (>1 GB), it is not included in this repo. You can:

- Place it in the models/ folder manually

## 📌 Requirements

- pip install -r requirements.txt

### Key Libraries:

- pandas, scikit-learn, xgboost, imblearn, shap, streamlit