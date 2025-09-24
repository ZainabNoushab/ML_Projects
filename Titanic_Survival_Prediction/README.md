# 🛳️ Titanic Survival Prediction
## 📌 Project Overview

This project uses the famous Titanic dataset to predict whether a passenger survived or not based on features such as age, gender, ticket class, and fare.
The goal is to build and evaluate multiple machine learning classification models to compare their performance and identify the best one.

## 📂 Dataset

The dataset is taken from Kaggle’s Titanic Competition.

## Key features used in this project:

Pclass – Ticket class (1st, 2nd, 3rd)

Sex – Gender of passenger

Age – Passenger’s age

SibSp – Number of siblings/spouses aboard

Parch – Number of parents/children aboard

Fare – Ticket price

Embarked – Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Target variable:

Survived – 0 = No, 1 = Yes

## ⚙️ Project Pipeline

### Data Preprocessing


Handle missing values (Age, Embarked)

Drop irrelevant columns (Cabin, Name, Ticket, PassengerId)

Encode categorical variables (Sex, Embarked)

Exploratory Data Analysis (EDA)

Visualizations of survival distribution

Insights by gender, class, and age

Model Training

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Model Evaluation

Accuracy Score

Classification Report (Precision, Recall, F1-score)

ROC-AUC Score

Confusion Matrix Visualization

Final Comparison

Leaderboard comparing all models

Best performance achieved by Random Forest

## 📊 Results

Model	Test Accuracy	ROC-AUC

Logistic Regression	~0.80	~0.86

Decision Tree	~0.76	~0.77

Random Forest	~0.84	~0.89

KNN	~0.78	~0.82

SVM	~0.82	~0.87

📌 Random Forest achieved the highest performance overall.

## 🚀 How to Run

Clone the repository

```bash
git clone https://github.com/ZainabNoushab/titanic-survival-prediction.git
cd titanic-survival-prediction
```

Run the notebook in Jupyter/Colab

## 🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

## 📌 Future Improvements

Feature engineering (e.g., FamilySize, Title extraction)

Hyperparameter tuning with GridSearchCV

Advanced models (XGBoost, LightGBM, CatBoost)

## 🙌 Acknowledgements

Dataset from Kaggle: (Titanic - Machine Learning from Disaster)
