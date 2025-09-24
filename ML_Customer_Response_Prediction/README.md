# 🧠 ML Customer Response Prediction  
Predicting whether a customer will respond to a marketing campaign using classification models.

## 📌 Project Overview
This project focuses on classifying customer responses to a marketing campaign based on demographic, behavioral, and product spending data.

📈 **Goal**: Help the marketing team identify which customers are most likely to respond to future campaigns.

---

## 🗂️ Dataset
The dataset includes:
- Demographics: Age, Income, Education, Marital Status
- Purchase history: Spending on products (wine, meat, fish, etc.)
- Web activity: Website visits, campaign history
- Target variable: `Response` (1 = Yes, 0 = No)

---

## 🔧 Steps Performed
1. 📊 **Data Cleaning**
   - Removed missing values
   - Fixed data types (dates, numbers)
   - Checked for duplicates

2. 🔧 **Feature Engineering**
   - `Customer_Tenure` (days since joining)
   - `Age` (based on year of birth)
   - `Family_Size` (adults + kids/teens)
   - `Total_Spending` (sum of product purchases)

3. 📈 **EDA using Matplotlib**
   - Age distribution
   - Spending vs response
   - Income vs response

4. 💡 **Business Questions**
   - What factors affect campaign response?
   - Do high spenders respond better?
   - How do age/income affect response?

5. 🧠 **Model Building**
   - Encoded categorical features
   - Trained Logistic Regression & Decision Tree Classifier

6. 🧾 **Model Evaluation**
   - Compared metrics: Accuracy, Precision, Recall, F1-score
   - Selected Decision Tree for better recall and balance

---

## 🧪 Final Decision: ✅ **Decision Tree Classifier**
Chosen for its ability to identify more actual responders (class 1), even with slightly lower accuracy — making it more effective for marketing targeting.

---

## 💻 Tools Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 👩‍💻 Author
**Zainab Noushab**  
_Aspiring Data Analyst | Computational Math & CS Student | Python + SQL Enthusiast_

🔗 [LinkedIn](https://www.linkedin.com/in/zainab-noushab)  
🔗 [GitHub](https://github.com/ZainabNoushab)

