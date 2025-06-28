# %% [markdown]
# # Marketing Campaign Response Prediction (ML Project)

# %% [markdown]
# ## Data Loading & Initial Exploration
# - Load dataset
# - Check structure, missing values, datatypes

# %%
import pandas as pd

df = pd.read_csv("marketing_campaign.csv", sep='\t')  
df.head()

# %%
print('Shape:', df.shape)

print('\nColumns:')
print(df.columns)
print(df.info)

# %% [markdown]
# ##  Data Cleaning
# - Fix date column
# - Handle missing income
# - Check duplicates

# %%
df.isnull().sum()
df.fillna(df['Income'].median(), inplace=True)

# %%
df['Dt_Customer'].head()
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

# %%
df.duplicated().sum()
df.nunique()

# %% [markdown]
# ## Feature Engineering
# - Customer Tenure
# - Age
# - Family Size
# - Total Spending

# %%
# 1. Customer Tenure (Days since joining)
import datetime
today = datetime.datetime.today()
df['Customer_Tenure'] = (today - df['Dt_Customer']).dt.days

# 2. Customer Age
df['Age'] = 2025 - df['Year_Birth']  # Fixed typo in comment

# 3. Family Size (Assuming 2 adults + kids at home)
df['Family_Size'] = 2 + df['Kidhome'] + df['Teenhome']

# 4. Total Spending 
spending_col = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spending'] = df[spending_col].sum(axis=1)


# %%
df[['Customer_Tenure', 'Age', 'Family_Size', 'Total_Spending']].head()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# - Age distribution
# - Spending vs response

# %%
import matplotlib.pyplot as plt

# Set figure size
plt.figure(figsize=(18, 12))

# Plot 1: Age Distribution
plt.subplot(3, 1, 1)
plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Customer Age', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Count')

# Plot 2: Total Spending by Campaign Response
plt.subplot(3, 1, 2)
avg_spend = df.groupby('Response')['Total_Spending'].mean()
plt.bar(['No (0)', 'Yes (1)'], avg_spend, color=['salmon', 'lightgreen'])
plt.title('Average Total Spending by Campaign Response')
plt.xlabel('Response')
plt.ylabel('Average Spending')

plt.show()


# %% [markdown]
# ## Business Questions
# 
# 1. **Which customer characteristics influence positive campaign response the most?**  
#    This will help identify which features (like age, income, spending) are most predictive of a “yes” response.
# 
# 2. **Do higher-spending customers respond better to marketing campaigns?**  
#    This will explore whether overall spending affects likelihood of response — helping prioritize customer segments.
# 
# 3. **How does customer age and income impact campaign response?**  
#    This will analyze the effect of age and income levels on engagement with the campaign.
# 

# %% [markdown]
# ## Model Building
# - Encode variables
# - Train/test split
# - Train Logistic Regression & Decision Tree

# %%
X = df.drop(['ID', 'Dt_Customer', 'Response'], axis=1)
y = df['Response']

# %%
X = pd.get_dummies(X, drop_first=True)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 1. Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# 2. Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

print("Both models trained successfully!")

# %% [markdown]
# ## Model Evaluation
# - Compare precision, recall, F1-score
# - Confusion matrix

# %%
from sklearn.metrics import classification_report, confusion_matrix

# Logistic Regression predictions
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Evaluation:\n")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Decision Tree predictions
y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree Evaluation:\n")
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# %% [markdown]
# ## Model Evaluation Summary
# 
# We trained and compared two classification models to predict customer responses to a marketing campaign:
# 
# ---
# 
# #### 1. Logistic Regression
# - **Accuracy:** 86%
# - **Precision (Class 1):** 0.61
# - **Recall (Class 1):** 0.20
# - **F1-Score (Class 1):** 0.30
# 
# Logistic Regression performed well in identifying non-responders (class 0), but failed to correctly detect most actual responders. It missed 55 out of 69 responders, which is a significant limitation for marketing campaign effectiveness.
# 
# ---
# 
# #### 2. Decision Tree Classifier
# - **Accuracy:** 82%
# - **Precision (Class 1):** 0.40
# - **Recall (Class 1):** 0.39
# - **F1-Score (Class 1):** 0.40
# 
# The Decision Tree model offered a better balance between identifying both responders and non-responders. It correctly identified 27 out of 69 responders, significantly outperforming Logistic Regression in that category.
# 
# ---
# 
# ## Conclusion
# While Logistic Regression has a higher overall accuracy, the Decision Tree model is **more effective for campaign targeting**, as it captures more actual responders. This makes it the **preferred model for marketing strategy** in this case.
# 
# **Final Model Selected:** Decision Tree Classifier
# 


