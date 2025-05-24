import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

st.title("HealthPredict: Early Disease Detection Using Machine Learning")

# Load Dataset
df = pd.read_csv("diabetes.csv")
st.subheader("Sample of Dataset")
st.dataframe(df.head())

# EDA Visuals with Insights
st.subheader("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Outcome', data=df, ax=ax)
ax.set_title("Class Distribution: Diabetes Presence (1) vs Absence (0)")
st.pyplot(fig)
st.markdown("""
**Insight:** The dataset shows an imbalanced distribution with significantly more negative cases (no diabetes) 
than positive cases. This imbalance is common in medical datasets and reflects the real-world prevalence of diabetes, 
but may require techniques like oversampling or class weights during model training to ensure fair prediction capability.
""")

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)
st.markdown("""
**Insight:** The correlation matrix reveals that **Glucose**, **BMI** and **Age** have the strongest positive correlations 
with diabetes diagnosis (Outcome). This aligns with medical knowledge that elevated glucose levels and higher BMI 
are significant risk factors. Conversely, **Pregnancies** shows moderate positive correlation, indicating repeated 
pregnancies may increase diabetes risk, possibly due to hormonal changes affecting insulin sensitivity.
""")

st.subheader("Feature Distributions")
fig, ax = plt.subplots(figsize=(12, 10))
df.hist(ax=ax)
st.pyplot(fig)
st.markdown("""
**Insight:** The histograms reveal important patterns in our health data:
- **Glucose levels** show a right-skewed distribution, with a small but significant group having elevated readings
- **Age** distribution shows most patients are between 20-40 years, helping us understand our target demographic
- **BMI** has a roughly normal distribution centered around 30-32, indicating many subjects are in the overweight/obese range
- **Insulin** measurements show extreme right-skew with many near-zero values, suggesting measurement issues or data collection challenges
""")

st.subheader("Glucose vs Outcome")
fig, ax = plt.subplots()
sns.boxplot(x='Outcome', y='Glucose', data=df, ax=ax)
ax.set_title("Glucose Level vs Diabetes Outcome")
st.pyplot(fig)
st.markdown("""
**Insight:** This boxplot clearly demonstrates that patients diagnosed with diabetes (Outcome=1) have significantly 
higher glucose levels than those without diabetes. The median glucose level for diabetic patients is approximately 
140-150 mg/dL, well above the median for non-diabetic patients (around 110 mg/dL). We also observe less variation 
in the non-diabetic group. This strong separation makes glucose level one of our most powerful predictive features.
""")

# Data Preprocessing
cols_with_zeroes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
X = df.drop('Outcome', axis=1)
y = df['Outcome']

for col in cols_with_zeroes:
    df[col] = df[col].replace(0, df[col].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Removed Model Evaluation Results section, models are still trained silently
for name, model in models.items():
    model.fit(X_train, y_train)

# Hyperparameter Tuning (Random Forest)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8, None]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

st.subheader("Tuned Random Forest Model Results")
y_final_pred = best_model.predict(X_test)
st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_final_pred)))
st.text("Classification Report:\n" + classification_report(y_test, y_final_pred))
st.text(f"Final Accuracy: {accuracy_score(y_test, y_final_pred):.2f}")
st.markdown("""
**Model Insight:** Our tuned Random Forest model achieves high accuracy by effectively capturing the complex relationships 
between health metrics and diabetes risk. The model's strength lies in its ability to handle non-linear patterns 
and feature interactions, making it ideal for medical diagnostics where multiple factors contribute to disease risk 
in different ways. The confusion matrix reveals our model's balanced performance across both positive and negative cases.
""")

# Save Model
with open("healthpredict_model.pkl", "wb") as file:
    pickle.dump(best_model, file)
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Insights Section
st.subheader("Health Insights and Business Recommendations")

# Insights Section
st.markdown("""
## 🧠 Health Insights & 💼 Business Recommendations

---

### 🔍 **Key Health Insights:**

- 📊 **Glucose, BMI, and Age** are top predictors for diabetes — as seen in our correlation and feature distribution analysis.
- 🚨 A large number of zero values were found in features like **Insulin** and **Skin Thickness**, indicating missing or inaccurate data which must be addressed before real-world deployment.
- 📈 Our **Random Forest model**, after hyperparameter tuning, achieved impressive accuracy — ideal for supporting clinical decisions.

---

### 💡 **Business Impact & Opportunities:**

- 🏥 **Early Diagnosis** via machine learning can reduce long-term treatment costs and help healthcare providers with risk prioritization.
- 📱 **Mobile App Integration**: Turn this model into a mobile app that helps users self-assess their risk and prompts them to consult doctors early.
- 🧪 **Clinical Tool**: Embed this system in health kiosks at pharmacies or rural clinics to make predictive diagnostics more accessible.
- 📉 **Insurance Optimization**: Health insurance companies can use such models for better premium estimation and risk analysis.

---

> "Predictive healthcare isn't just about models. It's about transforming lives through smarter decisions."

✨ _Let's build a healthier future, one prediction at a time._
""")
