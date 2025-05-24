# DiabetesPredictAI

> Early diabetes detection platform powered by machine learning

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 📌 Overview

**DiabetesPredictAI** is a powerful, interactive web-based tool that uses machine learning to assess diabetes risk. Designed for use by healthcare providers, researchers, and the general public, this tool facilitates early detection and informed clinical decision-making, ultimately contributing to better health outcomes.

---

## ✨ Features

- 📊 **Interactive Dashboard**: User-friendly Streamlit interface for real-time exploration.
- 📈 **Data Visualizations**: Correlation heatmaps, distribution plots, and outcome comparisons.
- 🧠 **Robust Machine Learning**: Tuned Random Forest classifier for accurate predictions.
- 🩺 **Clinical Insights**: Health insights derived from data aligned with medical standards.
- 🛠️ **Production-Ready**: Exportable models, suitable for integration in clinical software or mobile health apps.

---

## 🔍 Dataset

Utilizes the trusted **Pima Indians Diabetes Dataset**, which includes:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (target variable indicating diabetes presence)

---

## 📊 Visualizations

Gain deep insights with visual aids:

### ✅ Class Distribution
![Class Distribution](health1.png)

### 🔥 Correlation Heatmap
![Correlation Heatmap](health2.png)

### 📉 Feature Distributions
![Feature Distributions](health3.png)

### 📦 Glucose Level vs Outcome
![Glucose vs Outcome](health4.png)

### 🩺 Health Insights & Recommendations
![Health Insights](health5.png)

---

## 🤖 Model Performance

The tuned Random Forest model demonstrates:

- 🔍 **High Accuracy**: Reliable predictions across the test dataset
- 🧪 **Clinical Balance**: Strong precision and recall for real-world applicability
- 📌 **Feature Importance**: Easily explainable model to support transparency in healthcare

---

## 🚀 Getting Started

### 📋 Prerequisites
- Python 3.7 or later
- Libraries: `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/DiabetesPredictAI.git
cd DiabetesPredictAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the app:
```bash
streamlit run app.py
```

---

## 💡 Contribution & License

- Contributions are welcome! Open a pull request to add features or fix bugs.
- Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙌 Acknowledgments

- Pima Indians Diabetes Dataset – [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
- Open-source libraries used in the project

---

_"Empowering predictive healthcare through data and AI."_
