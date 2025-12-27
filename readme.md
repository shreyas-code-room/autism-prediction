# 🧠 Autism Spectrum Disorder Prediction System

## 📌 Overview

This project focuses on building a **Machine Learning–based system** to predict **Autism Spectrum Disorder (ASD)** using behavioral and demographic features. Early prediction of ASD can support timely clinical assessment and intervention.

The system evaluates multiple ML algorithms and compares their performance using standard classification metrics and confusion matrices.

---

## 🎯 Objectives

* Analyze behavioral and screening data related to Autism
* Train and evaluate multiple ML classification models
* Compare model performance using accuracy and confusion matrices
* Provide a reusable and extensible prediction pipeline

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries & Tools:**

  * NumPy
  * Pandas
  * Scikit-learn
  * XGBoost
  * Matplotlib
  * Joblib
* **Environment:** Virtual Environment (venv)

---

## 📂 Project Structure

```
autism-prediction/
│
├── data/                          # Dataset files                  
│
├── app.py                         # Prediction / inference script
├── train.py                       # Model training script
├── requirements.txt               # Project dependencies
├── readme.md                      # Project documentation
├── .gitignore                     # Ignored files
│
├── confusion_matrix.png
├── logistic_regression_confusion_matrix.png
├── svm_confusion_matrix.png
├── xgboost_confusion_matrix.png
│
├── ResultTesting.xlsx             # Testing results
└── code.txt                       # Additional notes (optional)
```

---

## 📊 Machine Learning Models Used

* **Logistic Regression**
* **Support Vector Machine (SVM)**
* **XGBoost Classifier**

Each model was trained and evaluated independently to identify the most accurate and reliable classifier for ASD prediction.

---

## 📈 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Model Comparison

Confusion matrices for each model are saved as images:

* Logistic Regression
* SVM
* XGBoost

---

## 📁 Dataset

* Behavioral and demographic screening data for Autism prediction
* Dataset includes features such as age, gender, behavioral responses, and screening scores
* (Dataset source can be mentioned here if publicly available)

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/autism-prediction.git
cd autism-prediction
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux / Mac**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Train Models

```bash
python train.py
```

---

### 5️⃣ Run Prediction

```bash
streamlit run app.py
```

---

## 🧪 Results

* Models were evaluated using unseen test data
* Performance comparison shows variation across classifiers
* XGBoost and SVM demonstrated strong predictive performance

(Refer to confusion matrix images for detailed analysis)

---

## 🔍 Key Highlights

* End-to-end ML pipeline (data preprocessing → training → evaluation)
* Multiple model comparison
* Clean, modular code structure
* Healthcare-focused AI application

---

## 🔮 Future Enhancements

* Integrate Flask web interface for real-time predictions
* Deploy the application on cloud platforms (AWS / Render)
* Add explainable AI (SHAP / LIME)
* Expand dataset for improved generalization

---

## 👤 Author

**Shreyas B**
MCA Graduate | Machine Learning & Full Stack Developer
GitHub: [https://github.com/YOUR_USERNAME](https://github.com/shreyas-code-room)

---

## 📜 License

This project is for educational and research purposes.

---

