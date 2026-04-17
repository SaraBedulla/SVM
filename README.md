# 🧠 Breast Cancer Classification using SVM

This project implements a **Support Vector Machine (SVM)** model to classify breast cancer tumors as **Benign (B)** or **Malignant (M)** using a structured medical dataset. The model includes hyperparameter tuning, cross-validation, and performance evaluation using multiple metrics.

---

## 📊 Dataset Information

- **Total Samples:** 569
- **Features:** 32 (including target)
- **Target Classes:**
  - Benign (B): 357 samples
  - Malignant (M): 212 samples

The dataset shows a slight class imbalance but is suitable for classification tasks.

---

## ⚙️ Model Overview

A **Support Vector Machine (SVM)** classifier is used with hyperparameter tuning via **GridSearchCV**.

### 🔧 Best Model Parameters:
- Kernel: `RBF`
- C: `1`
- Gamma: `scale`

The RBF kernel allows the model to capture non-linear relationships in the data.

---

## 🏋️ Model Training

- Hyperparameter tuning performed using **GridSearchCV**
- Cross-validation: **5-Fold CV**
- Scoring metric: **Accuracy**
- Data preprocessing includes feature scaling for optimal SVM performance

---

## 📈 Model Performance

### Cross-Validation Results:
- Mean Accuracy: **0.9758**
- CV Scores: `[0.967, 1.000, 0.967, 0.967, 0.978]`

### Test Set Results:
- Accuracy: **97%**

### Classification Report:
- **Benign**
  - Precision: 0.96
  - Recall: 1.00
  - F1-score: 0.98

- **Malignant**
  - Precision: 1.00
  - Recall: 0.93
  - F1-score: 0.96

### ROC AUC Score:
- **0.9947**

---

## 📊 Key Insights

- The model performs **very well with high accuracy and strong generalization**
- It shows **excellent class separation ability (ROC AUC ~ 0.99)**
- No false negatives for benign cases
- A small number of malignant cases are misclassified as benign (important in medical context)

---

## ⚠️ Limitations

- Slight class imbalance exists in the dataset
- A small number of malignant cases are missed (false negatives)
- RBF model is not directly interpretable for feature importance

---

## 🚀 Future Improvements

- Improve recall for malignant cases using class weighting
- Experiment with other models (Random Forest, XGBoost)
- Perform feature selection to reduce dimensionality
- Use recall-focused optimization for medical safety

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## 📌 Conclusion

The SVM model demonstrates strong predictive performance and high reliability for breast cancer classification. It is highly effective in distinguishing between benign and malignant cases, making it suitable as a baseline model for medical diagnostic support systems.

---
