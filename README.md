# Breast-Cancer-Detection-using-ML
Random Forest–based breast cancer detection using the Breast Cancer Wisconsin (Diagnostic) dataset, with stratified k‑fold cross‑validation, grid‑search hyperparameter tuning and evaluation.
# Breast Cancer Detection – Random Forest 

This repository contains my implementation of a machine learning classifier to diagnose breast cancer using the Breast Cancer Wisconsin (Diagnostic) dataset. The project was completed for the ML SAT 5114 course assignment.

## 1. Project overview

- **Goal:** Classify breast tumors as malignant or benign using a supervised learning model.
- **Dataset:** `load_breast_cancer()` from scikit-learn (569 samples, 30 numeric features, binary target: 0 = malignant, 1 = benign).
- **Model:** Random Forest classifier with stratified k-fold cross-validation and hyperparameter tuning via grid search.
- **Outputs:** Final test performance (accuracy, precision, recall, F1-score, sensitivity, specificity), confusion matrix, and feature importance plot.

## 2. Repository structure

├── Breast-Cancer-detection.ipynb               # Main Jupyter notebook
├── README.md                                   # Project description
└── requirements.txt                            # Key Python dependencies
