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

## Methods (brief)

1. **Exploratory Data Analysis**  
   - Created a pandas DataFrame, checked data types and missing values (none).  
   - Reviewed summary statistics, plotted class distribution, selected feature distributions by class, and a correlation heatmap.

2. **Train–test split**  
   - 80/20 split with train_test_split, stratify=y, random_state=42 (455 train, 114 test samples).

3. **Model and tuning**  
   - RandomForestClassifier(random_state=42, n_jobs=-1).  
   - Hyperparameter grid over n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.  
   - Stratified 5‑fold CV with GridSearchCV (scoring = accuracy).

4. **Evaluation**  
   - Best model refit on the full training set.  
   - Metrics on the test set: accuracy, precision, recall, F1‑score, sensitivity, specificity, confusion matrix, classification report.  
   - Top 10 feature importances plotted for interpretation.

## Key results (test set)

- **Accuracy:** 0.9474  
- **Precision:** 0.9583  
- **Recall:** 0.9583  
- **F1‑score:** 0.9583  
- **Sensitivity (TPR):** 0.9583  
- **Specificity (TNR):** 0.9286  

The model correctly classified 39/42 malignant and 69/72 benign tumors, with size and shape features (e.g., worst perimeter and worst area) among the most important.

## How to run the pipeline

'''bash
pip install -r requirements.txt

jupyter notebook
