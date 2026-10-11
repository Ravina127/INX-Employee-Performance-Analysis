# Employee Performance Analysis — INX Future Inc.

## Project Overview
This project analyzes employee performance data for INX Future Inc. using Python and Machine Learning. The goal is to identify important factors associated with employee performance and build a model to predict the `PerformanceRating`.

## Objectives
- Analyze employee performance across departments.
- Identify important factors related to employee performance.
- Compare different machine learning classification models.
- Build a model to predict employee performance ratings.
- Provide insights and recommendations based on the analysis.

## Dataset
- **Rows:** 1,200
- **Columns:** 28
- **Target Variable:** `PerformanceRating`
- **Target Classes:** 2, 3, and 4
- **Missing Values:** None found
- **Duplicate Records:** None found

The original Excel dataset is stored locally in the `data/` folder and excluded from Git using `.gitignore`.

## Tools and Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- VS Code

## Project Workflow
1. Data loading and inspection
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature engineering and encoding
5. Model training and comparison
6. Model evaluation
7. Saving the trained model
8. Streamlit application for prediction

## Model Performance

The Gradient Boosting Classifier was evaluated on a test set of 240 records using an 80/20 stratified train-test split.

| Metric | Result |
|---|---:|
| Accuracy | **93.33%** |
| Weighted Precision | 0.93 |
| Weighted Recall | 0.93 |
| Weighted F1-score | 0.93 |
| Macro Precision | 0.93 |
| Macro Recall | 0.87 |
| Macro F1-score | 0.90 |

### Class-wise Performance

| Performance Rating | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 2 | 0.89 | 0.87 | 0.88 | 39 |
| 3 | 0.94 | 0.97 | 0.96 | 175 |
| 4 | 0.95 | 0.77 | 0.85 | 26 |

### Confusion Matrix

Class order: `[2, 3, 4]`

| Actual / Predicted | Rating 2 | Rating 3 | Rating 4 |
|---|---:|---:|---:|
| Rating 2 | 34 | 5 | 0 |
| Rating 3 | 4 | 170 | 1 |
| Rating 4 | 0 | 6 | 20 |

The model performs best for Rating 3. The recall for Rating 4 is lower at 77%, meaning some actual Rating 4 cases are predicted as Rating 3.

## Important Features
The feature-importance analysis identified the following important features:

1. `EmpEnvironmentSatisfaction`
2. `EmpLastSalaryHikePercent`
3. `YearsSinceLastPromotion`
4. `EmpDepartment_Development`
5. `EmpWorkLifeBalance`

Feature importance describes how the trained model uses features. It does not prove that a feature causes employee performance.

## How to Run the Project

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

For reading the Excel `.xls` dataset in the evaluation script:

```bash
python -m pip install xlrd
```

### 2. Run the Streamlit App

```bash
python -m streamlit run app.py
```

### 3. Evaluate the Model from Terminal

```bash
python check_accuracy.py
```

This script prints accuracy, precision, recall, F1-score, and the confusion matrix.

## Project Structure

```text
employee analysis/
├── app.py
├── check_accuracy.py
├── encoder.pkl
├── feature_columns.pkl
├── gradient_boosting_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── notebooks/
│   └── Employee_Performance_Analysis.ipynb
└── screenshots/
```

## Limitations
- The target classes are imbalanced, with Rating 3 as the majority class.
- Overall accuracy does not show the full picture of class-wise performance.
- Rating 4 has a lower recall than Rating 2 and Rating 3.
- The results depend on the dataset, preprocessing, and train-test split.
- The model is an educational prototype and requires further validation before real-world use.

## Conclusion
The Gradient Boosting Classifier achieved **93.33% accuracy** on the 240-record test set. It performed strongly for Rating 3, while Rating 4 recall was lower. Further validation and class-wise evaluation are recommended.

## Disclaimer
This project is for educational and analytical purposes. Model predictions should not be used as the sole basis for hiring, promotion, appraisal, or disciplinary decisions.
