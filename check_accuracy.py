
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
file_path = (
    "data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8 (2).xls"
)

df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Separate features and target
X = df.drop("PerformanceRating", axis=1)
y = df["PerformanceRating"]

# 3. Remove unnecessary columns
columns_to_drop = [
    "EmpNumber",
    "Age_Group",
    "Experience_Group",
    "Attrition"
]

X = X.drop(columns=columns_to_drop, errors="ignore")

# 4. Identify numerical and categorical columns
numeric_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(include="object").columns

# 5. Load the saved encoder
encoder = joblib.load("encoder.pkl")

# 6. Encode categorical features
X_encoded = encoder.transform(X[categorical_features])

X_encoded_df = pd.DataFrame(
    X_encoded,
    columns=encoder.get_feature_names_out(categorical_features),
    index=X.index
)

# 7. Combine numerical and encoded features
X_numeric = X[numeric_features]

X_final = pd.concat(
    [X_numeric, X_encoded_df],
    axis=1
)

# 8. Load saved feature-column order
feature_columns = joblib.load("feature_columns.pkl")

X_final = X_final[feature_columns]

print("Preprocessing completed!")
print("Number of model features:", X_final.shape[1])

# 9. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 10. Load the saved model
model = joblib.load("gradient_boosting_model.pkl")

# 11. Make predictions
y_pred = model.predict(X_test)

# 12. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 45)
print("GRADIENT BOOSTING MODEL EVALUATION")
print("=" * 45)

print("Test samples:", len(y_test))
print(f"Accuracy: {accuracy * 100:.2f}%")

# 13. Classification report
print("\nCLASSIFICATION REPORT")
print("-" * 45)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[2, 3, 4],
        zero_division=0
    )
)

# 14. Confusion matrix
print("CONFUSION MATRIX")
print("-" * 45)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[2, 3, 4]
)

print("Class order: [2, 3, 4]")
print(cm)

print("\nEvaluation completed successfully!")
