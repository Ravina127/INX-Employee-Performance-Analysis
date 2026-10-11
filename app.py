
import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("gradient_boosting_model.pkl")
encoder = joblib.load("encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Numerical features used during training
numeric_features = [
    "Age",
    "DistanceFromHome",
    "EmpEducationLevel",
    "EmpEnvironmentSatisfaction",
    "EmpHourlyRate",
    "EmpJobInvolvement",
    "EmpJobLevel",
    "EmpJobSatisfaction",
    "NumCompaniesWorked",
    "EmpLastSalaryHikePercent",
    "EmpRelationshipSatisfaction",
    "TotalWorkExperienceInYears",
    "TrainingTimesLastYear",
    "EmpWorkLifeBalance",
    "ExperienceYearsAtThisCompany",
    "ExperienceYearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

st.set_page_config(
    page_title="Employee Performance Prediction",
    page_icon="📊"
)

st.title("Employee Performance Prediction")
st.write("Enter employee details to predict the performance rating.")

with st.form("employee_form"):
    st.subheader("Numerical Details")

    numeric_values = {}

    for feature in numeric_features:
        numeric_values[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    st.subheader("Categorical Details")

    categorical_values = {}

    for feature, categories in zip(
        encoder.feature_names_in_,
        encoder.categories_
    ):
        categorical_values[feature] = st.selectbox(
            feature,
            categories.tolist()
        )

    submitted = st.form_submit_button("Predict Performance")

if submitted:
    new_employee = pd.DataFrame([{
        **numeric_values,
        **categorical_values
    }])

    numeric_data = new_employee[numeric_features]

    encoded_data = pd.DataFrame(
        encoder.transform(
            new_employee[list(encoder.feature_names_in_)]
        ),
        columns=encoder.get_feature_names_out()
    )

    processed_data = pd.concat(
        [
            numeric_data.reset_index(drop=True),
            encoded_data.reset_index(drop=True)
        ],
        axis=1
    )

    processed_data = processed_data[feature_columns]

    prediction = model.predict(processed_data)[0]

    st.success(f"Predicted Performance Rating: {prediction}")