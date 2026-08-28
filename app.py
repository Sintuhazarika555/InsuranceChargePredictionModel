import streamlit as st 
import pandas as pd
import joblib


insuranceModel = joblib.load('regression_pipeline.pkl')
model = insuranceModel['model']
scaler = insuranceModel['scaler']
expected_columns = insuranceModel['features']

st.title("Insurance Charges Prediction Model")
st.markdown("Provide the following details :")


age = st.slider("AGE", 18, 100, 40)
bmi = st.slider("BMI", 10.0, 55.0, 25.0)  
sex = st.selectbox("SEX", ["M", "F"])
children = st.number_input("NUMBER OF CHILDREN", min_value=0, max_value=10, value=2, step=1)
smoker = st.selectbox("DO YOU SMOKE?", ["Y", "N"])
region = st.selectbox("SELECT REGION", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict"):
    #  Match dictionary values to UI inputs
    raw_input = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'is_female': 1 if sex == 'F' else 0,
        'is_smoker': 1 if smoker == 'Y' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'bmi_categories_obese': 1 if bmi > 29.9 else 0
    }
    
    #  Construct DataFrame cleanly from dict 
    input_df = pd.DataFrame([raw_input])
    
    #  Align with expected model features
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    
    #  Scale ONLY the numerical columns that the scaler knows about
    numerical_cols = list(scaler.feature_names_in_)
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Generate prediction
    prediction = model.predict(input_df)[0]
    
    #  Display in Streamlit UI instead of terminal print
    st.success(f"Predicted Insurance Charge: **${prediction:,.2f}**")
