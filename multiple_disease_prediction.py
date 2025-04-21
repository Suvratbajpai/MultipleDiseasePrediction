# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:17:20 2025

@author: suvra
"""

import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Health Assistant",layout="wide",page_icon="🧑‍⚕️")

# loading the saved models
diabetes_model = pickle.load(open('./diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('./heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('./parkinsons_model.sav', 'rb'))
kidney_model = pickle.load(open('./kidney_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Kidney Disease Prediction',
                            'Disease-Symptom Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)
    

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

    # Kidney Disease Prediction Page
if selected == "Kidney Disease Prediction":

    st.title("Kidney Disease Prediction using ML")

    # Create input fields for the 16 required features
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.text_input('Age')

    with col2:
        bp = st.text_input('Blood Pressure')

    with col3:
        sg = st.text_input('Specific Gravity')

    with col4:
        su = st.text_input('Sugar')

    with col1:
        bgr = st.text_input('Blood Glucose Random')

    with col2:
        bu = st.text_input('Blood Urea')

    with col3:
        sc = st.text_input('Serum Creatinine')

    with col4:
        sod = st.text_input('Sodium')

    with col1:
        pot = st.text_input('Potassium')

    with col2:
        hemo = st.text_input('Hemoglobin')

    with col3:
        pcv = st.text_input('Packed Cell Volume')

    with col4:
        wc = st.text_input('White Blood Cell Count')

    with col1:
        htn = st.selectbox('Hypertension', ['no', 'yes'])

    with col2:
        dm = st.selectbox('Diabetes Mellitus', ['no', 'yes'])

    with col3:
        appet = st.selectbox('Appetite', ['poor', 'good'])

    with col4:
        pe = st.selectbox('Pedal Edema', ['no', 'yes'])

    # Code for prediction
    kidney_disease_diagnosis = ''

    if st.button("Kidney Disease Test Result"):

        try:
            # Convert categorical to numeric
            htn_val = 1 if htn == 'yes' else 0
            dm_val = 1 if dm == 'yes' else 0
            appet_val = 1 if appet == 'good' else 0
            pe_val = 1 if pe == 'yes' else 0

            # Create final input list
            input_data = [
                float(age), float(bp), float(sg), float(su), float(bgr),
                float(bu), float(sc), float(sod), float(pot), float(hemo),
                float(pcv), float(wc), htn_val, dm_val, appet_val, pe_val
            ]

            # Reshape and predict
            input_array = np.array(input_data).reshape(1, -1)
            prediction = kidney_model.predict(input_array)

            if prediction[0] == 1:
                kidney_disease_diagnosis = "The person has kidney disease"
            else:
                kidney_disease_diagnosis = "The person does not have kidney disease"

        except:
            kidney_disease_diagnosis = "Please fill all inputs correctly."

    st.success(kidney_disease_diagnosis)

    # Disease-Symptom Prediction Page
if selected == "Disease-Symptom Prediction":

    if "svm_disease_model.sav" not in st.session_state:
        with open('svm_disease_model.sav', 'rb') as model_file:
            st.session_state.svm_model = pickle.load(model_file)

    if "svm_label_encoder.sav" not in st.session_state:
        with open('svm_label_encoder.sav', 'rb') as encoder_file:
            st.session_state.label_encoder = pickle.load(encoder_file)

    svm_model = st.session_state.svm_model
    le = st.session_state.label_encoder

    df = pd.read_csv('Training.csv')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')  # Normalize column names
    df = df.loc[:, ~df.columns.str.contains('^unnamed')]
    symptom_list = df.columns.drop('prognosis').tolist()

    st.title(" Disease Prediction from Symptoms")
    selected_symptoms = st.multiselect("Select your symptoms", symptom_list)

    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning(" Please select at least one symptom.")
        else:
            # Prepare input vector
            input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
            input_array = np.array(input_data).reshape(1, -1)

            # Predict disease
            prediction = svm_model.predict(input_array)
            predicted_disease = le.inverse_transform(prediction)[0]
            st.success(f"🩺 **Predicted Disease**: `{predicted_disease}`")

            # Predict probabilities
            try:
                probs = svm_model.predict_proba(input_array)[0]  # Get probs for all classes
                top_indices = np.argsort(probs)[::-1]  # Sort by probability (desc)

                top_prob = probs[top_indices[0]]
                close_diseases = []

                for idx in top_indices:
                    if top_prob - probs[idx] <= 0.05:  # ≤5% difference
                        disease = le.inverse_transform([idx])[0]
                        close_diseases.append((disease, probs[idx]))

                if close_diseases:
                    st.subheader("Closest Matching Diseases (≤5% range):")
                    for d, p in close_diseases:
                        st.write(f"- {d}: {p * 100:.2f}%")
            except AttributeError:
                st.info("ℹ️ Probability estimates not available. Enable `probability=True` when training SVM.")
