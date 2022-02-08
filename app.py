import numpy as np
import streamlit as st
import joblib
from xgboost import XGBClassifier
# import pyautogui
import sklearn
def scaling(datapoint):
    scaler = joblib.load('Diabetes Model/standard_scaler.save')
    data = scaler.transform(datapoint)
    return data

def prediction(data):
    xgb = XGBClassifier()
    xgb.load_model("./Diabetes Model/diabetes_model_new.json")
    result = xgb.predict(data)
    return result

def get_input():
    st.title("Diabetes Prediction using ML")
    col1, col2 = st.columns(2)
    with col1:
        pregnancy = st.number_input("Number of Pregnancies [0-20]", 0, 20,step=1)
        glucose = st.number_input("Glucose Concentration in 2 hrs [0-199]",0, 199, step=1)
        bp = st.number_input("Diastolic Blood Pressure (mm Hg) [0-130]", 0, 130, step=1)
        skin_thick = st.number_input("Triceps Skin Thickness (mm) [0-99]", 0, 99, step=1)
    with col2:
        insulin = st.number_input("2-Hour serum insulin (muU/ml) [0-846]",0,846, step=1)
        bmi = st.number_input("Body Mass Index [0.0-68.0]", 0.0, 68.0)
        di_ped = st.number_input("Diabetes Pedigree Function [0.00-2.42]", 0.00, 2.42)
        age = st.number_input("Age [21-81]", 21, 81, step=1)


    datapoint = np.array([[pregnancy, glucose, bp, skin_thick, insulin, bmi, di_ped, age]])
    return datapoint

if __name__ == '__main__':
    datapoint = get_input()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    data_main = scaling(datapoint)
    with c3:
        btn = st.button('Predict')
        if btn:
            res = prediction(data_main)
            if res == 0:
                    st.write("Congratulations! You have no Diabetes")
            else:
                    st.write("Sorry! You have Diabetes")
        else:
            pass
    # with c4:
        # if st.button("Reset"):
        #     pyautogui.hotkey("ctrl", "F5")
