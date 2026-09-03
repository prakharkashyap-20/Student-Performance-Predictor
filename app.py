import streamlit as st
import joblib
rf_model = joblib.load("rf_model.pkl")
st.title("Student Performance Predictor")
g1 = st.number_input("G1", min_value = 0, max_value = 20, step = 1)
g2 = st.number_input("G2", min_value = 0, max_value = 20, step = 1)
studytime = st.number_input("Study Time", min_value = 1, max_value = 4, step = 1)
failures = st.number_input("Failures", min_value = 0, max_value = 3, step = 1)
absences = st.number_input("Absences", min_value = 0, max_value = 93, step = 1)
predict = st.button("Predict Result")
if predict:
    input_data = [[studytime, g1, g2, failures, absences]]
    prediction = rf_model.predict(input_data)
    if prediction[0] == "Pass":
        st.success("🎉 Pass")
    else:
        st.warning("⚠️ At Risk")
    


