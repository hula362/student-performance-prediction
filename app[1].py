
import streamlit as st
import pandas as pd
import joblib

# Load the saved ML model and feature list
model = joblib.load("student_model.pkl")
features = joblib.load("student_features.pkl")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")
st.write(
    "Enter the student's details below to estimate "
    "their final academic grade using Machine Learning."
)

st.info(
    "This is an educational prediction tool. "
    "The result is an estimate, not a guarantee of performance."
)

# Collect student information
st.subheader("Student Details")

with st.form("prediction_form"):

    age = st.slider("Age", 15, 22, 17)

    studytime = st.selectbox(
        "Weekly Study Time",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Less than 2 hours",
            2: "2–5 hours",
            3: "5–10 hours",
            4: "More than 10 hours"
        }[x]
    )

    failures = st.number_input(
        "Previous Class Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=0
    )

    famrel = st.slider("Family Relationship Quality", 1, 5, 4)
    freetime = st.slider("Free Time", 1, 5, 3)
    goout = st.slider("Going Out with Friends", 1, 5, 3)
    health = st.slider("Current Health Rating", 1, 5, 4)

    schoolsup = st.selectbox(
        "Extra School Support",
        ["yes", "no"]
    )

    famsup = st.selectbox(
        "Family Educational Support",
        ["yes", "no"]
    )

    paid = st.selectbox(
        "Extra Paid Classes",
        ["yes", "no"]
    )

    activities = st.selectbox(
        "Extracurricular Activities",
        ["yes", "no"]
    )

    higher = st.selectbox(
        "Wants to Pursue Higher Education",
        ["yes", "no"]
    )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        type="primary",
        use_container_width=True
    )

# Make a prediction when the form is submitted
if submitted:

    student_data = pd.DataFrame([{
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "health": health,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "higher": higher
    }])

    # Ensure the input columns are in the correct order
    student_data = student_data[features]

    prediction = model.predict(student_data)[0]

    # Grades in the dataset range from 0 to 20
    prediction = max(0, min(20, prediction))

    st.subheader("Prediction Result")

    st.metric(
        label="Estimated Final Grade",
        value=f"{prediction:.2f} / 20"
    )

    st.progress(int(round(prediction / 20 * 100)))

    if prediction >= 16:
        st.success("The model estimates a high final grade.")
    elif prediction >= 10:
        st.info("The model estimates a moderate passing grade.")
    else:
        st.warning(
            "The model estimates a grade below 10. "
            "Additional academic support may be helpful."
        )

    st.caption(
        "This estimate is based on patterns in the training "
        "dataset and may not reflect an individual student's "
        "actual result."
    )
