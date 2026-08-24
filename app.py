import streamlit as st
import pandas as pd
import numpy as np
import pickle






# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f8ff;
}

h1 {
    color: #003366;
    text-align: center;
}

h2 {
    color: #003366;
}

h3 {
    color: #003366;
}

.stButton > button {
    background-color: #0099ff;
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #007acc;
    color: white;
}

div[data-testid="stNumberInput"] input {
    border-radius: 8px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.title("🩺 AI Powered Breast Cancer Prediction")

st.markdown(
    """
    <div style="text-align:center; font-size:18px;">
        Breast Cancer Classification using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

st.info(
    "This application uses Logistic Regression to classify a tumor "
    "as Benign or Malignant based on the input measurements."
)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("columns.pkl", "rb") as file:
        columns = pickle.load(file)

except FileNotFoundError:

    st.error(
        "❌ model.pkl or columns.pkl was not found.\n\n"
        "Please keep both files in the same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(f"❌ Error loading model: {e}")

    st.stop()


# =========================================================
# MODEL INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Logistic Regression")

with col2:
    st.metric("Features", len(columns))

with col3:
    st.metric("Accuracy", "96.50%")


st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("🔬 Enter Tumor Measurements")

st.write(
    "Enter the values for the 30 features used by the trained model."
)


data = {}


# =========================================================
# FEATURE INPUTS
# =========================================================

col1, col2, col3 = st.columns(3)


for i, column in enumerate(columns):

    # Use 3 columns for the input fields

    if i % 3 == 0:

        with col1:

            data[column] = st.number_input(
                label=column,
                value=0.0,
                step=0.01,
                format="%.5f"
            )

    elif i % 3 == 1:

        with col2:

            data[column] = st.number_input(
                label=column,
                value=0.0,
                step=0.01,
                format="%.5f"
            )

    else:

        with col3:

            data[column] = st.number_input(
                label=column,
                value=0.0,
                step=0.01,
                format="%.5f"
            )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

if st.button("🔮 Predict Breast Cancer Result"):

    try:

        # Create DataFrame
        input_data = pd.DataFrame([data])

        # Make sure columns are in exactly the same order
        input_data = input_data[columns]

        # Prediction
        prediction = model.predict(input_data)[0]

        # Probability
        probability = model.predict_proba(input_data)[0]

        malignant_probability = probability[1] * 100
        benign_probability = probability[0] * 100


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("📊 Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ Prediction: MALIGNANT"
            )

            st.markdown(
                f"""
                <div class="result-box">

                <h2>Malignant Tumor</h2>

                <p>
                Model confidence for Malignant:
                <strong>{malignant_probability:.2f}%</strong>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.success(
                "✅ Prediction: BENIGN"
            )

            st.markdown(
                f"""
                <div class="result-box">

                <h2>Benign Tumor</h2>

                <p>
                Model confidence for Benign:
                <strong>{benign_probability:.2f}%</strong>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # PROBABILITY DISPLAY
        # =================================================

        st.write("")

        probability_col1, probability_col2 = st.columns(2)

        with probability_col1:

            st.metric(
                "Benign Probability",
                f"{benign_probability:.2f}%"
            )

        with probability_col2:

            st.metric(
                "Malignant Probability",
                f"{malignant_probability:.2f}%"
            )


        # =================================================
        # PROGRESS BARS
        # =================================================

        st.write("### Prediction Probability")

        st.write(
            f"Benign: {benign_probability:.2f}%"
        )

        st.progress(
            int(round(benign_probability))
        )

        st.write(
            f"Malignant: {malignant_probability:.2f}%"
        )

        st.progress(
            int(round(malignant_probability))
        )


    except Exception as e:

        st.error(
            f"❌ Prediction Error: {e}"
        )


# =========================================================
# PROJECT HIGHLIGHTS
# =========================================================

st.divider()

st.subheader("🚀 Project Highlights")

col1, col2 = st.columns(2)


with col1:

    st.markdown("""
    ### 📊 Data Processing

    - Data Acquisition
    - Data Cleaning
    - Missing Value Checking
    - ID Column Removal
    - Target Encoding
    - Feature Scaling
    - StandardScaler
    """)


with col2:

    st.markdown("""
    ### 🤖 Machine Learning

    - Logistic Regression
    - Train-Test Split
    - Stratified Sampling
    - Classification
    - Confusion Matrix
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    """)


