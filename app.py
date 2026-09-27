import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    threshold = joblib.load("threshold.pkl")

    return model, scaler, threshold


model, scaler, threshold = load_artifacts()


# ============================================================
# HEADER
# ============================================================

st.title("📉 Customer Churn Prediction")

st.markdown(
    """
    Predict whether a telecom customer is likely to churn
    based on their account and service information.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("👤 Customer Information")


gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)


senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)


partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)


dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)


tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)


phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)


multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No"]
)


internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)


online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No"]
)


online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No"]
)


device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No"]
)


tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No"]
)


streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)


streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)


contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)


payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=1000.0,
    value=70.0
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "gender": [gender],

    "SeniorCitizen": [senior_citizen],

    "Partner": [partner],

    "Dependents": [dependents],

    "tenure": [tenure],

    "PhoneService": [phone_service],

    "MultipleLines": [multiple_lines],

    "InternetService": [internet_service],

    "OnlineSecurity": [online_security],

    "OnlineBackup": [online_backup],

    "DeviceProtection": [device_protection],

    "TechSupport": [tech_support],

    "StreamingTV": [streaming_tv],

    "StreamingMovies": [streaming_movies],

    "Contract": [contract],

    "PaperlessBilling": [paperless_billing],

    "PaymentMethod": [payment_method],

    "MonthlyCharges": [monthly_charges]
})


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_input(data):

    data = data.copy()

    # --------------------------------------------------------
    # Binary encoding
    # --------------------------------------------------------

    binary_map = {
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0
    }

    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    for col in binary_cols:
        data[col] = data[col].map(binary_map)


    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    data = pd.get_dummies(
        data,
        columns=[
            "InternetService",
            "Contract",
            "PaymentMethod"
        ],
        drop_first=True,
        dtype=int
    )


    # --------------------------------------------------------
    # Match EXACT training columns
    # --------------------------------------------------------

    expected_columns = scaler.feature_names_in_

    data = data.reindex(
        columns=expected_columns,
        fill_value=0
    )


    return data


# ============================================================
# PREDICTION
# ============================================================

st.subheader("🔮 Churn Prediction")

if st.button(
    "Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    # Preprocess
    processed_data = preprocess_input(input_data)

    # Scale
    scaled_data = scaler.transform(processed_data)

    # Probability
    probability = model.predict_proba(
        scaled_data
    )[0][1]

    # Apply saved threshold
    prediction = int(
        probability >= threshold
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )


    with col2:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )


    with col3:

        if prediction == 1:

            st.metric(
                "Risk Level",
                "HIGH"
            )

        else:

            st.metric(
                "Risk Level",
                "LOW"
            )


    # ========================================================
    # RESULT MESSAGE
    # ========================================================

    if prediction == 1:

        st.error(
            "⚠️ High Churn Risk"
        )

        st.warning(
            "This customer has a predicted churn probability "
            f"of {probability:.1%}. Consider targeted retention "
            "actions."
        )

    else:

        st.success(
            "✅ Low Churn Risk"
        )

        st.info(
            "This customer has a predicted churn probability "
            f"of {probability:.1%}."
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.subheader("Churn Probability")

    st.progress(
        float(probability)
    )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    st.subheader("Customer Summary")

    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.write(
            f"**Tenure:** {tenure} months"
        )

        st.write(
            f"**Contract:** {contract}"
        )

        st.write(
            f"**Internet Service:** {internet_service}"
        )


    with summary_col2:

        st.write(
            f"**Monthly Charges:** ${monthly_charges:.2f}"
        )

        st.write(
            f"**Payment Method:** {payment_method}"
        )

        st.write(
            f"**Tech Support:** {tech_support}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("ℹ️ About this model"):

    st.write(
        """
        This application uses the trained Logistic Regression model
        from the Customer Churn Prediction project.

        The model was selected after comparing Logistic Regression,
        Random Forest and XGBoost.

        A decision threshold of 0.40 is used to identify customers
        at higher risk of churn.
        """
    )