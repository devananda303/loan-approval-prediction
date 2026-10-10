
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="LoanWise AI",
    page_icon="🏦",
    layout="wide"
)


# -----------------------------------
# CUSTOM CSS - VISIBLE TEXT
# -----------------------------------

st.markdown("""
<style>

/* Main page */
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #f4f7fb !important;
    color: #172033 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #17365d !important;
}

/* Normal text */
.stMarkdown,
.stMarkdown p,
.stMarkdown span,
.stText,
p {
    color: #172033 !important;
}

/* Form labels and widget labels */
label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] p {
    color: #172033 !important;
}

/* Text and number input fields */
input,
textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    background-color: #ffffff !important;
}

/* Dropdown selected values */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
}

[data-baseweb="select"] *,
[data-testid="stSelectbox"] *,
[data-testid="stMultiSelect"] * {
    color: #172033 !important;
}

/* Dropdown menu options */
[role="listbox"],
[role="option"],
[data-baseweb="popover"] li {
    color: #172033 !important;
    background-color: #ffffff !important;
}

/* Radio buttons and checkboxes */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: #172033 !important;
}

/* Slider labels and values */
[data-testid="stSlider"] *,
[data-testid="stSlider"] label {
    color: #172033 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e8eef7 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #172033 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 16px !important;
    border-radius: 12px !important;
    border: 1px solid #dce4ef !important;
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #172033 !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button,
[data-testid="stFormSubmitButton"] button {
    background-color: #17365d !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

.stButton button *,
.stDownloadButton button *,
[data-testid="stFormSubmitButton"] button * {
    color: #ffffff !important;
}

/* Prediction result boxes */
.success-box {
    background-color: #d9fbe5 !important;
    color: #14532d !important;
    padding: 20px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: bold;
}

.reject-box {
    background-color: #fee2e2 !important;
    color: #991b1b !important;
    padding: 20px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: bold;
}

/* Informational panels */
.info-box {
    background-color: #ffffff !important;
    color: #172033 !important;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce4ef;
}

.info-box h3,
.info-box p {
    color: #172033 !important;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# 3. GENERATE SAMPLE DATASET
# -----------------------------------

@st.cache_data
def generate_data(n=4000):
    np.random.seed(42)

    applicant_income = np.random.randint(15000, 150001, n)
    coapplicant_income = np.random.choice(
        [0, 5000, 10000, 15000, 20000, 30000, 40000],
        n
    )

    loan_amount = np.random.randint(50000, 1000001, n)
    loan_term = np.random.choice(
        [120, 180, 240, 300, 360, 480],
        n
    )

    # Actual numerical credit score feature
    credit_score = np.clip(
        np.random.normal(650, 80, n),
        300,
        850
    ).astype(int)

    credit_history = np.random.choice(
        [0, 1], n, p=[0.2, 0.8]
    )

    employment = np.random.choice(
        ["Salaried", "Self-Employed"],
        n
    )

    education = np.random.choice(
        ["Graduate", "Not Graduate"],
        n
    )

    married = np.random.choice(
        ["Yes", "No"],
        n
    )

    dependents = np.random.choice(
        ["0", "1", "2", "3+"],
        n
    )

    property_area = np.random.choice(
        ["Urban", "Semiurban", "Rural"],
        n
    )

    # -----------------------------------
    # 4. GENERATE SAMPLE APPROVAL LABELS
    # -----------------------------------
    # These rules create illustrative training
    # labels; they are not real bank policies.

    score = (
        (credit_score - 600) / 35
        + credit_history * 1.8
        + ((applicant_income + coapplicant_income) / 1000) * 0.015
        - (loan_amount / 100000) * 0.65
        + (loan_term / 360) * 0.3
        + (employment == "Salaried") * 0.2
        + (education == "Graduate") * 0.2
        + (married == "Yes") * 0.1
        + (property_area == "Semiurban") * 0.3
        + np.random.normal(0, 1.5, n)
    )

    loan_status = (score > 1.5).astype(int)

    data = pd.DataFrame({
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_Score": credit_score,
        "Credit_History": credit_history,
        "Employment": employment,
        "Education": education,
        "Married": married,
        "Dependents": dependents,
        "Property_Area": property_area,
        "Loan_Status": loan_status
    })

    return data


df = generate_data()


# -----------------------------------
# 5. SELECT FEATURES AND TARGET
# -----------------------------------

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_Score",
    "Credit_History"
]

categorical_features = [
    "Employment",
    "Education",
    "Married",
    "Dependents",
    "Property_Area"
]

X = df[numeric_features + categorical_features]
y = df["Loan_Status"]


# -----------------------------------
# 6. PREPROCESS THE DATA
# -----------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# -----------------------------------
# 7. SPLIT DATA INTO TRAINING AND TESTING
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 8. CREATE AND TRAIN MODEL
# -----------------------------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    )
])

model.fit(X_train, y_train)


# -----------------------------------
# 9. EVALUATE MODEL
# -----------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)


# -----------------------------------
# 10. SIDEBAR NAVIGATION
# -----------------------------------

st.sidebar.title("🏦 LoanWise AI")
st.sidebar.write("Loan Approval Prediction System")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Loan Prediction",
        "Model Evaluation",
        "Dataset Explorer"
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "Educational project using synthetic data."
)


# -----------------------------------
# 11. HOME PAGE
# -----------------------------------

if page == "Home":

    st.title("🏦 Welcome to LoanWise AI")

    st.write(
        "A machine learning application that predicts "
        "loan approval using applicant information, "
        "including credit score."
    )

    st.markdown("""
    <div class="info-box">
        <h3>About this project</h3>
        LoanWise AI uses a Random Forest machine learning
        model to learn patterns from a generated loan dataset.
        You can enter applicant details, view a prediction,
        explore the sample data, and evaluate the model.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Dataset overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Applications", len(df))
    col2.metric("Input Features", len(numeric_features) + len(categorical_features))
    col3.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Loan Approval Distribution")

    approval_counts = df["Loan_Status"].map({
        0: "Rejected",
        1: "Approved"
    }).value_counts().reset_index()

    approval_counts.columns = ["Loan Status", "Count"]

    fig = px.pie(
        approval_counts,
        names="Loan Status",
        values="Count",
        title="Sample Loan Status Distribution",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The dataset is artificially generated. "
        "The results do not represent real bank decisions."
    )


# -----------------------------------
# 12. LOAN PREDICTION PAGE
# -----------------------------------

elif page == "Loan Prediction":

    st.title("🔍 Loan Approval Prediction")

    st.write(
        "Enter the applicant details below to get a "
        "prediction from the trained machine learning model."
    )

    with st.form("loan_form"):

        st.subheader("Applicant Information")

        col1, col2 = st.columns(2)

        with col1:
            applicant_income = st.number_input(
                "Applicant Monthly Income (₹)",
                min_value=0,
                max_value=10000000,
                value=50000,
                step=1000
            )

            coapplicant_income = st.number_input(
                "Co-applicant Monthly Income (₹)",
                min_value=0,
                max_value=10000000,
                value=0,
                step=1000
            )

            education = st.selectbox(
                "Education",
                ["Graduate", "Not Graduate"]
            )

            employment = st.selectbox(
                "Employment Type",
                ["Salaried", "Self-Employed"]
            )

            married = st.selectbox(
                "Married",
                ["Yes", "No"]
            )

        with col2:
            credit_score = st.slider(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=700,
                help="Enter a score from 300 to 850."
            )

            credit_history = st.selectbox(
                "Credit History",
                options=[1, 0],
                format_func=lambda x: (
                    "Good / Available" if x == 1
                    else "Poor / Not Available"
                )
            )

            loan_amount = st.number_input(
                "Requested Loan Amount (₹)",
                min_value=1000,
                max_value=10000000,
                value=200000,
                step=10000
            )

            loan_term = st.selectbox(
                "Loan Term (Months)",
                [120, 180, 240, 300, 360, 480],
                index=4
            )

            dependents = st.selectbox(
                "Number of Dependents",
                ["0", "1", "2", "3+"]
            )

            property_area = st.selectbox(
                "Property Area",
                ["Urban", "Semiurban", "Rural"]
            )

        submitted = st.form_submit_button(
            "Predict Loan Status",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_Score": credit_score,
            "Credit_History": credit_history,
            "Employment": employment,
            "Education": education,
            "Married": married,
            "Dependents": dependents,
            "Property_Area": property_area
        }])

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]
        approval_probability = float(
            probabilities[list(model.classes_).index(1)]
        )

        st.subheader("Prediction Result")

        if prediction == 1:
            st.markdown(
                '<div class="success-box">'
                'Predicted Result: Loan Approved'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="reject-box">'
                'Predicted Result: Loan Rejected'
                '</div>',
                unsafe_allow_html=True
            )

        st.metric(
            "Model-estimated approval probability",
            f"{approval_probability * 100:.2f}%"
        )

        st.progress(approval_probability)

        st.caption(
            "This probability is the model's estimate, not a "
            "guarantee of approval or a real bank assessment."
        )

        st.write("Applicant details entered:")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


# -----------------------------------
# 13. MODEL EVALUATION PAGE
# -----------------------------------

elif page == "Model Evaluation":

    st.title("📊 Model Evaluation")

    st.write(
        "These metrics measure how the model performed "
        "on the held-out test portion of the synthetic dataset."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Precision", f"{precision * 100:.2f}%")
    col3.metric("Recall", f"{recall * 100:.2f}%")
    col4.metric("F1 Score", f"{f1 * 100:.2f}%")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Rejected", "Actual Approved"],
        columns=["Predicted Rejected", "Predicted Approved"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels={
            "x": "Predicted Status",
            "y": "Actual Status",
            "color": "Number of Applications"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Rejected", "Approved"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.info(
        "High test accuracy does not prove that this model "
        "would work well on real loan applications. "
        "It was trained and tested on generated data."
    )


# -----------------------------------
# 14. DATASET EXPLORER PAGE
# -----------------------------------

elif page == "Dataset Explorer":

    st.title("📁 Dataset Explorer")

    st.write(
        "Explore the generated loan application dataset."
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Credit Score Distribution")

    fig = px.histogram(
        df,
        x="Credit_Score",
        nbins=30,
        title="Distribution of Credit Scores"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Credit Score by Loan Status")

    credit_summary = (
        df.groupby("Loan_Status")["Credit_Score"]
        .mean()
        .reset_index()
    )

    credit_summary["Loan_Status"] = (
        credit_summary["Loan_Status"]
        .map({0: "Rejected", 1: "Approved"})
    )

    fig = px.bar(
        credit_summary,
        x="Loan_Status",
        y="Credit_Score",
        title="Average Credit Score by Sample Loan Status",
        text_auto=".1f"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Download Dataset")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Dataset as CSV",
        data=csv_data,
        file_name="loanwise_ai_dataset.csv",
        mime="text/csv"
    )

# -----------------------------------
# 15. FOOTER
# -----------------------------------

st.divider()

st.caption(
    "LoanWise AI | College Machine Learning Project | "
    "Predictions use synthetic data and are not financial advice."
)
