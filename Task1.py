import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import subprocess

# =================== PAGE SETTINGS (MUST BE FIRST)=================
st.set_page_config(page_title="Credit Scoring App", page_icon="💳", layout="wide")

# ==================== SIDEBAR ====================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go To:", ["Dashboard", "Predict", "Chatbot"])

st.sidebar.markdown("---")
st.sidebar.write("📂 Dataset Loaded: `german_credit_data.csv`")

# ==================== LOAD DATA ====================
df = pd.read_csv("german_credit_data.csv")

# Convert categorical to numerical
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Identify target column
target_col = "Creditability" if "Creditability" in df.columns else df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ==================== DASHBOARD PAGE ====================
if page == "Dashboard":
    st.title("💳 Credit Scoring Dashboard")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Accuracy", f"{accuracy_score(y_test, predictions):.2f}")
    col2.metric("🎯 Precision", f"{precision_score(y_test, predictions, average='weighted'):.2f}")
    col3.metric("📌 Recall", f"{recall_score(y_test, predictions, average='weighted'):.2f}")
    col4.metric("⭐ F1 Score", f"{f1_score(y_test, predictions, average='weighted'):.2f}")

    try:
        roc_val = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        st.success(f"🔥 ROC-AUC Score: {roc_val:.2f}")
    except:
        st.info("ROC-AUC not available for this dataset.")

    st.subheader("🌟 Feature Importance")
    importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    st.dataframe(importance.sort_values(by="Importance", ascending=False))

# ==================== PREDICT PAGE ====================
elif page == "Predict":
    st.title("🔍 Predict Creditworthiness")

    inputs = {}
    for col in X.columns:
        inputs[col] = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))

    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]

    st.markdown("---")
    if pred == 1:
        st.success("✅ This person is **Creditworthy** (Low Risk).")
    else:
        st.error("⚠️ High Credit Risk detected. Loan approval should be careful.")
# ==================== CHATBOT PAGE ====================
elif page == "Chatbot":
    st.title("🤖 Smart Credit Chatbot (Fast Response)")

    OLLAMA_MODEL = "phi3"   # model must exist in your local `ollama list`

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def ask_ollama(prompt):
        # Use shell=True (Windows Fix) + input piped correctly
        result = subprocess.run(
            f"ollama run {OLLAMA_MODEL}",
            input=prompt,
            capture_output=True,
            text=True,
            shell=True
        )

        # Debug check
        if result.stderr:
            return "⚠️ **Error:** " + result.stderr.strip()

        return result.stdout.strip() if result.stdout else "⚠️ No response received"

    # Display previous chat
    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

    # User input box
    user_msg = st.chat_input("Ask anything about credit score, loans, EMI...")

    if user_msg:
        st.chat_message("user").write(user_msg)

        prompt = f"You are a helpful credit advisor. Reply simply.\nUser: {user_msg}"

        bot_reply = ask_ollama(prompt)

        st.chat_message("assistant").write(bot_reply)

        st.session_state.chat_history.append(("user", user_msg))
        st.session_state.chat_history.append(("assistant", bot_reply))

    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

