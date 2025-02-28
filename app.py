import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load CO₂ Model & Machine Failure Model
co2_model = joblib.load("models/co2_model.pkl")
co2_feature_names = joblib.load("models/co2_feature_names.pkl")

failure_model = joblib.load("models/machine_failure_model.pkl")
failure_feature_names = ["Temperature", "Vibration", "Working_Hours"]

# Streamlit UI Setup (DARK THEME)
st.set_page_config(page_title="Machine & CO₂ Prediction App", layout="wide")

# Custom Dark Theme Styling
st.markdown("""
    <style>
    body { background-color: #121212; color: white; }
    .stApp { background-color: #121212; padding: 20px; border-radius: 10px; }
    .title { text-align: center; font-size: 30px; color: #00c8ff; }
    .subtitle { text-align: center; font-size: 20px; color: #aaaaaa; }
    .upload { text-align: center; }
    .stDataFrame { color: white; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>⚙️ Machine Failure & CO₂ Emission Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>📊 Predict Machine Failures & CO₂ Emissions</p>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.header("Navigation")
nav_option = st.sidebar.radio("📌 Select Section:", 
                              ["Machine Failure Prediction", "CO₂ Prediction", 
                               "Batch Prediction", "Machine CSV Prediction", "Analytics Dashboard"])

# Prediction Function for CO₂ Emissions
def predict_co2(data):
    """Ensure column consistency before making predictions"""
    data = data.reindex(columns=co2_feature_names, fill_value=0)
    prediction = co2_model.predict(data)
    return prediction

# Prediction Function for Machine Failures
def predict_failure(data):
    """Ensure column consistency before making predictions"""
    data = data[failure_feature_names]
    prediction = failure_model.predict(data)
    return prediction

# 🌟 **Machine Failure Prediction**
if nav_option == "Machine Failure Prediction":
    st.subheader("🔹 Enter Machine Sensor Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.number_input("Temperature (°C)", min_value=50, max_value=120, step=1)
    with col2:
        vibration = st.number_input("Vibration Level", min_value=0.5, max_value=5.0, step=0.1)
    with col3:
        working_hours = st.number_input("Working Hours", min_value=1000, max_value=10000, step=100)

    if st.button("🔮 Predict Machine Failure"):
        input_data = pd.DataFrame([[temperature, vibration, working_hours]], columns=failure_feature_names)
        prediction = predict_failure(input_data)
        result = "⚠️ High Failure Risk!" if prediction[0] == 1 else "✅ Machine is Stable"
        st.success(f"Prediction: {result}")

# 🌟 **CO₂ Emission Prediction**
elif nav_option == "CO₂ Prediction":
    st.subheader("🔹 Enter Processing Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        tons_ore = st.number_input("Tons of Ore Processed", min_value=500, max_value=10000, step=100)
    with col2:
        energy_used = st.number_input("Energy Used (MWh)", min_value=100, max_value=5000, step=50)
    with col3:
        processing_method = st.selectbox("Processing Method", ["Method_A", "Method_B", "Method_C"])

    if st.button("🔮 Predict CO₂ Emissions"):
        input_data = pd.DataFrame([[tons_ore, energy_used, processing_method]], columns=co2_feature_names)
        prediction = predict_co2(input_data)
        st.success(f"🔥 Estimated CO₂ Emissions: {prediction[0]:.2f} tons")

# 🌟 **Batch Prediction**
elif nav_option == "Batch Prediction":
    st.subheader("📂 Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("📥 Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.dataframe(df.head())

        df = df.reindex(columns=co2_feature_names, fill_value=0)
        predictions = predict_co2(df)
        df["Predicted_CO2_Emissions"] = predictions

        st.subheader("📊 Prediction Results")
        st.dataframe(df)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv_data, "batch_predictions.csv", "text/csv")

# 🌟 **Machine CSV Prediction**
elif nav_option == "Machine CSV Prediction":
    st.subheader("🤖 Upload Machine Sensor CSV")

    machine_file = st.file_uploader("📥 Upload Machine Data", type=["csv"])

    if machine_file:
        df = pd.read_csv(machine_file)
        st.write("📄 Machine Data Preview:")
        st.dataframe(df.head())

        df = df[failure_feature_names]
        predictions = predict_failure(df)
        df["Failure_Prediction"] = predictions

        st.subheader("📊 Machine Failure Prediction Results")
        st.dataframe(df)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Machine Predictions", csv_data, "machine_failure_predictions.csv", "text/csv")

# 🌟 **Analytics Dashboard**
elif nav_option == "Analytics Dashboard":
    st.subheader("📈 Real-time Analytics on CO₂ Emissions & Machine Failures")

    df = pd.read_csv("random_data.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚡ CO₂ Emissions vs Ore Processed")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df["Tons_Ore_Processed"], y=df["CO2_Emissions_Tons"], hue=df["Processing_Method"], palette="coolwarm", ax=ax)
        plt.style.use("dark_background")
        st.pyplot(fig)

    with col2:
        st.markdown("### 🔥 CO₂ Emissions vs Energy Used")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df["Energy_Used_MWh"], y=df["CO2_Emissions_Tons"], hue=df["Processing_Method"], palette="viridis", ax=ax)
        plt.style.use("dark_background")
        st.pyplot(fig)

    st.markdown("### 📊 Model Performance Metrics")
    st.markdown("""
    - **R² Score:** 0.89
    - **Mean Absolute Error (MAE):** 25.3 tons
    - **Mean Squared Error (MSE):** 780.4 tons²
    """)

st.sidebar.markdown("💡 **Tip:** Use batch prediction for bulk data & analytics for insights.")
