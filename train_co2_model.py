import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("random_data.csv")

# Define Features & Target
X = df.drop(columns=["CO2_Emissions_Tons"])
y = df["CO2_Emissions_Tons"]

# Identify categorical & numerical columns
categorical_features = ["Processing_Method"]
numerical_features = ["Tons_Ore_Processed", "Energy_Used_MWh"]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Build Model Pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Model
model_pipeline.fit(X_train, y_train)

# Save Model and Encoded Feature Names
joblib.dump(model_pipeline, "models/co2_model.pkl")
joblib.dump(X_train.columns, "models/co2_feature_names.pkl")
print("Model training complete and saved!")
