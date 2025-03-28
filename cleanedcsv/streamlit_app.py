import streamlit as st
import pandas as pd
import joblib

# 1) Load your XGBoost model
model = joblib.load("xgboost_day_ahead.pkl")

st.title("Day-Ahead Electricity Price Predictor")

# 2) Define all columns your model expects (including 'hour')
MODEL_FEATURES = [
    "Austria MWh Original resolutions_Imported_balancing_services_202301010000_202503050000_Quarterhour",
    "Biomass MWh Original resolutions_Actual_generation_Quarterhour",
    "Hydropower MWh Original resolutions_Actual_generation_Quarterhour",
    "Wind offshore MWh Original resolutions_Actual_generation_Quarterhour",
    "Wind onshore MWh Original resolutions_Actual_generation_Quarterhour",
    "Photovoltaics MWh Original resolutions_Actual_generation_Quarterhour",
    "Other renewable MWh Original resolutions_Actual_generation_Quarterhour",
    "Nuclear MWh Original resolutions_Actual_generation_Quarterhour",
    "Lignite MWh Original resolutions_Actual_generation_Quarterhour",
    "Hard coal MWh Original resolutions_Actual_generation_Quarterhour",
    "Fossil gas MWh Original resolutions_Actual_generation_Quarterhour",
    "Hydro pumped storage MWh Original resolutions_Actual_generation_Quarterhour",
    "Other conventional MWh Original resolutions_Actual_generation_Quarterhour",
    "Total (grid load) MWh Original resolutions_Actual_consumption_Quarterhour",
    "Residual load MWh Original resolutions_Actual_consumption_Quarterhour",
    "Hydro pumped storage MWh Original resolutions_Actual_consumption_Quarterhour",
    "Austria MWh Original resolutions_Exported_balancing_services_202301010000_202503050000_Quarterhour",
    "Price €/MWh Original resolutions_Balancing_energy_Quarterhour_Month",
    "Net export MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Netherlands (export) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Netherlands (import) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Switzerland (export) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Switzerland (import) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Denmark (export) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Denmark (import) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Czech Republic (export) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Czech Republic (import) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Luxembourg (export) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Sweden (import) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Austria (export) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Austria (import) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "France (export) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "France (import) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Poland (export) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Poland (import) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Norway (import) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    "Belgium (import) MWh Original resolutions_Cross-border_physical_flows_Quarterhour",
    # Finally, your hour featur
]

# 3) Collect user inputs for the main features you want to manipulate
residual_load = st.slider("Residual Load (MW)", 0, 20000, 10000)
fossil_gas = st.slider("Fossil Gas Generation (MW)", 0, 15000, 5000)
lignite = st.slider("Lignite Generation (MW)", 0, 15000, 4000)

# 4) Build a small dictionary with the user inputs
#    (All other features in MODEL_FEATURES will be filled with 0 by default)
user_input = {
    "Residual load MWh Original resolutions_Actual_consumption_Quarterhour": residual_load,
    "Fossil gas MWh Original resolutions_Actual_generation_Quarterhour": fossil_gas,
    "Lignite MWh Original resolutions_Actual_generation_Quarterhour": lignite,
}

# 5) Create a DataFrame with just these inputs
input_df = pd.DataFrame([user_input])

# 6) Reindex to match the exact model feature columns, fill missing columns with 0
input_df = input_df.reindex(columns=MODEL_FEATURES, fill_value=0)

# 7) Predict using the aligned DataFrame
prediction = model.predict(input_df)[0]

st.subheader("🔮 Predicted Day-Ahead Price")
st.metric(label="Price (€ / MWh)", value=f"{prediction:.2f}")
