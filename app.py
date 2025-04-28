import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from train_model import train_model
from utils import build_training_features, build_simulation_features
from plotting import plot_resistance_vs_vgs


st.set_page_config(page_title="GFET Resistance Simulator", layout="wide")

st.title("Graphene FET Resistance Simulator")

# Session state to store model and scaler
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Sidebar - Upload Training Data
st.sidebar.header("Upload Training Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel File (.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # Check if only required columns are present
    expected_columns = ["Vgs (V)", "Resistance (Ohm)"]
    if list(df.columns) != expected_columns:
        st.error(f"Uploaded file must have exactly two columns: {expected_columns}")
    else:
        st.write("### Uploaded Data Preview", df.head())

        st.sidebar.subheader("Device Parameters for Uploaded Data")
        
        # Sliders for Device Parameters
        L_um = st.sidebar.number_input("Channel Length (μm)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
        W_um = st.sidebar.number_input("Channel Width (μm)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
        tox_nm = st.sidebar.number_input("Oxide Thickness (nm)", min_value=1.0, max_value=500.0, value=300.0, step=1.0)
        mobility_cm2Vs = st.sidebar.number_input("Carrier Mobility (cm²/Vs)", min_value=10000.0, max_value=100000.0, value=15000.0, step=1000.0)
        n0 = st.sidebar.number_input("Intrinsic Carrier Density (1e12 /cm²)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        v_dirac = st.sidebar.number_input("Dirac Point Voltage (V)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)

        if st.sidebar.button("Train Model"):
            # Build full training data with device parameters
            df_features = build_training_features(df["Vgs (V)"], L_um, W_um, tox_nm, mobility_cm2Vs, n0, v_dirac)
            df_features["Resistance (Ohm)"] = df["Resistance (Ohm)"].values

            # Train the model with the device parameters
            st.session_state.model, st.session_state.scaler = train_model(df_features, L_um, W_um, tox_nm, mobility_cm2Vs, n0, v_dirac)
            st.success("Model trained successfully! Ready for simulation.")

# Simulation Section
if st.session_state.model is not None:
    st.sidebar.header("Simulation Parameters")

    # Device Parameters for simulation
    L_um_sim = st.sidebar.number_input("Sim Channel Length (μm)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
    W_um_sim = st.sidebar.number_input("Sim Channel Width (μm)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    tox_nm_sim = st.sidebar.number_input("Sim Oxide Thickness (nm)", min_value=1.0, max_value=500.0, value=300.0, step=1.0)
    mobility_cm2Vs_sim = st.sidebar.number_input("Sim Carrier Mobility (cm²/Vs)", min_value=10000.0, max_value=100000.0, value=15000.0, step=1000.0)
    n0_sim = st.sidebar.number_input("Sim Intrinsic Carrier Density (1e12 /cm²)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    v_dirac_sim = st.sidebar.number_input("Sim Dirac Point Voltage (V)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)

    # Vgs Range
    st.sidebar.header("Vgs Sweep Settings")
    Vgs_min = st.sidebar.number_input("Vgs Min (V)", min_value=-50.0, max_value=50.0, value=-30.0, step=1.0)
    Vgs_max = st.sidebar.number_input("Vgs Max (V)", min_value=-50.0, max_value=50.0, value=30.0, step=1.0)
    num_points = st.sidebar.slider("Number of Vgs Points", min_value=10, max_value=500, value=100)

    if st.sidebar.button("Simulate and Plot"):
        if Vgs_min >= Vgs_max:
            st.error("Vgs Min must be less than Vgs Max.")
        else:
            # Generate Vgs values for simulation
            Vgs_array = np.linspace(Vgs_min, Vgs_max, num_points)
            
            # Build simulation features with device parameters
            X_sim = build_simulation_features(Vgs_array, L_um_sim, W_um_sim, tox_nm_sim, mobility_cm2Vs_sim, n0_sim, v_dirac_sim)
            X_sim_scaled = st.session_state.scaler.transform(X_sim)

            # Predict resistance using the trained model
            predicted_R = st.session_state.model.predict(X_sim_scaled)

            # Plot results
            plot_resistance_vs_vgs(Vgs_array, predicted_R)

            # Prepare simulated data for download
            simulated_data = pd.DataFrame({"Vgs (V)": Vgs_array, "Predicted Resistance (Ohm)": predicted_R})
            st.download_button(
                label="Download Simulated Data as CSV",
                data=simulated_data.to_csv(index=False),
                file_name="simulated_data.csv",
                mime="text/csv"
            )
else:
    st.info("Please upload training data and train the model first.")
