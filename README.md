# Machine Learning model for GFFET simulation

This project provides a **Graphene Field-Effect Transistor (GFET)** resistance simulator using machine learning. The simulator allows users to upload data, train a machine learning model, and simulate the resistance of GFETs under various device parameters. The system uses **XGBoost** for training the model and provides a **Streamlit** web interface for interactive use.

## Features

- **Upload Training Data**: Upload an Excel file with `Vgs (V)` and `Resistance (Ohm)` data for training.
- **Training Model**: Train an XGBoost-based model using the uploaded data along with device parameters.
- **Simulate Resistance**: Simulate and plot resistance vs. gate-source voltage (`Vgs`) for the trained model with user-specified device parameters.
- **Interactive Interface**: Modify device parameters such as channel length, width, oxide thickness, carrier mobility, etc., using a Streamlit interface.
- **Download Simulated Data**: Download the simulated resistance data as a CSV file for further analysis.

## Requirements

To run this project, you need to have the following Python packages installed:

- **Streamlit**: For building the web interface.
- **XGBoost**: For machine learning model training.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For preprocessing and model evaluation.
- **Matplotlib**: For plotting the results.
- **Joblib**: For saving and loading the trained model.

### Installation

Clone the repository and install the required dependencies using the following commands:

```bash
git clone https://github.com/your-username/gfet-resistance-simulator.git
cd gfet-resistance-simulator
pip install -r requirements.txt
streamlit run app.py
