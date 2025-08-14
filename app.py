# app.py
import streamlit as st                # Import Streamlit for building web apps
import numpy as np                     # Import NumPy for numerical operations
import tensorflow as tf                # Import TensorFlow for deep learning
from tensorflow.keras.models import load_model  # Import load_model to load trained Keras models
import joblib                          # Import joblib to load saved scalers
import matplotlib.pyplot as plt        # Import matplotlib for plotting

# -----------------------
# Load model and scalers
# -----------------------
model = load_model("lstm_sensor_model.keras")   # Load trained LSTM model
scaler_x = joblib.load("scaler_x.save")        # Load feature scaler (StandardScaler for X)
scaler_y = joblib.load("scaler_y.save")        # Load target scaler (StandardScaler for y)

SEQ_LENGTH = 40  # Number of time steps used in RNN sequences

# -----------------------
# Streamlit app UI
# -----------------------
st.set_page_config(page_title="Sensor Performance Predictor", layout="wide")  # Set page title and layout
st.title("🚀 Sensor Performance Prediction")  # Main title of the app
st.markdown(
    """
    Enter the last 40 sensor readings to predict the **next `performance_index`**.
    You need sequences for:
    - Temperature
    - Vibration
    - Pressure
    """
)  # Description of app usage

# -----------------------
# Collect user input for sequences
# -----------------------
def user_input_sequence(seq_length=SEQ_LENGTH):
    # Create three columns for user input
    col1, col2, col3 = st.columns(3)

    # Temperature input
    with col1:
        temperature_seq = st.text_area(
            f"Temperature sequence ({seq_length} values, comma-separated)"
        )

    # Vibration input
    with col2:
        vibration_seq = st.text_area(
            f"Vibration sequence ({seq_length} values, comma-separated)"
        )

    # Pressure input
    with col3:
        pressure_seq = st.text_area(
            f"Pressure sequence ({seq_length} values, comma-separated)"
        )

    try:
        # Convert string inputs into NumPy arrays of floats
        temp_vals = np.array([float(x) for x in temperature_seq.split(",")])
        vib_vals = np.array([float(x) for x in vibration_seq.split(",")])
        pres_vals = np.array([float(x) for x in pressure_seq.split(",")])

        # Check if the length of each sequence matches SEQ_LENGTH
        if len(temp_vals) != seq_length or len(vib_vals) != seq_length or len(pres_vals) != seq_length:
            st.error(f"⚠️ Each sequence must have exactly {seq_length} values.")  # Show error if length mismatch
            return None

        # Stack sequences together and reshape for RNN input (1 sample, SEQ_LENGTH timesteps, 3 features)
        return np.stack([temp_vals, vib_vals, pres_vals], axis=1).reshape(1, seq_length, 3)
    except:
        st.error("⚠️ Invalid input. Please enter numeric values separated by commas.")  # Error for invalid input
        return None

# Get user input sequences
X_input = user_input_sequence()

# -----------------------
# Make prediction
# -----------------------
if X_input is not None and st.button("Predict Performance Index"):  # Run prediction when button clicked
    with st.spinner("Predicting... ⏳"):  # Show spinner while predicting
        # Scale input sequences using saved scaler
        X_scaled = scaler_x.transform(X_input.reshape(-1, 3)).reshape(1, SEQ_LENGTH, 3)

        # Make prediction using the trained LSTM model
        y_pred_scaled = model.predict(X_scaled)

        # Inverse transform to get prediction in original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Display prediction with animation
    st.balloons()  # Balloon animation
    st.success(f"🎯 Predicted Performance Index: **{y_pred[0][0]:.4f}**")  # Show predicted value

    # -----------------------
    # Plot input sequences for visualization
    # -----------------------
    temp_seq = X_input[0,:,0]  # Extract temperature sequence
    vib_seq = X_input[0,:,1]   # Extract vibration sequence
    pres_seq = X_input[0,:,2]  # Extract pressure sequence

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(range(1, SEQ_LENGTH+1), temp_seq, label='Temperature', marker='o')  # Plot temperature
    ax.plot(range(1, SEQ_LENGTH+1), vib_seq, label='Vibration', marker='x')     # Plot vibration
    ax.plot(range(1, SEQ_LENGTH+1), pres_seq, label='Pressure', marker='s')    # Plot pressure
    ax.set_title("📊 Sensor Input Sequences")  # Add title
    ax.set_xlabel("Time Step")                 # X-axis label
    ax.set_ylabel("Sensor Values")            # Y-axis label
    ax.legend()                               # Show legend
    st.pyplot(fig)                             # Display figure in Streamlit
