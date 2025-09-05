# -----------------------
# Import required libraries
# -----------------------
import streamlit as st                          # Streamlit: For creating the interactive web app UI
import numpy as np                              # NumPy: For numerical operations and arrays
import tensorflow as tf                         # TensorFlow: For running deep learning models
from tensorflow.keras.models import load_model  # Keras helper: To load saved neural network models
import joblib                                   # Joblib: For loading serialized scalers
import matplotlib.pyplot as plt                 # Matplotlib: For plotting sensor sequences

# -----------------------
# Streamlit app UI configuration
# -----------------------
st.set_page_config(page_title="Sensor Performance Predictor", layout="wide")  # Set page title and layout
st.title("🚀 Sensor Performance Prediction")  # Display main page title

# -----------------------
# Model selection dropdown
# -----------------------
model_display_names = ["LSTM Model", "RNN Model"]  # Options shown to the user
model_file_mapping = {                             # Map display names to actual model filenames
    "LSTM Model": "lstm_sensor_model.keras",
    "RNN Model": "saved_rnn_model.keras"
}
model_choice_display = st.selectbox(               # Dropdown for selecting model
    "Select Model",
    options=model_display_names,
    index=0
)
model_choice = model_file_mapping[model_choice_display]  # Map selected display name to actual file

# -----------------------
# Dynamic sequence length based on model
# -----------------------
SEQ_LENGTH = 40 if model_choice_display == "LSTM Model" else 20  # LSTM expects 40 steps, RNN 20 steps

# Instruction for users
st.markdown(
    f"""
    Enter the last **{SEQ_LENGTH} sensor readings** to predict the **next `performance_index`**.  
    You need sequences for:
    - Temperature  
    - Vibration  
    - Pressure  
    """
)

# -----------------------
# Load the selected model and scalers
# -----------------------
model = load_model(model_choice)             # Load the neural network model
scaler_x = joblib.load("scaler_x.save")      # Load input feature scaler
scaler_y = joblib.load("scaler_y.save")      # Load output target scaler

# -----------------------
# Function to collect user input sequences
# -----------------------
def user_input_sequence(seq_length):
    """
    Collects sensor input sequences from the user.
    Returns a numpy array of shape (1, seq_length, 3) or None if invalid.
    """
    col1, col2, col3 = st.columns(3)  # Create 3 input columns

    # Column 1: Temperature
    with col1:
        temperature_seq = st.text_area(f"Temperature sequence ({seq_length} values, comma-separated)")

    # Column 2: Vibration
    with col2:
        vibration_seq = st.text_area(f"Vibration sequence ({seq_length} values, comma-separated)")

    # Column 3: Pressure
    with col3:
        pressure_seq = st.text_area(f"Pressure sequence ({seq_length} values, comma-separated)")

    try:
        # Convert user input to float arrays
        temp_vals = np.array([float(x) for x in temperature_seq.split(",")])
        vib_vals = np.array([float(x) for x in vibration_seq.split(",")])
        pres_vals = np.array([float(x) for x in pressure_seq.split(",")])

        # Validate sequence lengths
        if len(temp_vals) != seq_length or len(vib_vals) != seq_length or len(pres_vals) != seq_length:
            st.error(f"⚠️ Each sequence must have exactly {seq_length} values.")
            return None

        # Stack sequences to shape (1, seq_length, 3)
        return np.stack([temp_vals, vib_vals, pres_vals], axis=1).reshape(1, seq_length, 3)

    except:
        st.error("⚠️ Invalid input. Please enter numeric values separated by commas.")
        return None

# -----------------------
# Get user input
# -----------------------
X_input = user_input_sequence(SEQ_LENGTH)  # Call function to read user input

# -----------------------
# Prediction logic
# -----------------------
if X_input is not None and st.button("Predict Performance Index"):  # Execute on button click
    with st.spinner("Predicting... ⏳"):  # Show loading spinner
        # Scale input using pre-fitted scaler
        X_scaled = scaler_x.transform(X_input.reshape(-1, 3)).reshape(1, SEQ_LENGTH, 3)

        # Model prediction
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        predicted_value = y_pred[0][0]  # Extract scalar prediction

        # -----------------------
        # Performance label logic
        # -----------------------
        if predicted_value < 0:
            performance_label = "⚠️ Need Maintenance"
            st.error(f"🎯 Predicted Performance Index: **{predicted_value:.4f}** → {performance_label}")
        elif 0 <= predicted_value <= 1:
            performance_label = "ℹ️ Normal"
            st.info(f"🎯 Predicted Performance Index: **{predicted_value:.4f}** → {performance_label}")
        else:
            performance_label = "✅ Excellent"
            st.success(f"🎯 Predicted Performance Index: **{predicted_value:.4f}** → {performance_label}")

        # -----------------------
        # Performance score visualization
        # -----------------------
        performance_score = np.clip((predicted_value + 1) * 50, 0, 100)  # Map prediction to 0-100 scale
        st.subheader("📊 Performance Score")
        st.progress(int(performance_score))  # Show progress bar
        st.write(f"**Performance Score:** {performance_score:.2f}")

        # -----------------------
        # Plot sensor sequences
        # -----------------------
        temp_seq = X_input[0, :, 0]  # Extract temperature sequence
        vib_seq = X_input[0, :, 1]   # Extract vibration sequence
        pres_seq = X_input[0, :, 2]  # Extract pressure sequence

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, SEQ_LENGTH + 1), temp_seq, label='Temperature', marker='o', linestyle='-', linewidth=2)
        ax.plot(range(1, SEQ_LENGTH + 1), vib_seq, label='Vibration', marker='x', linestyle='--', linewidth=2)
        ax.plot(range(1, SEQ_LENGTH + 1), pres_seq, label='Pressure', marker='s', linestyle='-.', linewidth=2)
        ax.set_title("Sensor Input Sequences", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Step", fontsize=12, fontweight='bold')
        ax.set_ylabel("Sensor Values", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)  # Render plot in Streamlit
# -----------------------
# End of app.py