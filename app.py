# Import required libraries
# -----------------------
import streamlit as st                # type: ignore # Streamlit: For creating the interactive web app UI
import numpy as np                     # NumPy: For handling numerical operations and arrays
import tensorflow as tf                # TensorFlow: For loading and running deep learning models
from tensorflow.keras.models import load_model  # type: ignore # Keras helper: Loads pre-trained models from files
import joblib                          # Joblib: Loads serialized objects (e.g., scalers) from disk
import matplotlib.pyplot as plt        # Matplotlib: For creating visual plots

# -----------------------
# Constants
# -----------------------
SEQ_LENGTH = 40  # Expected number of time steps (length) in each input sequence for the model

# -----------------------
# Streamlit app UI configuration
# -----------------------
st.set_page_config(page_title="Sensor Performance Predictor", layout="wide")  # Set app title and layout
st.title("🚀 Sensor Performance Prediction")  # Display main page title
st.markdown(  # Provide user instructions in markdown format
    """
    Enter the last 40 sensor readings to predict the **next `performance_index`**.  
    You need sequences for:
    - Temperature  
    - Vibration  
    - Pressure  
    """
)

# -----------------------
# Model selection dropdown with display names
# -----------------------
model_display_names = ["LSTM Model", "RNN Model"]  # Names shown in dropdown
model_file_mapping = {  # Map display names to actual model filenames
    "LSTM Model": "lstm_sensor_model.keras",
    "RNN Model": "saved_rnn_model.keras"
}

# Create a dropdown list for selecting model
model_choice_display = st.selectbox(
    "Select Model",                  # Label for dropdown
    options=model_display_names,     # Options shown to user
    index=0                          # Default option is the first (LSTM Model)
)

# Map user’s selection to the actual model file path
model_choice = model_file_mapping[model_choice_display]

# -----------------------
# Load the selected model
# -----------------------
model = load_model(model_choice)  # Load the neural network model from file

# Load the scalers used during training (to normalize/denormalize data)
scaler_x = joblib.load("scaler_x.save")  # Input features scaler
scaler_y = joblib.load("scaler_y.save")  # Output target scaler

# -----------------------
# Function to collect user input sequences
# -----------------------
def user_input_sequence(seq_length=SEQ_LENGTH):
    # Create three equally wide columns for user inputs
    col1, col2, col3 = st.columns(3)

    # Column 1: Temperature sequence input
    with col1:
        temperature_seq = st.text_area(
            f"Temperature sequence ({seq_length} values, comma-separated)"  # Input hint
        )

    # Column 2: Vibration sequence input
    with col2:
        vibration_seq = st.text_area(
            f"Vibration sequence ({seq_length} values, comma-separated)"
        )

    # Column 3: Pressure sequence input
    with col3:
        pressure_seq = st.text_area(
            f"Pressure sequence ({seq_length} values, comma-separated)"
        )

    try:
        # Convert the comma-separated input strings into NumPy arrays of floats
        temp_vals = np.array([float(x) for x in temperature_seq.split(",")])
        vib_vals = np.array([float(x) for x in vibration_seq.split(",")])
        pres_vals = np.array([float(x) for x in pressure_seq.split(",")])

        # Ensure each sequence has exactly the required length
        if len(temp_vals) != seq_length or len(vib_vals) != seq_length or len(pres_vals) != seq_length:
            st.error(f"⚠️ Each sequence must have exactly {seq_length} values.")  # Show error message
            return None

        # Stack sequences together into shape (1, time_steps, features)
        return np.stack([temp_vals, vib_vals, pres_vals], axis=1).reshape(1, seq_length, 3)

    except:
        # Show error if conversion to float fails
        st.error("⚠️ Invalid input. Please enter numeric values separated by commas.")
        return None

# -----------------------
# Get user input data
# -----------------------
X_input = user_input_sequence()  # Call the function to collect and process user data

# -----------------------
# Prediction logic
# -----------------------
if X_input is not None and st.button("Predict Performance Index"):  # Run only if data entered and button clicked
    with st.spinner("Predicting... ⏳"):  # Show a loading spinner while model predicts
        # Normalize input data using pre-fitted scaler
        X_scaled = scaler_x.transform(X_input.reshape(-1, 3)).reshape(1, SEQ_LENGTH, 3)

        # Make prediction using the loaded model
        y_pred_scaled = model.predict(X_scaled)
        # Convert scaled prediction back to original range
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        predicted_value = y_pred[0][0]  # Extract scalar prediction value

        # -----------------------
        # Performance label logic
        # -----------------------
        if predicted_value < 0:
            performance_label = "⚠️ Need Maintenance"  # Poor performance
            st.error(f"🎯 Predicted Performance Index: **{predicted_value:.4f}** → {performance_label}")
        elif 0 <= predicted_value <= 1:
            performance_label = "ℹ️ Normal"  # Acceptable performance
            st.info(f"🎯 Predicted Performance Index: **{predicted_value:.4f}** → {performance_label}")
        else:
            performance_label = "✅ Excellent"  # Good performance
            st.success(f"🎯 Predicted Performance Index: **{predicted_value:.4f}** → {performance_label}")

        # -----------------------
        # Performance score (0-100 scale)
        # -----------------------
        performance_score = np.clip((predicted_value + 1) * 50, 0, 100)  # Map range and cap values
        st.subheader("📊 Performance Score")  # Display score title
        st.progress(int(performance_score))   # Show progress bar
        st.write(f"**Performance Score:** {performance_score:.2f}")  # Show numeric score

    # -----------------------
    # Enhanced sequence plot
    # -----------------------
    temp_seq = X_input[0, :, 0]  # Get temperature values from input
    vib_seq = X_input[0, :, 1]   # Get vibration values from input
    pres_seq = X_input[0, :, 2]  # Get pressure values from input
    # Create the plot with custom styles
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, SEQ_LENGTH+1), temp_seq, label='Temperature', marker='o', 
            linestyle='-', linewidth=2)
    ax.plot(range(1, SEQ_LENGTH+1), vib_seq, label='Vibration', marker='x', 
            linestyle='--', linewidth=2)
    ax.plot(range(1, SEQ_LENGTH+1), pres_seq, label='Pressure', marker='s', 
            linestyle='-.', linewidth=2)
    # Add plot title and labels
    ax.set_title("Sensor Input Sequences", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time Step", fontsize=12, fontweight='bold')
    ax.set_ylabel("Sensor Values", fontsize=12, fontweight='bold')
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    # Render plot in Streamlit
    st.pyplot(fig)
#end of App 
