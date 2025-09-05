============================================================
                  🚀 SENSOR PERFORMANCE PREDICTION
============================================================

Description:
-------------
Predict the next performance index of a sensor system using
the last sequential readings from three sensors:

    - Temperature
    - Vibration
    - Pressure

Supported Models:
-----------------
    1. RNN (Recurrent Neural Network) - requires 20 sequential readings
    2. LSTM (Long Short-Term Memory Network) - requires 40 sequential readings

------------------------------------------------------------
HOW IT WORKS
------------------------------------------------------------

1. Input Sequences:
   - Provide the last readings for each sensor (comma-separated)
   - Sequence length depends on the selected model:
        LSTM: 40 values per sensor
        RNN: 20 values per sensor
   - Example:
        Temperature: 45.88, 44.86, 45.04, ... (20 or 40 values)
        Vibration:   0.31, 0.32, 0.30, ... (20 or 40 values)
        Pressure:    101.15, 101.08, 101.18, ... (20 or 40 values)

2. Model Selection:
   - Choose between RNN or LSTM
   - The selected model predicts the next `performance_index` based on the input sequences

3. Output:
   - Predicted `performance_index`
   - Performance label:
        ⚠️ Need Maintenance   → Poor performance
        ℹ️ Normal              → Acceptable performance
        ✅ Excellent           → Good performance
   - Performance score (0–100 scale) visualized as a progress bar

------------------------------------------------------------
TECH STACK
------------------------------------------------------------
- Python 3.10+
- TensorFlow / Keras – RNN & LSTM model training & inference
- NumPy – Numerical operations and array handling
- Scikit-learn – Preprocessing & scaling (input/output)
- Streamlit – Interactive web app for predictions
- Matplotlib – Plotting sensor sequences

------------------------------------------------------------
INSTALLATION
------------------------------------------------------------
1. Clone the repository:
       git clone https://github.com/your-username/Sensor-Performance-Prediction.git
       cd Sensor-Performance-Prediction

2. Create and activate environment:
       python -m venv env
       source env/bin/activate   # Linux/Mac
       env\Scripts\activate     # Windows

3. Install dependencies:
       pip install -r requirements.txt

------------------------------------------------------------
USAGE
------------------------------------------------------------
Run the Streamlit app:
       streamlit run app.py

Open in your browser:
       http://localhost:8501

Input Example:
--------------
Temperature: 45.88, 44.86, 45.04, 44.15, 44.67, ... (20 or 40 values)
Vibration:   0.31, 0.32, 0.30, 0.30, 0.34, ... (20 or 40 values)
Pressure:    101.15, 101.08, 101.18, 101.26, 101.48, ... (20 or 40 values)

Output:
--------
Predicted `performance_index`, performance label, and score

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------
Sensor-Performance-Prediction/
 ┣ app.py                # Streamlit application
 ┣ lstm_sensor_model.keras  # Pre-trained LSTM model
 ┣ saved_rnn_model.keras     # Pre-trained RNN model
 ┣ scaler_x.save          # Input scaler
 ┣ scaler_y.save          # Output scaler
 ┣ requirements.txt       # Python dependencies
 ┣ README.txt             # Documentation
 
------------------------------------------------------------
FUTURE IMPROVEMENTS
------------------------------------------------------------
- Add support for more advanced models (GRU, Transformer)
- Deploy on Streamlit Cloud / Heroku / Docker
- Enhance visualization of input sequences & predictions
- Add automated data validation for sequence length and format

------------------------------------------------------------
CONTRIBUTION
------------------------------------------------------------
Contributions are welcome! Please fork the repo and create a pull request.

------------------------------------------------------------
LICENSE
------------------------------------------------------------
This project is licensed under the MIT License.
============================================================
