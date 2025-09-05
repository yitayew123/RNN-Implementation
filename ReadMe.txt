============================================================
                  🚀 SENSOR PERFORMANCE PREDICTION
============================================================

Description:
-------------
Predict the next performance index of a sensor system using
the last 40 sequential readings from:

    - Temperature
    - Vibration
    - Pressure

Supported Models:
-----------------
    1. RNN (Recurrent Neural Network)
    2. LSTM (Long Short-Term Memory Network)

------------------------------------------------------------
HOW IT WORKS
------------------------------------------------------------

1. Input Sequences:
   - Provide the last 40 values for each sensor (comma-separated)
   - Example:
        Temperature: 30.1, 30.5, 29.9, ... (40 values)
        Vibration: 0.02, 0.05, 0.03, ... (40 values)
        Pressure: 101.2, 101.5, 101.3, ... (40 values)

2. Model Selection:
   - Choose between RNN or LSTM
   - The selected model predicts the next performance_index

------------------------------------------------------------
TECH STACK
------------------------------------------------------------
- Python 3.10+
- TensorFlow / Keras – RNN & LSTM model training & inference
- NumPy & Pandas – Data handling
- Scikit-learn – Preprocessing & scaling
- Streamlit – Web app for predictions
- Matplotlib / Seaborn – Data visualization

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
Temperature: 30.1, 30.5, 29.9, ... (40 values)
Vibration:   0.02, 0.05, 0.03, ... (40 values)
Pressure:    101.2, 101.5, 101.3, ... (40 values)

Output:
--------
Predicted performance_index

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------
Sensor-Performance-Prediction/
 ┣ app.py               # Streamlit app
 ┣ model_rnn.h5         # Pre-trained RNN model
 ┣ model_lstm.h5        # Pre-trained LSTM model
 ┣ requirements.txt     # Dependencies
 ┣ README.txt           # Documentation
 ┗ data/                # Sample datasets

------------------------------------------------------------
FUTURE IMPROVEMENTS
------------------------------------------------------------
- Add support for more advanced models (GRU, Transformer)
- Deploy on Streamlit Cloud / Heroku / Docker
- Improve visualization of input sequences & predictions

------------------------------------------------------------
CONTRIBUTION
------------------------------------------------------------
Contributions are welcome! Please fork the repo and create a pull request.

------------------------------------------------------------
LICENSE
------------------------------------------------------------
This project is licensed under the MIT License.
============================================================
