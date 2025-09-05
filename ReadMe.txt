===========================================================
EV Car Purchase Prediction App 🚗
===========================================================

Project Overview:
-----------------
This project is an EV Car Purchase Prediction App that predicts whether a customer is likely 
to purchase an SUV electric vehicle (EV) using the XGBoost machine learning algorithm. 
The model is trained on historical customer data to provide accurate predictions based 
on key customer features.

The app is deployed using Streamlit, allowing users to input customer information interactively 
and get real-time predictions.

Features:
---------
- Predicts if a customer will purchase an SUV EV using XGBoost.
- Interactive Streamlit interface for real-time predictions.
- Handles multiple customer input features including salary, credit score, and more.
- Includes preprocessing steps for clean and reliable predictions.
- Easily extendable to include additional customer features or data.

Customer Input Features:
------------------------
The app requires the following customer features to make a prediction:

1. Estimated Salary ($)
2. Credit Score
3. Internet Usage per Day (hours)
4. Age
5. Annual Expenses ($)
6. Number of Dependents
7. Loan Amount ($)
8. Employment Status (e.g., Employed, Unemployed)

Installation:
-------------
1. Clone the repository:
   git clone https://github.com/yitayew123/XGBOOST.git
   cd XGBOOST

2. Create a virtual environment (recommended):
   python -m venv XGBOOST_ENV
   # Activate environment:
   # Linux/Mac:
   source XGBOOST_ENV/bin/activate
   # Windows:
   XGBOOST_ENV\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

Usage:
------
1. Run the Streamlit app:
   streamlit run app.py

2. Open the local URL provided by Streamlit in your browser.

3. Enter the customer features listed above to predict whether the customer is likely 
   to purchase an SUV EV.

Project Structure:
------------------
XGBOOST/
│
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── README.txt             # Project documentation

Dependencies:
-------------
- Python >= 3.8
- XGBoost
- Streamlit
- pandas
- scikit-learn
- numpy

Contributing:
-------------
Contributions are welcome! To improve the model, add new features, or enhance the app, 
please open an issue or submit a pull request.

License:
--------
This project is open-source and available under the MIT License.

Author:
-------
Yitayew Solomon
Email: yitayewsolomon3@gmail.com
GitHub: https://github.com/yitayew123

===========================================================
