# AIR QUALITY MONITORING AND PREDICTION SYSTEM

OVERVIEW
The Air Quality Monitoring and Prediction System is a machine learning and data analysis project designed to analyze environmental air quality data, predict pollution levels, classify air quality conditions, and provide health recommendations based on predicted air quality status.

made by :

1- Mahmoud Ashraf

2- Shahd 

3- Mahmoud sadek

4- Farah 

5- Abdelrahman Mohamed

FEATURES
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Interactive data visualization
- Correlation analysis using heatmaps
- CO concentration prediction using regression models
- Air quality classification
- Manual air quality prediction
- Health recommendations
- Future CO level forecasting
- Interactive Streamlit dashboard

DATASET
The project uses the Air Quality UCI dataset containing measurements from multiple air quality sensors.

DATASET ATTRIBUTES
- CO(GT)
- NMHC(GT)
- C6H6(GT)
- NOx(GT)
- NO2(GT)
- Temperature (T)
- Relative Humidity (RH)
- Absolute Humidity (AH)

MACHINE LEARNING MODELS

Regression Models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Classification Model:
- Random Forest Classifier

AIR QUALITY CATEGORIES
- Good
- Moderate
- Poor

TECHNOLOGIES USED
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Streamlit

PROJECT STRUCTURE
Air_QualityUCI.csv
app.py
DataAnalysis.ipynb
README.txt

INSTALLATION / How to run

1. Clone the repository:
https://github.com/namharledba/Air-Quality-Prediction-Dashboard-

2. Install dependencies:
pip install -r requirements.txt

RUNNING THE APPLICATION

streamlit run app.py 

or 

through the link :

DASHBOARD SECTIONS
- Dashboard
- Dataset
- Visualization
- Correlation
- Regression
- Classification
- Manual Prediction
- Forecast

HEALTH RECOMMENDATIONS

Good:
- Outdoor activities are safe.
- No special precautions are required.

Moderate:
- Sensitive individuals should reduce outdoor exposure.
- Consider wearing a mask.

Poor:
- Avoid outdoor activities.
- Wear a protective mask.
- Stay indoors whenever possible.


