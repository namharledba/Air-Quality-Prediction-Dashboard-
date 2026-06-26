import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report

st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("Air Quality Monitoring Dashboard")

file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    df.replace(-200, np.nan, inplace=True)

    obj_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(include=np.number).columns

    for col in obj_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    def air_quality_category(row):
        score = 0
        if 'CO(GT)' in df.columns and row['CO(GT)'] > 2:
            score += 1
        if 'NO2(GT)' in df.columns and row['NO2(GT)'] > 80:
            score += 1
        if 'NOx(GT)' in df.columns and row['NOx(GT)'] > 150:
            score += 1
        if 'C6H6(GT)' in df.columns and row['C6H6(GT)'] > 10:
            score += 1
        if 'RH' in df.columns and row['RH'] > 60:
            score += 1

        if score <= 1:
            return "Good"
        elif score <= 3:
            return "Moderate"
        return "Poor"

    df["Air_Quality"] = df.apply(air_quality_category, axis=1)
    numeric_df = df.select_dtypes(include=np.number)

    section = st.sidebar.selectbox(
        "Choose Section",
        ["Dashboard","Dataset","Visualization","Correlation",
         "Regression","Classification","Manual Prediction","Forecast"]
    )

    if section == "Dashboard":
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        if "CO(GT)" in df.columns:
            c4.metric("Average CO", round(df["CO(GT)"].mean(),2))
        st.dataframe(df.head())

    elif section == "Dataset":
        st.dataframe(df)
        st.write(df.describe())

    elif section == "Visualization":
        feature = st.selectbox("Feature", numeric_df.columns)
        st.plotly_chart(px.histogram(df, x=feature), use_container_width=True)
        st.plotly_chart(px.line(df, y=feature), use_container_width=True)

    elif section == "Correlation":
        fig, ax = plt.subplots(figsize=(12,8))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif section == "Regression":
        if "CO(GT)" in numeric_df.columns:
            X = numeric_df.drop("CO(GT)", axis=1)
            y = numeric_df["CO(GT)"]

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            name = st.selectbox("Model",["Linear Regression","Decision Tree","Random Forest"])

            if name == "Linear Regression":
                model = LinearRegression()
            elif name == "Decision Tree":
                model = DecisionTreeRegressor()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train,y_train)
            pred = model.predict(X_test)

            c1,c2,c3 = st.columns(3)
            c1.metric("R2", round(r2_score(y_test,pred),3))
            c2.metric("MAE", round(mean_absolute_error(y_test,pred),3))
            c3.metric("RMSE", round(np.sqrt(mean_squared_error(y_test,pred)),3))

            st.plotly_chart(px.scatter(x=y_test,y=pred,labels={"x":"Actual","y":"Predicted"}))

    elif section == "Classification":
        X = numeric_df.copy()
        y = df["Air_Quality"]

        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        st.metric("Accuracy", round(accuracy_score(y_test,pred),3))
        st.text(classification_report(y_test,pred))

    elif section == "Manual Prediction":
        X = numeric_df.copy()
        y = df["Air_Quality"]

        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X,y)

        values = []
        cols = st.columns(3)

        for i, feature in enumerate(X.columns):
            values.append(
                cols[i % 3].number_input(feature, value=float(X[feature].mean()))
            )

        if st.button("Predict"):
            sample = pd.DataFrame([values], columns=X.columns)
            result = encoder.inverse_transform(model.predict(sample))[0]

            st.subheader(f"Air Quality: {result}")

            if result == "Good":
                st.success("Safe air quality. Outdoor activities are fine.")
            elif result == "Moderate":
                st.warning("Sensitive people should reduce outdoor activities.")
            else:
                st.error("Poor air quality. Stay indoors and wear a mask.")

    elif section == "Forecast":
        if "CO(GT)" in numeric_df.columns:
            days = st.slider("Forecast Days",1,30,7)
            current = numeric_df["CO(GT)"].iloc[-1]
            preds = []

            for _ in range(days):
                current += np.random.normal(0,0.2)
                preds.append(current)

            forecast_df = pd.DataFrame({"Day":range(1,days+1),"Predicted CO":preds})
            st.dataframe(forecast_df)
            st.plotly_chart(px.line(forecast_df,x="Day",y="Predicted CO"))

else:
    st.info("Upload a CSV file to start")
