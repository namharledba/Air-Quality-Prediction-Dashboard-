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

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    classification_report
)

# Page Setup


st.set_page_config(
    page_title="Air Quality Dashboard",
    layout="wide"
)


st.title("🌫️ Air Quality Analysis Dashboard")


# Upload Data

file = st.file_uploader(
    "Upload Air Quality Dataset",
    type=["csv"]
)


if file:

    df = pd.read_csv(file)


    st.subheader("Dataset Preview")

    st.dataframe(df.head())


    # Cleaning

    df.replace(-200, np.nan, inplace=True)

    for col in df.columns:

        if df[col].isnull().sum() > 0:

            if df[col].dtype == "object":
                df[col].fillna(
                    df[col].mode()[0],
                    inplace=True
                )

            else:
                df[col].fillna(
                    df[col].median(),
                    inplace=True
                )


    df.drop_duplicates(inplace=True)


    st.success("Data cleaned successfully")


    # Sidebar


    choice = st.sidebar.selectbox(
        "Choose Section",
        [
            "Statistics",
            "Visualization",
            "Correlation",
            "Regression",
            "Classification"
        ]
    )


    # Statistics


    if choice == "Statistics":


        st.subheader("Dataset Information")

        st.write(df.info())

        st.write(
            df.describe()
        )


    # Visualization


    elif choice == "Visualization":


        pollutant = st.selectbox(
            "Select pollutant",
            [
                "CO(GT)",
                "NO2(GT)",
                "NOx(GT)"
            ]
        )


        fig = px.histogram(
            df,
            x=pollutant,
            title=f"{pollutant} Distribution"
        )


        st.plotly_chart(fig)



        fig2 = px.line(
            df,
            y=pollutant,
            title=f"{pollutant} Trend"
        )


        st.plotly_chart(fig2)


    # Correlation


    elif choice == "Correlation":


        st.subheader(
            "Correlation Heatmap"
        )


        numeric = df.select_dtypes(
            include=np.number
        )


        fig, ax = plt.subplots(
            figsize=(10,6)
        )


        sns.heatmap(
            numeric.corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )


        st.pyplot(fig)


    # Regression


    elif choice == "Regression":


        st.subheader(
            "Predict CO Level"
        )


        target = "CO(GT)"


        X = df.select_dtypes(
            include=np.number
        ).drop(
            target,
            axis=1
        )


        y = df[target]


        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )


        scaler = StandardScaler()


        X_train = scaler.fit_transform(
            X_train
        )


        X_test = scaler.transform(
            X_test
        )



        model_name = st.selectbox(
            "Choose Model",
            [
                "Linear Regression",
                "Decision Tree",
                "Random Forest"
            ]
        )



        if model_name == "Linear Regression":

            model = LinearRegression()


        elif model_name == "Decision Tree":

            model = DecisionTreeRegressor()


        else:

            model = RandomForestRegressor(
                n_estimators=100
            )



        model.fit(
            X_train,
            y_train
        )


        pred = model.predict(
            X_test
        )



        st.metric(
            "R2 Score",
            round(
                r2_score(y_test,pred),
                3
            )
        )


        fig = px.scatter(
            x=y_test,
            y=pred,
            labels={
                "x":"Actual",
                "y":"Prediction"
            }
        )


        st.plotly_chart(fig)


    # Classification



    elif choice == "Classification":


        st.subheader(
            "Air Quality Classification"
        )


        if "Air Quality" in df.columns:


            X = df.drop(
                "Air Quality",
                axis=1
            )


            y = df["Air Quality"]



            le = LabelEncoder()

            y = le.fit_transform(y)



            X = X.select_dtypes(
                include=np.number
            )



            X_train,X_test,y_train,y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )


            model = RandomForestClassifier(
                n_estimators=100
            )


            model.fit(
                X_train,
                y_train
            )


            pred = model.predict(
                X_test
            )



            st.metric(
                "Accuracy",
                round(
                    accuracy_score(
                        y_test,
                        pred
                    ),
                    3
                )
            )


            st.text(
                classification_report(
                    y_test,
                    pred
                )
            )


        else:

            st.warning(
                "No Air Quality column found"
            )


else:

    st.info(
        "Upload CSV file to start"
    )