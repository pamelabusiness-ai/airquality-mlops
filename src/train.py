# train.py
# Model Training and Testing Stage
# This script trains a Linear Regression model using the
# preprocessed Air Quality data and saves the model.
# Reference: Scikit-learn documentation
# https://scikit-learn.org/stable/modules/linear_model.html

import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_model():
    # Load preprocessed data
    df = pd.read_csv("data/processed.csv")

    # Define features and target
    X = df[["T"]].values
    y = df["CO(GT)"].values

    # Split data into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start MLflow experiment tracking
    mlflow.set_experiment("airquality_experiment")

    with mlflow.start_run():
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Log parameters and metrics to MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Save model
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")
        mlflow.sklearn.log_model(model, "model")

        print(f"Training complete. RMSE: {rmse:.4f} | R2: {r2:.4f}")
        print("Model saved to model/model.pkl")

    return model


if __name__ == "__main__":
    train_model()
