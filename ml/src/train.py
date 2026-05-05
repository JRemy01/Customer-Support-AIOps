# Training a support ticket classifier with MLflow tracking
# trigger: retraining/ new credentials

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import  Model
from azure.ai.ml.constants import AssetTypes

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    data = {
        "text": [
            "System is completely down, no one can access the platform",
            "Payment processing is failing for all customers",
            "Cannot login at all, getting error 500",
            "Database crashed and we are losing data",
            "Email notifications are not being sent",
            "Dashboard is loading slowly but still works",
            "Export to PDF feature is not working properly",
            "Search results are sometimes incorrect",
            "How do I change my account password?",
            "Can you add a dark mode to the interface?",
            "I would like to request a new reporting feature",
            "Where can I find the user documentation?",
            "Our entire API is down and clients cannot connect",
            "Critical security breach detected in our system",
            "All users have been logged out and cannot get back in",
            "Production server is throwing errors for every request",
            "We cannot process any orders, revenue is stopped",
            "The mobile app crashes immediately on launch",
            "Reports are generating incorrect numbers",
            "File uploads are failing for most users",
            "Two-factor authentication is not sending codes",
            "The admin panel is inaccessible for our team",
            "Can you add CSV export to the reports page?",
            "It would be nice to have keyboard shortcuts",
            "Where can I find my billing history?",
            "How do I invite a new team member?",
            "Is there a way to customize the dashboard layout?",
            "The chart colors could be more accessible",
            "I have a question about my subscription plan",
            "Can you add support for multiple languages?",
            "The tooltip text is a bit confusing on the settings page",
            "I'd like to suggest an improvement to the onboarding flow",
        ],
        "priority": [
            "urgent",
            "urgent",
            "urgent",
            "urgent",
            "normal",
            "normal",
            "normal",
            "normal",
            "low",
            "low",
            "low",
            "low",
            "urgent",
            "urgent",
            "urgent",
            "urgent",
            "urgent",
            "normal",
            "normal",
            "normal",
            "normal",
            "normal",
            "low",
            "low",
            "low",
            "low",
            "low",
            "low",
            "low",
            "low",
            "low",
            "low",
        ]
    }

    df = pd.DataFrame(data)

    X = df['text']
    y  = df['priority']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=42, shuffle=True)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('rf', RandomForestClassifier(n_estimators= 100, random_state=42))
    ])


    # name of the experiment
    mlflow.set_experiment("support-ticket-classifier")

    with mlflow.start_run():
        # parameters
        mlflow.log_param('number of estimators', 100)
        mlflow.log_param('test size', 0.2)
        mlflow.log_param('random state', 42)

        # train the pipeline
        pipeline.fit(X_train, y_train)

        # predict
        y_pred = pipeline.predict(X_test)

        # log metric
        mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
        mlflow.log_metric('f1_score', f1_score(y_test,y_pred, average='weighted'))

        # log the model
        mlflow.sklearn.log_model(pipeline, "model")


    # Azure ML model registration
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential = credential,
        subscription_id = "7e4bf314-42fd-483c-9c78-37836c63124d",
        resource_group_name = "mlops-project-rg",
        workspace_name = "mlops-project-dev-ws" 
    )

    # Register the model
    model = Model(
        path = "./mlruns",
        type = AssetTypes.CUSTOM_MODEL,
        name = "support-ticket-classifier",
        description = "Classifies support tickets into urgent, normal, low"
    )

    ml_client.models.create_or_update(model)
    print("Model registered successfully!")

if __name__ == "__main__":
    main()