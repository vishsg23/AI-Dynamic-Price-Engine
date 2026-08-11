import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
import os

# ==========================================================
# Configure MLflow Tracking
# ==========================================================

if os.getenv("AIRFLOW_HOME"):
    # Running inside the Airflow Docker container.
    # "localhost" here would mean the container itself, not your
    # Windows machine — Docker Desktop's host.docker.internal is the
    # correct address to reach the MLflow server actually running on
    # your Windows host at port 5000.
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    print("Airflow detected — using MLflow server: http://host.docker.internal:5000")
else:
    # Local Windows MLflow server
    mlflow.set_tracking_uri("http://localhost:5000")
    print("Using MLflow server: http://localhost:5000")


def log_model_run(model, model_name, params, metrics,
                  feature_names=None, X_test=None, y_test=None):

    experiment_name = f"pricing_engine_{model_name}"

    try:
        mlflow.set_experiment(experiment_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        with mlflow.start_run(run_name=f"{model_name}_{timestamp}"):

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            if "xgb" in model_name.lower():
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

            try:
                mlflow.register_model(model_uri, model_name)
            except Exception as e:
                print(f"Model registration skipped: {e}")

            print(f"Run logged: {run_id}")
            return run_id

    except Exception as e:
        # If the MLflow server genuinely isn't reachable (for example,
        # you forgot to start it before triggering the Airflow DAG),
        # this logs a clear warning and lets training continue instead
        # of crashing the whole Airflow task over a tracking failure.
        print(f"⚠️ MLflow logging failed, continuing without it: {e}")
        return None