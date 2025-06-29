"""Run prediction on diabetes data."""
from pathlib import Path
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import shap
import joblib
import os
from ARISA_DSML.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    MODEL_NAME,
)
from ARISA_DSML.resolve import get_model_by_alias
import mlflow
from mlflow.client import MlflowClient
import json
import nannyml as nml
from ARISA_DSML.helpers import get_git_commit_hash
import ast


def plot_shap(model: CatBoostClassifier, df_plot: pd.DataFrame) -> None:
    """Plot model shapley overview plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_plot)
    shap.summary_plot(shap_values, df_plot, show=False)
    plt.savefig(FIGURES_DIR / "diabetes_shap_overall.png")


def predict(model: CatBoostClassifier, df_pred: pd.DataFrame, params: dict, probs=False) -> str | Path:
    """Run prediction."""
    feature_columns = params.pop("feature_columns", None)
    if feature_columns is None:
        raise ValueError("Missing 'feature_columns' in params. Cannot perform prediction.")

    expected_features = params.get("feature_columns", [])
    missing_cols = [col for col in expected_features if col not in df_pred.columns]
    if missing_cols:
        raise RuntimeError(f"The following expected feature columns are missing in test data: {missing_cols}")

    preds = model.predict(df_pred[feature_columns])
    if probs:
        df_pred["predicted_probability"] = [p[1] for p in model.predict_proba(df_pred[feature_columns])]

    plot_shap(model, df_pred[feature_columns])
    df_pred["prediction"] = preds
    preds_path = MODELS_DIR / "preds.csv"
    df_pred[["prediction", "predicted_probability"]].to_csv(preds_path, index=False)

    return preds_path


if __name__ == "__main__":
    df_test = pd.read_csv(PROCESSED_DATA_DIR / "diabetes.csv")

    client = MlflowClient(mlflow.get_tracking_uri())
    model_info = get_model_by_alias(client, alias="champion")
    logger.info(f"Loaded test data with columns: {df_test.columns.tolist()}")
    logger.info(f"Model info: {model_info}")

    if model_info is None:
        logger.info("No champion model, predicting using latest model")
        model_info = client.get_latest_versions(MODEL_NAME)[0]

    run = client.get_run(model_info.run_id)
    run_data_dict = run.data.to_dictionary()

    _, artifact_folder = os.path.split(model_info.source)
    model_uri = f"runs:/{model_info.run_id}/{artifact_folder}"
    loaded_model = mlflow.catboost.load_model(model_uri)

    client.download_artifacts(model_info.run_id, "udc.pkl", "models")
    client.download_artifacts(model_info.run_id, "estimator.pkl", "models")
    store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
    udc = store.load(filename="udc.pkl", as_type=nml.UnivariateDriftCalculator)
    estimator = store.load(filename="estimator.pkl", as_type=nml.CBPE)

    params = run_data_dict.get("params", {})

    try:
        log_model_meta = json.loads(run.data.tags["mlflow.log-model.history"])
        signature = log_model_meta[0].get("signature", None)
        if signature:
            inputs = signature["inputs"]
            if isinstance(inputs, str):
                inputs = json.loads(inputs)
            params["feature_columns"] = [inp["name"] for inp in inputs]
            logger.info(f"Extracted feature columns from signature: {params['feature_columns']}")
    except Exception as e:
        logger.warning(f"Signature not found in mlflow log-model history: {e}")

    if "feature_columns" in params:
        if isinstance(params["feature_columns"], str):
            try:
                params["feature_columns"] = ast.literal_eval(params["feature_columns"])
                logger.info(f"Parsed feature_columns from string: {params['feature_columns']}")
            except Exception as e:
                raise RuntimeError("Failed to parse feature_columns from string.") from e

    if "feature_columns" not in params:
        try:
            model_params_path = MODELS_DIR / "model_params.pkl"
            if model_params_path.exists():
                params = joblib.load(model_params_path)
                logger.info("Loaded feature_columns from model_params.pkl.")
        except Exception as e:
            raise RuntimeError("Cannot resolve feature_columns.") from e

    logger.info(f"Loaded test data with columns: {df_test.columns.tolist()}")
    expected_features = params.get("feature_columns", [])
    missing_cols = [col for col in expected_features if col not in df_test.columns]
    if missing_cols:
        raise RuntimeError(f"The following expected feature columns are missing in test data: {missing_cols}")

    preds_path = predict(loaded_model, df_test, params, probs=True)
    df_preds = pd.read_csv(preds_path)

    analysis_df = df_test.copy()
    analysis_df["prediction"] = df_preds["prediction"]
    analysis_df["predicted_probability"] = df_preds["predicted_probability"]
    git_hash = get_git_commit_hash()
    mlflow.set_experiment("diabetes_predictions")
    with mlflow.start_run(tags={"git_sha": git_hash}):
        estimated_performance = estimator.estimate(analysis_df)
        mlflow.log_figure(estimated_performance.plot(), "estimated_performance.png")

        drop_cols = ["prediction", "predicted_probability"]
        features = analysis_df.drop(columns=drop_cols, errors="ignore").columns
        univariate_drift = udc.calculate(analysis_df.drop(columns=drop_cols, errors="ignore"))

        for p in features:
            try:
                fig = univariate_drift.filter(column_names=[p]).plot()
                mlflow.log_figure(fig, f"univariate_drift_{p}.png")
                fig_dist = univariate_drift.filter(period="analysis", column_names=[p]).plot(kind="distribution")
                mlflow.log_figure(fig_dist, f"univariate_drift_dist_{p}.png")
            except Exception as e:
                logger.warning(f"Drift plotting failed for {p}: {e}")
        mlflow.log_params({"git_hash": git_hash})
        mlflow.log_artifact(preds_path, "predictions")
        mlflow.log_artifact(FIGURES_DIR / "diabetes_shap_overall.png", "figures")
        mlflow.log_artifact(MODELS_DIR / "udc.pkl", "models")
        mlflow.log_artifact(MODELS_DIR / "estimator.pkl", "models")
        mlflow.log_artifact(MODELS_DIR / "model_params.pkl", "models")
        mlflow.log_artifact(MODELS_DIR / "preds.csv", "models")
        mlflow.log_artifact(MODELS_DIR / "model_info.json", "models")

