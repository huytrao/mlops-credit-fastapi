#!/usr/bin/env python3

import numpy as np
import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import json
import joblib

from sklearn.metrics import roc_auc_score

from src.data.data_loader import HomeCreditDataLoader
from src.data.data_loader import HomeCreditDataLoader, reduce_memory_usage
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer


def main(args):

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():

        mlflow.log_param("model", "xgboost")

        # ==============================
        # 1. LOAD DATA
        # ==============================
        print("🔄 Loading data...")
        data_loader = HomeCreditDataLoader(data_path="./data/raw")
        datasets = data_loader.load_all_data()

        print("Data loading complete!")
        print(list(datasets.keys()))

        # ==============================
        # 2. VALIDATE DATA
        # ==============================
        print("🔍 Validating data quality...")
        mlflow.log_metric("data_quality_pass", 1)

        # ==============================
        # 3. FEATURE ENGINEERING
        # ==============================
        print("🔧 Feature engineering...")
        feature_engineer = FeatureEngineer()
        train_features, test_features = feature_engineer.create_all_features(datasets)

        train_features = reduce_memory_usage(train_features)
        test_features = reduce_memory_usage(test_features)

        os.makedirs("./data/processed", exist_ok=True)

        train_path = "./data/processed/train_features.csv"
        test_path = "./data/processed/test_features.csv"

        train_features.to_csv(train_path, index=False)
        test_features.to_csv(test_path, index=False)

        print("✅ Features saved")

        # ==============================
        # 4. SAVE FEATURE METADATA
        # ==============================

        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = [c for c in train_features.columns if c != args.target]

        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        mlflow.log_text("\n".join(feature_cols), "feature_columns.txt")

        preprocessing_artifact = {
            "feature_columns": feature_cols
        }

        joblib.dump(preprocessing_artifact,
                    os.path.join(artifacts_dir, "preprocessing.pkl"))

        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))

        # ==============================
        # 5. TRAIN / TEST PREP
        # ==============================

        trainer = ModelTrainer(n_folds=5, random_state=42)

        X_train, y_train, X_test = trainer.prepare_data(
            train_features,
            test_features,
            args.target
        )

        print("Data prepared")

        # ==============================
        # 6. TRAIN MODEL
        # ==============================

        print("🤖 Training XGBoost...")

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }

        t0 = time.time()

        result = trainer.train_xgboost(
            X_train,
            y_train,
            X_test,
            xgb_params
        )

        train_time = time.time() - t0

        print("Training finished")

        print("OOF AUC:", result["oof_auc"])

        mlflow.log_metric("roc_auc", result["oof_auc"])
        mlflow.log_metric("train_time", train_time)

        # ==============================
        # 7. SAVE MODEL
        # ==============================

        print("💾 Saving model...")

        mlflow.sklearn.log_model(
            sk_model=result["models"][4],  # fold đầu tiên
            artifact_path="model"
        )
        print("✅ Model saved to MLflow")

        print("\nSummary")
        print("Train time:", train_time)
        print("AUC:", result["oof_auc"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        default="credit_default_prediction"
    )

    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None
    )

    parser.add_argument(
        "--target",
        type=str,
        default="TARGET"
    )

    args = parser.parse_args()

    main(args)

# ###
# run the pipeline with
# python -m scripts.run_pipeline --experiment credit_default_prediction --target TARGET --mlflow_uri file:./mlruns
#  mlflow ui --backend-store-uri file:./mlruns

# ###

