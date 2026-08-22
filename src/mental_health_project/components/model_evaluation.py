import os
import sys
import numpy as np
import joblib
import mlflow

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.mental_health_project.logger import logging
from src.mental_health_project.exception import CustomException

from src.mental_health_project.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact,
    ModelEvaluationArtifact,
)

from src.mental_health_project.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig,
    ):

        try:

            self.data_transformation_artifact = data_transformation_artifact

            self.model_trainer_artifact = model_trainer_artifact

            self.model_evaluation_config = model_evaluation_config

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # LOAD TEST DATA
    # =========================================================

    def load_test_data(self):

        try:

            test_array = np.load(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_test = test_array[:, :-1]

            y_test = test_array[:, -1]

            return X_test, y_test

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # LOAD TRAINED MODEL
    # =========================================================

    def load_model(self):

        try:

            model = joblib.load(self.model_trainer_artifact.trained_model_file_path)

            return model

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # CALCULATE REGRESSION METRICS
    # =========================================================

    def calculate_metrics(self, actual, predicted):

        try:

            mae = mean_absolute_error(actual, predicted)

            mse = mean_squared_error(actual, predicted)

            rmse = np.sqrt(mse)

            r2 = r2_score(actual, predicted)

            metric_artifact = RegressionMetricArtifact(
                mae=mae, rmse=rmse, mse=mse, r2_score=r2
            )

            return metric_artifact

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # INITIATE MODEL EVALUATION
    # =========================================================

    def initiate_model_evaluation(self):

        try:

            logging.info("========== MODEL EVALUATION STARTED ==========")

            # -------------------------------------------------
            # STEP 1:
            # Load untouched test data
            # -------------------------------------------------

            X_test, y_test = self.load_test_data()

            logging.info(f"X_test shape: {X_test.shape}")

            logging.info(f"y_test shape: {y_test.shape}")

            # -------------------------------------------------
            # STEP 2:
            # Load final trained model
            # -------------------------------------------------

            model = self.load_model()

            logging.info("Final trained model loaded successfully")

            # -------------------------------------------------
            # STEP 3:
            # Make predictions on test data
            # -------------------------------------------------

            y_test_pred = model.predict(X_test)

            # -------------------------------------------------
            # STEP 4:
            # Calculate TEST metrics
            # -------------------------------------------------

            test_metric_artifact = self.calculate_metrics(y_test, y_test_pred)

            # -------------------------------------------------
            # STEP 5:
            # Get TRAIN metrics from ModelTrainerArtifact
            # -------------------------------------------------

            train_metric_artifact = self.model_trainer_artifact.train_metric_artifact

            # -------------------------------------------------
            # STEP 6:
            # Log Train vs Test performance
            # -------------------------------------------------

            logging.info("========== TRAIN METRICS ==========")

            logging.info(f"Train MAE: " f"{train_metric_artifact.mae:.4f}")

            logging.info(f"Train RMSE: " f"{train_metric_artifact.rmse:.4f}")

            logging.info(f"Train MSE: " f"{train_metric_artifact.mse:.4f}")

            logging.info(f"Train R2: " f"{train_metric_artifact.r2_score:.4f}")

            logging.info("========== TEST METRICS ==========")

            logging.info(f"Test MAE: " f"{test_metric_artifact.mae:.4f}")

            logging.info(f"Test RMSE: " f"{test_metric_artifact.rmse:.4f}")

            logging.info(f"Test MSE: " f"{test_metric_artifact.mse:.4f}")

            logging.info(f"Test R2: " f"{test_metric_artifact.r2_score:.4f}")

            # -------------------------------------------------
            # STEP 7:
            # Model acceptance condition
            # -------------------------------------------------

            r2_threshold = self.model_evaluation_config.r2_threshold

            model_accepted = test_metric_artifact.r2_score >= r2_threshold

            if model_accepted:

                logging.info(
                    f"MODEL ACCEPTED | "
                    f"Test R2 = "
                    f"{test_metric_artifact.r2_score:.4f}"
                )

            else:

                logging.info(
                    f"MODEL REJECTED | "
                    f"Test R2 = "
                    f"{test_metric_artifact.r2_score:.4f}"
                )

            with mlflow.start_run(run_name="FINAL_MODEL_EVALUATION"):

                # -----------------------------
                # Model information
                # -----------------------------

                mlflow.log_param("model", self.model_trainer_artifact.best_model_name)

                mlflow.log_metric("cv_r2", self.model_trainer_artifact.best_cv_score)

                # -----------------------------
                # Best parameters
                # -----------------------------

                best_params = self.model_trainer_artifact.best_params

                for param_name, value in best_params.items():
                    mlflow.log_param(param_name, value)

                # -----------------------------
                # Train metrics
                # -----------------------------

                mlflow.log_metric("train_r2", train_metric_artifact.r2_score)

                mlflow.log_metric("train_mae", train_metric_artifact.mae)

                mlflow.log_metric("train_rmse", train_metric_artifact.rmse)

                mlflow.log_metric("train_mse", train_metric_artifact.mse)

                # -----------------------------
                # Test metrics
                # -----------------------------

                mlflow.log_metric("test_r2", test_metric_artifact.r2_score)

                mlflow.log_metric("test_mae", test_metric_artifact.mae)

                mlflow.log_metric("test_rmse", test_metric_artifact.rmse)

                mlflow.log_metric("test_mse", test_metric_artifact.mse)

                # -----------------------------
                # Model acceptance
                # -----------------------------

                mlflow.log_param("model_accepted", model_accepted)

            # -------------------------------------------------
            # STEP 8:
            # Create ModelEvaluationArtifact
            # -------------------------------------------------
            model_evaluation_artifact = ModelEvaluationArtifact(
                trained_model_file_path=self.model_trainer_artifact.trained_model_file_path,
                best_model_name=self.model_trainer_artifact.best_model_name,
                best_cv_score=self.model_trainer_artifact.best_cv_score,
                best_params=self.model_trainer_artifact.best_params,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact,
                model_accepted=model_accepted,
            )

            logging.info("========== MODEL EVALUATION COMPLETED ==========")

            return model_evaluation_artifact

        except Exception as e:

            raise CustomException(e, sys)
