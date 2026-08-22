import os
import sys
import joblib
import numpy as np
from mental_health_project.utils.mlflow_utils import setup_mlflow
import mlflow
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor


from src.mental_health_project.logger import logging
from src.mental_health_project.exception import CustomException

from src.mental_health_project.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact,
)

from src.mental_health_project.entity.config_entity import ModelTrainerConfig


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):

        try:

            self.data_transformation_artifact = data_transformation_artifact

            self.model_trainer_config = model_trainer_config

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # LOAD TRAINING DATA
    # =========================================================

    def load_training_data(self):

        try:

            train_array = np.load(
                self.data_transformation_artifact.transformed_train_file_path
            )

            X_train = train_array[:, :-1]

            y_train = train_array[:, -1]

            return X_train, y_train

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # GET ALL MODELS
    # =========================================================

    def get_models(self):

        try:

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
            }

            return models

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # HYPERPARAMETER GRIDS
    # =========================================================

    def get_params(self):

        try:

            params = {
                "Linear Regression": {"fit_intercept": [True, False]},
                "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
                "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
                "ElasticNet": {
                    "alpha": [0.001, 0.01, 0.1, 1],
                    "l1_ratio": [0.2, 0.5, 0.8],
                },
                "Decision Tree": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0],
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },
            }

            return params

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # EVALUATE ALL MODELS USING 5-FOLD CROSS VALIDATION
    # =========================================================

    def evaluate_models(self, models, X_train, y_train):

        try:

            results = {}
            with mlflow.start_run(run_name="MODEL_SELECTION"):

                for model_name, model in models.items():

                    logging.info(f"Evaluating {model_name} " f"using 5-Fold CV")

                    scores = cross_val_score(
                        model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
                    )

                    cv_mean = scores.mean()

                    cv_std = scores.std()

                    results[model_name] = {
                        "model": model,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                    }

                    with mlflow.start_run(run_name=f"{model_name}__CV",nested=True):

                        mlflow.log_param("model", model_name)
                        mlflow.log_metric("cv_mean_r2", cv_mean)
                        mlflow.log_param("n_folds", 5)
                        mlflow.log_metric("cv_std_r2", cv_std)

                    logging.info(
                        f"{model_name} | "
                        f"CV R2 = {cv_mean:.4f} | "
                        f"STD = {cv_std:.4f}"
                    )

                return results

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # SELECT TOP 3 MODELS
    # =========================================================

    def select_top_models(self, cv_results, top_n=3):

        try:

            # Sort models by highest CV R2

            sorted_models = sorted(
                cv_results.items(), key=lambda x: x[1]["cv_mean"], reverse=True
            )

            # Select top 3

            top_models = dict(sorted_models[:top_n])

            logging.info("========== TOP MODELS ==========")

            for model_name, result in top_models.items():

                logging.info(
                    f"Selected: {model_name} | "
                    f"CV R2: {result['cv_mean']:.4f} | "
                    f"STD: {result['cv_std']:.4f}"
                )

            return top_models

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # RANDOMIZED SEARCH ON TOP 3 MODELS ONLY
    # =========================================================

    def tune_top_models(self, top_models, params, X_train, y_train):

        try:

            tuned_models = {}
            with mlflow.start_run(run_name="TOP_3_TUNING"):

                for model_name, result in top_models.items():

                    logging.info(f"Starting RandomizedSearchCV " f"for {model_name}")

                    model = result["model"]

                    param_grid = params[model_name]

                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grid,
                        n_iter=10,
                        cv=5,
                        scoring="r2",
                        random_state=42,
                        n_jobs=1,
                        refit=True,
                    )

                    random_search.fit(X_train, y_train)

                    best_model = random_search.best_estimator_

                    best_score = random_search.best_score_

                    best_params = random_search.best_params_

                    tuned_models[model_name] = {
                        "model": random_search.best_estimator_,
                        "cv_score": random_search.best_score_,
                        "best_params": random_search.best_params_,
                    }

                    logging.info(
                        f"{model_name} | "
                        f"Best CV R2: "
                        f"{random_search.best_score_:.4f}"
                    )

                    logging.info(
                        f"{model_name} | "
                        f"Best Params: "
                        f"{random_search.best_params_}"
                    )

                    with mlflow.start_run(run_name=f"{model_name}__Tuned",nested=True):

                        mlflow.log_param("model", model_name)
                        mlflow.log_param("n_iter", 10)
                        mlflow.log_param("cv_folds", 5)

                        for param_name, value in best_params.items():

                            mlflow.log_param(param_name, value)
                            mlflow.log_metric("best_cv_r2", best_score)

                return tuned_models

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # SELECT BEST TUNED MODEL
    # =========================================================

    def select_best_tuned_model(self, tuned_models):

        try:

            best_model_name = max(
                tuned_models, key=lambda name: tuned_models[name]["cv_score"]
            )

            best_result = tuned_models[best_model_name]

            best_model = best_result["model"]

            best_cv_score = best_result["cv_score"]

            best_params = best_result["best_params"]

            return (best_model_name, best_model, best_cv_score, best_params)

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # CALCULATE TRAINING METRICS
    # =========================================================

    def calculate_train_metrics(self, model, X_train, y_train):

        try:

            y_train_pred = model.predict(X_train)

            mae = mean_absolute_error(y_train, y_train_pred)

            mse = mean_squared_error(y_train, y_train_pred)

            rmse = np.sqrt(mse)

            r2 = r2_score(y_train, y_train_pred)

            train_metric_artifact = RegressionMetricArtifact(
                mae=mae, rmse=rmse, mse=mse, r2_score=r2
            )

            return train_metric_artifact

        except Exception as e:

            raise CustomException(e, sys)

    # =========================================================
    # INITIATE MODEL TRAINER
    # =========================================================

    def initiate_model_trainer(self):

        try:

            logging.info("========== MODEL TRAINING STARTED ==========")

            # -------------------------------------------------
            # STEP 1: Load transformed training data
            # -------------------------------------------------

            X_train, y_train = self.load_training_data()

            logging.info(f"X_train shape: {X_train.shape}")

            logging.info(f"y_train shape: {y_train.shape}")

            # -------------------------------------------------
            # STEP 2: Get all candidate models
            # -------------------------------------------------

            models = self.get_models()

            # -------------------------------------------------
            # STEP 3: Get hyperparameter grids
            # -------------------------------------------------

            params = self.get_params()

            # -------------------------------------------------
            # STEP 4:
            # Evaluate ALL models using 5-Fold CV
            # -------------------------------------------------

            cv_results = self.evaluate_models(models, X_train, y_train)

            # -------------------------------------------------
            # STEP 5:
            # Select TOP 3 models
            # -------------------------------------------------

            top_models = self.select_top_models(cv_results, top_n=3)

            # -------------------------------------------------
            # STEP 6:
            # Run RandomizedSearchCV ONLY
            # on the TOP 3 models
            # -------------------------------------------------

            tuned_models = self.tune_top_models(top_models, params, X_train, y_train)

            # -------------------------------------------------
            # STEP 7:
            # Select BEST tuned model
            # -------------------------------------------------

            best_model_name, best_model, best_cv_score, best_params = (
                self.select_best_tuned_model(tuned_models)
            )

            logging.info(f"Best Model: " f"{best_model_name}")

            logging.info(f"Best CV R2: " f"{best_cv_score:.4f}")

            logging.info(f"Best Params: " f"{best_params}")

            # -------------------------------------------------
            # STEP 8:
            # Calculate training metrics
            # -------------------------------------------------

            train_metric_artifact = self.calculate_train_metrics(
                best_model, X_train, y_train
            )

            logging.info(f"Train MAE: " f"{train_metric_artifact.mae:.4f}")

            logging.info(f"Train RMSE: " f"{train_metric_artifact.rmse:.4f}")

            logging.info(f"Train MSE: " f"{train_metric_artifact.mse:.4f}")

            logging.info(f"Train R2: " f"{train_metric_artifact.r2_score:.4f}")

            # -------------------------------------------------
            # STEP 9:
            # Create model directory
            # -------------------------------------------------

            model_path = self.model_trainer_config.trained_model_file_path

            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # -------------------------------------------------
            # STEP 10:
            # Save final trained model
            # -------------------------------------------------

            joblib.dump(best_model, model_path)

            logging.info(f"Best model saved at: " f"{model_path}")

            # -------------------------------------------------
            # STEP 11:
            # Create ModelTrainerArtifact
            # -------------------------------------------------

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=model_path,
                best_model_name=best_model_name,
                best_cv_score=best_cv_score,
                best_params=best_params,
                train_metric_artifact=train_metric_artifact,
            )

            logging.info("========== MODEL TRAINING COMPLETED ==========")

            return model_trainer_artifact

        except Exception as e:

            raise CustomException(e, sys)
