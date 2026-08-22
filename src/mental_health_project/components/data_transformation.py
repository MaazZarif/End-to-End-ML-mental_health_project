import pandas as pd
import numpy as np
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
import joblib

from src.mental_health_project.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.mental_health_project.entity.config_entity import DataTransformationConfig
from src.mental_health_project.logger import logging
from src.mental_health_project.exception import CustomException


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        try:
            X = X.copy()

            self.top_countries = (
                X["Country"]
                .value_counts()
                .head(10)
                .index
                .tolist()
            )

            return self

        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X):
        try:
            X = X.copy()

            X["Physical_Activity_Hours"] = (
                X["Physical_Activity_Hours"].clip(lower=0)
            )

            X["Grouped_country"] = X["Country"].apply(
                lambda x: x if x in self.top_countries else "Other"
            )

            X.drop(columns=["Country"], inplace=True)

            return X

        except Exception as e:
            raise CustomException(e, sys)


class DataTransformation:

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise CustomException(e, sys)

    def get_preprocessor(self):

        try:
            skewed_col = ["Study_Hours"]

            other_numeric_cols = [
                "Age",
                "Avg_Daily_Usage_Hours",
                "Daily_Unlocks",
                "Physical_Activity_Hours",
                "Sleep_Hours_Per_Night",
            ]

            ordinal_col = ["Stress_Level"]

            normal_col = [
                "Gender",
                "Academic_Level",
                "Most_Used_Platform",
                "Purpose_Of_Use",
                "Grouped_country",
            ]

            skew_pipeline = Pipeline(
                steps=[
                    ("log_transform", FunctionTransformer(np.log1p)),
                    ("scale", StandardScaler()),
                ]
            )

            plain_numeric_pipeline = Pipeline(
                steps=[
                    ("scale", StandardScaler())
                ]
            )

            ordinal_pipeline = OrdinalEncoder(
                categories=[
                    ["Low", "Medium", "High", "Very High"]
                ]
            )

            nominal_pipeline = Pipeline(
                steps=[
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        ),
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "skewed_pipeline",
                        skew_pipeline,
                        skewed_col
                    ),
                    (
                        "numeric_pipeline",
                        plain_numeric_pipeline,
                        other_numeric_cols
                    ),
                    (
                        "ordinal_pipeline",
                        ordinal_pipeline,
                        ordinal_col
                    ),
                    (
                        "nominal_pipeline",
                        nominal_pipeline,
                        normal_col
                    ),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def read_data(self, file_path):

        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data__transformation(self):

        try:

            logging.info("Data transformation started.")

            train_file = (
                self.data_validation_artifact.valid_train_file_path
            )

            test_file = (
                self.data_validation_artifact.valid_test_file_path
            )

            target = "Mental_Health_Score"

            train_df = self.read_data(train_file)
            test_df = self.read_data(test_file)

            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()

            input_train_df = train_df.drop(
                columns=[target]
            )

            target_train_df = train_df[target]

            input_test_df = test_df.drop(
                columns=[target]
            )

            target_test_df = test_df[target]

            transformation_pipeline = Pipeline(
                steps=[
                    (
                        "feature_engineering",
                        FeatureEngineeringTransformer()
                    ),
                    (
                        "preprocessor",
                        self.get_preprocessor()
                    ),
                ]
            )

            transformed_input_train_feature = (
                transformation_pipeline.fit_transform(
                    input_train_df
                )
            )

            transformed_input_test_feature = (
                transformation_pipeline.transform(
                    input_test_df
                )
            )

            logging.info(
                "Feature engineering and preprocessing completed."
            )

            train_arr = np.c_[
                transformed_input_train_feature,
                np.array(target_train_df)
            ]

            test_arr = np.c_[
                transformed_input_test_feature,
                np.array(target_test_df)
            ]

            transformed_data_dir = os.path.dirname(
                self.data_transformation_config.transformed_test_file_path
            )

            os.makedirs(
                transformed_data_dir,
                exist_ok=True
            )

            transformation_pipeline_dir = os.path.dirname(
                self.data_transformation_config
                .transformation_pipeline_file_path
            )

            os.makedirs(
                transformation_pipeline_dir,
                exist_ok=True
            )

            np.save(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            np.save(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            joblib.dump(
                transformation_pipeline,
                self.data_transformation_config
                .transformation_pipeline_file_path
            )

            logging.info(
                "Transformation pipeline saved successfully."
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformation_pipeline_file_path=(
                    self.data_transformation_config
                    .transformation_pipeline_file_path
                ),
                transformed_train_file_path=(
                    self.data_transformation_config
                    .transformed_train_file_path
                ),
                transformed_test_file_path=(
                    self.data_transformation_config
                    .transformed_test_file_path
                ),
            )

            logging.info(
                "Data transformation completed successfully."
            )

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)