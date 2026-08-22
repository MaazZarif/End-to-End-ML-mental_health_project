import sys
import os
import joblib

from src.mental_health_project.exception import CustomException
from src.mental_health_project.logger import logging
from src.mental_health_project.cloud.aws_storage import S3Sync
from src.mental_health_project.entity.config_entity import PredictionPipelineConfig
from src.mental_health_project.constant.training_pipeline import TRAINING_BUCKET_NAME


class PredictionPipeline:

    def __init__(self):

        try:

            self.prediction_pipeline_config = PredictionPipelineConfig()

            self.s3_sync = S3Sync()

            self.bucket_name = TRAINING_BUCKET_NAME

            os.makedirs(
                os.path.dirname(
                    self.prediction_pipeline_config.model_path
                ),
                exist_ok=True
            )

            model_version = self.get_model_version()

            self.download_model(model_version)

            self.download_transformation_pipeline(model_version)

            self.model = self.load_model()

            self.transformation_pipeline = (
                self.load_transformation_pipeline()
            )

            logging.info(
                "Prediction pipeline initialized successfully"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def get_model_version(self):

        try:

            logging.info(
                "Getting production model version from S3"
            )

            model_version = self.s3_sync.read_file_from_s3(
                bucket_name=self.bucket_name,
                s3_key=(
                    self.prediction_pipeline_config
                    .model_version_s3_key
                )
            )

            if not model_version:
                raise Exception(
                    "Production model version is empty"
                )

            model_version = model_version.strip()

            logging.info(
                f"Production model version: {model_version}"
            )

            return model_version

        except Exception as e:
            raise CustomException(e, sys)

    def download_model(self, model_version):

        try:

            model_s3_key = (
                f"{self.prediction_pipeline_config.model_s3_prefix}/"
                f"{model_version}/"
                f"model.pkl"
            )

            logging.info(
                f"Downloading model: {model_s3_key}"
            )

            self.s3_sync.download_file_from_s3(
                bucket_name=self.bucket_name,
                s3_key=model_s3_key,
                local_file_path=(
                    self.prediction_pipeline_config.model_path
                )
            )

            logging.info(
                "Model downloaded successfully"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def download_transformation_pipeline(self, model_version):

        try:

            pipeline_s3_key = (
                f"{self.prediction_pipeline_config.transformation_pipeline_s3_prefix}/"
                f"{model_version}/"
                f"transformation_pipeline.pkl"
            )

            logging.info(
                f"Downloading transformation pipeline: "
                f"{pipeline_s3_key}"
            )

            self.s3_sync.download_file_from_s3(
                bucket_name=self.bucket_name,
                s3_key=pipeline_s3_key,
                local_file_path=(
                    self.prediction_pipeline_config
                    .transformation_pipeline_path
                )
            )

            logging.info(
                "Transformation pipeline downloaded successfully"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def load_model(self):

        try:

            model = joblib.load(
                self.prediction_pipeline_config.model_path
            )

            logging.info(
                "Model loaded successfully"
            )

            return model

        except Exception as e:
            raise CustomException(e, sys)

    def load_transformation_pipeline(self):

        try:

            transformation_pipeline = joblib.load(
                self.prediction_pipeline_config
                .transformation_pipeline_path
            )

            logging.info(
                "Transformation pipeline loaded successfully"
            )

            return transformation_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):

        try:

            logging.info(
                "========== PREDICTION STARTED =========="
            )

            transformed_data = (
                self.transformation_pipeline.transform(
                    features
                )
            )

            logging.info(
                "Feature engineering and preprocessing completed"
            )

            prediction = self.model.predict(
                transformed_data
            )

            logging.info(
                f"Prediction: {prediction}"
            )

            logging.info(
                "========== PREDICTION COMPLETED =========="
            )

            return prediction

        except Exception as e:
            raise CustomException(e, sys)