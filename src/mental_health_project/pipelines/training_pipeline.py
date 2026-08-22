from src.mental_health_project.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from src.mental_health_project.cloud.aws_storage import S3Sync
from src.mental_health_project.constant.training_pipeline import TRAINING_BUCKET_NAME
from src.mental_health_project.components.data_ingestion import DataIngestion
from src.mental_health_project.components.data_validation import DataValidaion
from src.mental_health_project.components.data_transformation import DataTransformation
from src.mental_health_project.components.model_trainer import ModelTrainer
from src.mental_health_project.components.model_evaluation import ModelEvaluation
from src.mental_health_project.utils.mlflow_utils import setup_mlflow

from src.mental_health_project.logger import logging
from src.mental_health_project.exception import CustomException
import sys


class TrainingPipeline:

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self):
        try:

            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("Start data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            self.data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(
                f"Data Ingestion completed and artifact: {self.data_ingestion_artifact}"
            )
            return self.data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self):

        try:
            self.data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidaion(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=self.data_ingestion_artifact,
            )
            logging.info("Initiate the data Validation")
            self.data_validation_artifact = data_validation.initiate_data_validation()
            return self.data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self):
        try:

            self.data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_transformation = DataTransformation(
                data_validation_artifact=self.data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
            )

            self.data_transformation_artifact = (
                data_transformation.initiate_data__transformation()
            )

            return self.data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self):
        try:
            self.model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=self.data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            self.model_trainer_artifact = model_trainer.initiate_model_trainer()

            return self.model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(self):
        try:

            self.model_evaluation_config = ModelEvaluationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_evaluation = ModelEvaluation(
                data_transformation_artifact=self.data_transformation_artifact,
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifact=self.model_trainer_artifact,
            )

            self.model_evaluation_artifact = (
                model_evaluation.initiate_model_evaluation()
            )
            return self.model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def sync_artifact_dir_to_s3(self):

        try:

            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                bucket_name=TRAINING_BUCKET_NAME,
                s3_prefix=(f"artifact/" f"{self.training_pipeline_config.timestamp}"),
            )

            logging.info("Artifacts successfully synced to S3")

        except Exception as e:

            raise CustomException(e, sys)

    def sync_model_to_s3(self):

        try:

            model_path = self.model_trainer_artifact.trained_model_file_path

            self.s3_sync.upload_file_to_s3(
                local_file_path=model_path,
                bucket_name=TRAINING_BUCKET_NAME,
                s3_key=(
                    f"final_model/"
                    f"{self.training_pipeline_config.timestamp}/"
                    f"model.pkl"
                ),
            )

            logging.info("Final model successfully synced to S3")

        except Exception as e:

            raise CustomException(e, sys)

    def sync_transformation_pipeline_to_s3(self):

        try:

            transformation_pipeline_path = (
                self.data_transformation_artifact
                .transformation_pipeline_file_path
            )

            self.s3_sync.upload_file_to_s3(
                local_file_path=transformation_pipeline_path,
                bucket_name=TRAINING_BUCKET_NAME,
                s3_key=(
                    f"transformation_pipeline/"
                    f"{self.training_pipeline_config.timestamp}/"
                    f"transformation_pipeline.pkl"
                )
            )

            logging.info(
                "Transformation pipeline successfully synced to S3"
            )

        except Exception as e:
            raise CustomException(e, sys)
            

    def update_production_model(self):

        try:

            version_file = "model_version.txt"

            with open(version_file, "w") as f:
                f.write(
                    self.training_pipeline_config.timestamp
                )

            self.s3_sync.upload_file_to_s3(
                local_file_path=version_file,
                bucket_name=TRAINING_BUCKET_NAME,
                s3_key="production/model_version.txt"
            )

            logging.info(
                "Production model version updated"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            self.start_data_ingestion()
            self.start_data_validation()
            self.start_data_transformation()
            self.start_model_trainer()
            self.start_model_evaluation()
            self.sync_artifact_dir_to_s3()
            if self.model_evaluation_artifact.model_accepted:

                logging.info("Model accepted. Uploading final model and preprocessor to S3.")

                self.sync_model_to_s3()
                self.sync_transformation_pipeline_to_s3()
                self.update_production_model()


            else:

                logging.info("Model rejected. Final model will NOT be uploaded to S3.")

                return self.model_trainer_artifact

        
        except Exception as e:
            raise CustomException(e, sys)
