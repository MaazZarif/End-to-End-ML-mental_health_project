import os
from datetime import datetime
from src.mental_health_project.constant import training_pipeline
from dataclasses import dataclass


class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir = os.path.join("final_model")
        self.timestamp: str = timestamp


class DataIngestionConfig:

    def __init__(self, training_pipeline_config):

        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )

        self.feature_store_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.RAW_FILE_NAME,
        )

        self.train_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME,
        )

        self.test_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME,
        )

        self.train_test_split_ratio: float = (
            training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )


class DataValidationConfig:

    def __init__(self, training_pipeline_config):

        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR
        )
        self.data_validation_valid_dir = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.data_validation_vaild_train_file_path = os.path.join(
            self.data_validation_valid_dir, training_pipeline.TRAIN_FILE_NAME
        )
        self.data_validation_vaild_test_file_path = os.path.join(
            self.data_validation_valid_dir, training_pipeline.TEST_FILE_NAME
        )
        self.data_validation_invalid_dir = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        self.data_validation_invaild_train_file_path = os.path.join(
            self.data_validation_invalid_dir, training_pipeline.TRAIN_FILE_NAME
        )
        self.data_validation_invaild_test_file_path = os.path.join(
            self.data_validation_invalid_dir, training_pipeline.TEST_FILE_NAME
        )
        self.data_validation_drift_report_filepath = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config):
        self.data_transformation_file_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
        )
        self.transformed_data_file_dir = os.path.join(
            self.data_transformation_file_dir,
            training_pipeline.TRANSFORMED_DATA_FILE_DIR,
        )
        self.transfomed_object_file_dir = os.path.join(
            self.data_transformation_file_dir,
            training_pipeline.TRANSFORMED_OBJECT_FILE_DIR,
        )
        self.transformed_train_file_path = os.path.join(
            self.transformed_data_file_dir,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path = os.path.join(
            self.transformed_data_file_dir,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.transformation_pipeline_file_path = os.path.join(
            self.transfomed_object_file_dir,
            "transformation_pipeline.pkl",
        )


class ModelTrainerConfig:

    def __init__(self, training_pipeline_config):

        self.model_trainer_file_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME,
        )

        self.trained_model_dir = os.path.join(
            self.model_trainer_file_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
        )

        self.trained_model_file_path = os.path.join(
            self.trained_model_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME
        )


class ModelEvaluationConfig:

    def __init__(self, training_pipeline_config):

        self.r2_threshold = 0.85

        self.model_evaluation_file_dir = os.path.join(
            training_pipeline_config.artifact_dir, "model_evaluation"
        )

        self.evaluation_report_file_path = os.path.join(
            self.model_evaluation_file_dir, "evaluation_report.yaml"
        )


class PredictionPipelineConfig:

    def __init__(self):

        self.model_version_s3_key = "production/model_version.txt"

        self.model_s3_prefix = "final_model"

        self.transformation_pipeline_s3_prefix = "transformation_pipeline"

        self.model_path = os.path.join("prediction_artifacts", "model.pkl")

        self.transformation_pipeline_path = os.path.join(
            "prediction_artifacts", "transformation_pipeline.pkl"
        )
