import os


"""
Defining common constant variables for training variables
"""

PIPELINE_NAME = "mental_health_project"
ARTIFACT_DIR = "artifacts"
TRAINING_BUCKET_NAME = "student-mental-health-ml-maaz"
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_DIR_NAME = "data_ingestion"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
RAW_FILE_NAME = "data.csv"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
Data validation related constants
"""


DATA_VALIDATION_DIR = "data_validation"
DATA_VALIDATION_INVALID_DIR = "invalid"
DATA_VALIDATION_VALID_DIR = "validated"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


"""
Data Transformation related constants
"""

DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
TRANSFORMED_OBJECT_FILE_DIR = "transformed_object"
TRANSFORMED_DATA_FILE_DIR = "transformed"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"



"""
Model Trainer related constants
"""

MODEL_TRAINER_DIR_NAME="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR="trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME = "model.pkl"
