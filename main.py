from src.mental_health_project.entity.config_entity import (
DataIngestionConfig,
 TrainingPipelineConfig,
 DataValidationConfig,
 DataTransformationConfig,
 ModelTrainerConfig,ModelEvaluationConfig
 )
from src.mental_health_project.components.data_ingestion import DataIngestion
from src.mental_health_project.components.data_validation import DataValidaion
from src.mental_health_project.components.data_transformation import DataTransformation
from src.mental_health_project.components.model_trainer import ModelTrainer
from src.mental_health_project.components.model_evaluation import ModelEvaluation
from src.mental_health_project.utils.mlflow_utils import setup_mlflow
from src.mental_health_project.pipelines.training_pipeline import TrainingPipeline

from src.mental_health_project.logger import logging
from src.mental_health_project.exception import CustomException
import sys

if __name__ == "__main__":
    try:
        setup_mlflow()

        training_pipeline = TrainingPipeline()

        training_pipeline.run_pipeline()

    except Exception as e:
        raise CustomException(e,sys)
