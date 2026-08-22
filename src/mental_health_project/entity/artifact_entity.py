from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:

    trained_file_path:str 
    tested_file_path:str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformation_pipeline_file_path: str
    transformed_train_file_path:str
    transformed_test_file_path:str


@dataclass
class RegressionMetricArtifact:

    mae: float
    rmse: float
    mse: float
    r2_score: float


@dataclass
class ModelTrainerArtifact:

    trained_model_file_path: str

    best_model_name: str

    best_cv_score: float

    best_params: dict

    train_metric_artifact: RegressionMetricArtifact


@dataclass
class ModelEvaluationArtifact:

    trained_model_file_path: str

    best_model_name: str

    best_cv_score: float

    best_params: dict

    train_metric_artifact: RegressionMetricArtifact

    test_metric_artifact: RegressionMetricArtifact

    model_accepted: bool