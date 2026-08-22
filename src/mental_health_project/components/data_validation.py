from src.mental_health_project.logger import logging
from src.mental_health_project.exception import CustomException
from src.mental_health_project.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from src.mental_health_project.entity.config_entity import DataValidationConfig
from src.mental_health_project.constant.training_pipeline import SCHEMA_FILE_PATH
from mental_health_project.utils.utils import read_yaml,write_yaml_file
from scipy.stats import ks_2samp
import sys
import os
import pandas as pd

class DataValidaion:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)


    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)


    def validate_columns(self,dataframe):
        try:
            number_of_columns = len(self._schema_config)
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has number of columns:{len(dataframe.columns)}")

            if len(dataframe.columns) == number_of_columns:
              return True

            return False    

        except Exception as e:
            raise CustomException(e,sys)


    def detect_dataset_drift(self,base_df,current_df,threshold=0.05):

        try:
            status = True
            report = {}

            for column in base_df.columns:
                df1 = base_df[column]
                df2 = current_df[column]

                is_same_dist = ks_2samp(df1,df2)
                if threshold<=is_same_dist.pvalue:
                    is_found = False

                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                }})

                drift_report_file_path = self.data_validation_config.data_validation_drift_report_filepath

                dir_path = os.path.dirname(drift_report_file_path)
                os.makedirs(dir_path,exist_ok=True)
                write_yaml_file(file_path=drift_report_file_path,content=report)

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_validation(self):
        try:

            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.tested_file_path

            train_dataframe = DataValidaion.read_data(train_file_path)
            test_dataframe = DataValidaion.read_data(test_file_path)

            status = self.validate_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"Train dataframe does not contain all columns.\n"

            status = self.validate_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"Test dataframe does not contain all columns.\n"

            status = self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)

            dir_path = os.path.dirname(self.data_validation_config.data_validation_vaild_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(self.data_validation_config.data_validation_vaild_train_file_path,index=False,header=True)
            test_dataframe.to_csv(self.data_validation_config.data_validation_vaild_test_file_path,index=False,header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.tested_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.data_validation_drift_report_filepath

            )
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)











