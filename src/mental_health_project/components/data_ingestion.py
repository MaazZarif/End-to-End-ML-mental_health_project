import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.mental_health_project.logger import logging
from src.mental_health_project.exception import CustomException
from dotenv import load_dotenv
from src.mental_health_project.entity.config_entity import DataIngestionConfig
from src.mental_health_project.entity.artifact_entity import DataIngestionArtifact
from mental_health_project.utils.utils import read_sql_data
import os
import sys

load_dotenv()


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:

        
            train_set,test_set = train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio,)
            logging.info("Performed train test split on the dataframe")

            logging.info(
                    "Exited split_data_as_train_test method of Data_Ingestion class"
                )

            feature_store_path = self.data_ingestion_config.feature_store_path

            feature_store_dir = os.path.dirname(feature_store_path)

            os.makedirs(feature_store_dir,exist_ok=True)

            dataframe.to_csv(feature_store_path,index=False,header=True)



            ingested_dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)

            os.makedirs(ingested_dir_path,exist_ok=True)

            logging.info(f"Exporting train and test file path.")

            train_set.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)

            test_set.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)

            logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise CustomException(e,sys)




    def initiate_data_ingestion(self):
        try:

            dataframe = read_sql_data()

            self.split_data_as_train_test(dataframe=dataframe)

            dataingestionartifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.train_file_path,tested_file_path=self.data_ingestion_config.test_file_path)

            return dataingestionartifact

        except Exception as e:
            raise CustomException(e,sys)






