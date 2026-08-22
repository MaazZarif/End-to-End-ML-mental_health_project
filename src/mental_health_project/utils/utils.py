import pandas as pd
from src.mental_health_project.logger import logging
from src.mental_health_project.exception import CustomException
import sys
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import yaml
import pickle
load_dotenv()

HOST = os.getenv("DB_HOST")
DATABASE = os.getenv("DB_NAME")
USERNAME = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
PORT = int(os.getenv("DB_PORT", 3306))


def read_sql_data():
    logging.info("Reading SQL Database Started")

    try:

        connection_string = (
            f"mysql+pymysql://{USERNAME}:{PASSWORD}"
            f"@{HOST}:{PORT}/{DATABASE}"
        )

        engine = create_engine(connection_string)

        query = "Select * from mental_health_data"


        df = pd.read_sql(query, engine)

        logging.info("Connection Established")

        return df

    except Exception as e:
        raise CustomException(e,sys)

    
def read_yaml(filepath):

    try:
        with open(filepath,"rb") as yaml_file:

            return yaml.safe_load(yaml_file)

    except Exception as e:
            raise CustomException(e,sys)

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(filepath):

    try:
        with open(filepath,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    



