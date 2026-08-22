import os
import boto3
import sys

from src.mental_health_project.exception import CustomException
from src.mental_health_project.logger import logging
from dotenv import load_dotenv

load_dotenv()

class S3Sync:

    def __init__(self):
        try:
            self.s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))

        except Exception as e:
            raise CustomException(e, sys)

    def sync_folder_to_s3(self, folder, bucket_name, s3_prefix):
        try:

            for root, dirs, files in os.walk(folder):

                for file in files:

                    local_file_path = os.path.join(root, file)

                    relative_path = os.path.relpath(local_file_path, folder)

                    s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

                    self.s3.upload_file(local_file_path, bucket_name, s3_key)

                    print(
                        f"Uploaded: {local_file_path} " f"→ s3://{bucket_name}/{s3_key}"
                    )

        except Exception as e:
            raise CustomException(e, sys)

    def upload_file_to_s3(self, local_file_path, bucket_name, s3_key):

        try:

            self.s3.upload_file(local_file_path, bucket_name, s3_key)

            logging.info(
                f"Uploaded {local_file_path} " f"to s3://{bucket_name}/{s3_key}"
            )

        except Exception as e:

            raise CustomException(e, sys)

    def download_file_from_s3(self, local_file_path, bucket_name, s3_key):

        try:

            # Create local directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            self.s3.download_file(bucket_name, s3_key, local_file_path)

            logging.info(
                f"Downloaded s3://{bucket_name}/{s3_key} " f"to {local_file_path}"
            )

        except Exception as e:

            raise CustomException(e, sys)

    def read_file_from_s3(self, bucket_name, s3_key):

        try:

            response = self.s3.get_object(Bucket=bucket_name, Key=s3_key)

            content = response["Body"].read().decode("utf-8")

            return content.strip()

        except Exception as e:
            raise CustomException(e, sys)
