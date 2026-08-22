import os
import boto3
from dotenv import load_dotenv

load_dotenv()

print("Access key loaded:", bool(os.getenv("AWS_ACCESS_KEY_ID")))
print("Secret key loaded:", bool(os.getenv("AWS_SECRET_ACCESS_KEY")))
print("Region:", os.getenv("AWS_DEFAULT_REGION"))

s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

response = s3.list_buckets()

for bucket in response["Buckets"]:
    print(bucket["Name"])