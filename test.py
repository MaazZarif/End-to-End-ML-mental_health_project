import sys
import pandas as pd

from src.mental_health_project.pipelines.prediction_pipeline import PredictionPipeline
from src.mental_health_project.exception import CustomException


try:

    features = pd.DataFrame([
        {
            "Age": 21,
            "Gender": "Male",
            "Academic_Level": "Undergraduate",
            "Avg_Daily_Usage_Hours": 5.5,
            "Daily_Unlocks": 40,
            "Most_Used_Platform": "Instagram",
            "Time_Spent_on_Social_Media": 3.5,
            "Number_of_Social_Media_Apps": 4,
            "Country": "Pakistan",
            "Study_Hours": 5,
            "Physical_Activity_Hours": 1.5,
            "Sleep_Hours_Per_Night": 7,
            "Stress_Level": "Medium",
            "Purpose_Of_Use": "Entertainment"
        }
    ])

    prediction_pipeline = PredictionPipeline()

    prediction = prediction_pipeline.predict(features)

    print("Prediction:", prediction)

except Exception as e:
    raise CustomException(e, sys)