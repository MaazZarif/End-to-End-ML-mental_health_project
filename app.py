import sys
import pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field
from src.mental_health_project.pipelines.prediction_pipeline import PredictionPipeline
from src.mental_health_project.exception import CustomException

app = FastAPI(
    title="Student Mental Health Prediction API", 
    description="API for predicting student mental health score", 
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    Age:int = Field(...,ge=1)
    Gender:str
    Academic_Level:str
    Avg_Daily_Usage_Hours: float = Field(..., ge=0) 
    Daily_Unlocks: int = Field(..., ge=0) 
    Most_Used_Platform: str 
    Time_Spent_on_Social_Media: float = Field(..., ge=0) 
    Number_of_Social_Media_Apps: int = Field(..., ge=0) 
    Country: str 
    Study_Hours: float = Field(..., ge=0) 
    Physical_Activity_Hours: float = Field(..., ge=0) 
    Sleep_Hours_Per_Night: float = Field(..., ge=0) 
    Stress_Level: str 
    Purpose_Of_Use: str



class PredictionResponse(BaseModel):
    prediction : float



@app.get("/")
def home():
    return {
        "message": "Student Mental Health Prediction API is running"
    }

@app.get("/health")
def health():
    return{
        "message":"Status healthy"
    }


prediction_pipeline = PredictionPipeline()

@app.post("/predict",response_model=PredictionResponse)
def predict(request: PredictionRequest): 
    try: 
        features = pd.DataFrame([ request.model_dump() ]) 
        prediction = prediction_pipeline.predict(features) 
        prediction_value = float(prediction[0]) 
        return PredictionResponse( prediction=prediction_value ) 
    except Exception as e: 
        raise HTTPException( status_code=500, detail=str(e) )
