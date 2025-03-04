from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np

app = FastAPI()
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Load dataset
df = pd.read_csv("balanced_dataset.csv")

# Define request models
class PromptRequest(BaseModel):
    prompt: str

class SensorRequest(BaseModel):
    sensors: list[float]

@app.post("/generate")
def generate_text(request: PromptRequest):
    payload = {
        "model": "meditron:7b",
        "prompt": request.prompt,
        "stream": False
    }
    
    response = requests.post(OLLAMA_API_URL, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    return response.json()

@app.post("/find_similar")
def find_similar(request: SensorRequest):
    if len(request.sensors) != 8:
        raise HTTPException(status_code=400, detail="Exactly 8 sensor values are required")
    
    # Compute Euclidean distance
    sensor_values = np.array(request.sensors)
    df["distance"] = df.iloc[:, :-1].apply(lambda row: np.linalg.norm(row.values - sensor_values), axis=1)
    
    # Get top 5 closest matches
    similar_rows = df.nsmallest(5, "distance")[["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_5", "Sensor_6", "Sensor_7", "Sensor_8", "label"]]
    
    # Predict label by selecting the most common label among closest matches
    predicted_label = similar_rows["label"].mode()[0] if not similar_rows["label"].empty else None
    
    return {"similar_matches": similar_rows.to_dict(orient="records"), "predicted_label": predicted_label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
