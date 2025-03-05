from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np

app = FastAPI()
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Load dataset
df = pd.read_csv("balanced_dataset.csv")

# Ensure column names are stripped of whitespace
df.columns = df.columns.str.strip()

# Sensor definitions
sensor_definitions = {
    "Sensor_1": "Measures exhaled breath temperature.",
    "Sensor_2": "Detects oxygen levels in breath.",
    "Sensor_3": "Measures carbon dioxide concentration.",
    "Sensor_4": "Detects nitrogen compounds in breath.",
    "Sensor_5": "Monitors volatile organic compounds (VOCs).",
    "Sensor_6": "Detects methane and other hydrocarbon gases.",
    "Sensor_7": "Measures exhaled breath humidity.",
    "Sensor_8": "Detects sulfur compounds indicating lung function."
}

# Sensor boundaries based on established thresholds
sensor_boundaries = {
    "Sensor_1": {"bad": (0, 35), "moderate": (35, 37), "healthy": (37, 42)},
    "Sensor_2": {"bad": (0, 18), "moderate": (18, 20), "healthy": (20, 25)},
    "Sensor_3": {"bad": (1000, 5000), "moderate": (400, 1000), "healthy": (0, 400)},
    "Sensor_4": {"bad": (500, 1000), "moderate": (200, 500), "healthy": (0, 200)},
    "Sensor_5": {"bad": (500, 1000), "moderate": (100, 500), "healthy": (0, 100)},
    "Sensor_6": {"bad": (5000, 10000), "moderate": (1000, 5000), "healthy": (0, 1000)},
    "Sensor_7": {"bad": (100, 200), "moderate": (50, 100), "healthy": (0, 50)},
    "Sensor_8": {"bad": (300, 600), "moderate": (50, 300), "healthy": (0, 50)}
}

def categorize_sensor_values(sensor_data):
    categorized_data = {}
    for sensor, value in sensor_data.items():
        for category, (low, high) in sensor_boundaries.get(sensor, {}).items():
            if low <= value < high:
                categorized_data[sensor] = category
                break
        else:
            categorized_data[sensor] = "unknown"
    return categorized_data

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
    
    # Ensure sensor columns exist
    sensor_columns = [col for col in df.columns if "sensor_" in col]
    if len(sensor_columns) != 8:
        raise HTTPException(status_code=500, detail=f"Expected 8 sensor columns, found {len(sensor_columns)}: {sensor_columns}")
    
    # Compute Euclidean distance
    sensor_values = np.array(request.sensors)
    df["distance"] = df[sensor_columns].apply(lambda row: np.linalg.norm(row.values - sensor_values), axis=1)
    
    # Get top 5 closest matches
    similar_rows = df.nsmallest(5, "distance")[sensor_columns + ["label"]]
    
    if similar_rows.empty:
        return {
            "message": "No similar cases found in our system. Please consult a qualified doctor for further assistance."
        }
    
    # Predict label by selecting the most common label among closest matches
    predicted_label = similar_rows["label"].mode()[0] if not similar_rows["label"].empty else None
    
    categorized_sensors = categorize_sensor_values(dict(zip(sensor_columns, request.sensors)))
    
    # Generate AI health assistant response
    analysis_prompt = f"""
    You are Meditron, a virtual health assistant. Your primary role is to assist both patients and doctors by providing:
    
    1. **Personalized health advice** based on health indicators such as sensor data or symptoms.
    2. **Clear and understandable explanations** of possible causes of health conditions.
    3. **Lifestyle improvement suggestions** like diet, exercise, sleep habits, and stress management.
    4. **Limitations and referrals**: If you are unsure about the diagnosis or the query, you will encourage the user to consult a healthcare professional.
    
    Here is the sensor analysis and predicted health label:
    
    Predicted label: {predicted_label}
    Sensor analysis: {categorized_sensors}
    
    Please provide insights based on this information.
    """
    
    ai_response = generate_text(PromptRequest(prompt=analysis_prompt))
    
    return {
        "similar_matches": similar_rows.to_dict(orient="records"),
        "predicted_label": predicted_label,
        "sensor_analysis": categorized_sensors,
        "ai_health_assistant_response": ai_response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
