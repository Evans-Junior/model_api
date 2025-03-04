from fastapi import FastAPI
from classifier import classify_input
from meditron import run_meditron_health_assistant
from sensor_info import interpret_sensor_data
import time

app = FastAPI()

@app.post("/predict/")
async def predict_health(request_data: dict):
    try:
        sensor_data = request_data["sensor_data"]

        # Call classify_input function
        classification, similar_cases = classify_input(sensor_data)

        # Handle errors properly
        if classification == "error":
            return {"status": "error", "message": similar_cases}

        return {
            "status": "success",
            "classification": classification,
            "similar_cases": similar_cases
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
