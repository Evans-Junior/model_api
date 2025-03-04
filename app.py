from fastapi import FastAPI
from classifier import classify_input
from meditron import run_meditron_health_assistant
from sensor_info import interpret_sensor_data
import time

app = FastAPI()

@app.post("/predict")
def predict_health(sensor_data: dict):
    start_time = time.time()

    # Step 1: Classify input using RAG
    classification, similar_cases = classify_input(sensor_data)

    # Step 2: Interpret sensor data
    sensor_interpretations = interpret_sensor_data(sensor_data)

    # Step 3: Prepare final prompt for Meditron
    final_prompt = f"""
    Based on similar cases from the dataset, this case is classified as: {classification}.
    Sensor Data Analysis:
    {sensor_interpretations}
    Medical AI Advice Needed.
    """

    # Step 4: Get Meditron Model Advice
    medical_advice = run_meditron_health_assistant(final_prompt)

    processing_time = round(time.time() - start_time, 2)

    return {
        "status": "success",
        "processing_time": f"{processing_time} seconds",
        "classification": classification,
        "sensor_interpretations": sensor_interpretations,
        "similar_cases": similar_cases,
        "medical_advice": medical_advice
    }
