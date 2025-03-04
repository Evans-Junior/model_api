from fastapi import FastAPI
from classifier import classify_input
from meditron import run_meditron_health_assistant
from sensor_info import interpret_sensor_data
import time
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ensure Ollama is running on this port

def call_ollama(prompt):
    """Sends a request to the Ollama model and correctly handles streamed responses."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": "meditron", "prompt": prompt, "stream": False}  # Force non-streaming mode
    )

    print("Ollama Raw Response:", response.text)  # Debugging

    if response.status_code == 200:
        try:
            json_data = response.json()
            
            # If it's a list of responses, concatenate them
            if isinstance(json_data, list):
                return " ".join([item.get("response", "") for item in json_data])
            else:
                return json_data.get("response", "No response from model")
        
        except ValueError as e:
            return f"JSON Decode Error: {e}, Response Text: {response.text}"
    else:
        return f"Error: {response.status_code}, {response.text}"


@app.post("/predict")
async def predict_health(sensor_data: dict):
    try:
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
        medical_advice = call_ollama(final_prompt)

        processing_time = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "processing_time": f"{processing_time} seconds",
            "classification": classification,
            "sensor_interpretations": sensor_interpretations,
            "similar_cases": similar_cases,
            "medical_advice": medical_advice
        }
    except Exception as e:
            return {"status": "error", "message": str(e)}
