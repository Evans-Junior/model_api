from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Define request model
class PromptRequest(BaseModel):
    prompt: str

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
