from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()
OLLAMA_API_URL = "http://localhost:11434/api/generate"

@app.post("/generate")
def generate_text(prompt: str):
    payload = {
        "model": "meditron:7b",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(OLLAMA_API_URL, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    return response.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
