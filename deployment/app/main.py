import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.inference import load_model, run_prediction

app = FastAPI(title="Politician Classifier")

# Global variable to hold our model so it stays in memory
MODEL = None
MODEL_PATH = "model/trained_model.pth"

@app.on_event("startup")
def startup_event():
    """Runs when the server starts. Loads the model into memory."""
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH)
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}")

@app.get("/")
def health_check():
    return {"status": "ready", "model_loaded": MODEL is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives an image file and returns the AI's guess."""
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    image_data = await file.read()
    
    try:
        results = run_prediction(MODEL, image_data)
        
        return {
            "prediction": results[0].label,
            "confidence": results[0].confidence,
            "all_results": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)