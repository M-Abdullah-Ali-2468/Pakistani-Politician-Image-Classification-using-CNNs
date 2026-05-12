import torch
import sys
import traceback
from dataclasses import dataclass
from app.utils import CLASS_NAMES, prepare_image

@dataclass
class Prediction:
    label: str
    confidence: float

def load_model(model_path: str):
    """Loads the AI brain and sets it to evaluation mode."""
    # Intentionally load the full checkpoint (not weights-only), using the
    # user's trained model. If loading fails, print the error and terminate
    # immediately so the app does not continue with an invalid model.
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print("Model load error:", str(e))
        traceback.print_exc()
        sys.exit(1)

    model.eval()
    return model

def run_prediction(model, image_bytes: bytes) -> list[Prediction]:
    """Takes an image, runs it through the model, and returns named results."""
    # 1. Prepare image
    batch = prepare_image(image_bytes)

    # 2. Predict (without saving math for training)
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits[0], dim=0)

    # 3. Get Top 3 results
    top_prob, top_idx = torch.topk(probabilities, k=3)
    
    predictions = []
    for score, idx in zip(top_prob, top_idx):
        predictions.append(Prediction(
            label=CLASS_NAMES[idx.item()],
            confidence=round(score.item(), 4)
        ))
    
    return predictions