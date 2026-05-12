import gradio as gr
import torch
from app.inference import load_model, run_prediction
import os
import io

# 1. Model Load Karein
MODEL_PATH = "model/trained_model.pth"
model = load_model(MODEL_PATH)

def predict_ui(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()
    results = run_prediction(model, image_bytes)
    
    return {res.label: float(res.confidence) for res in results}

# 2. Interface Banayein
demo = gr.Interface(
    fn=predict_ui,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Pakistani Politician Classifier",
    description="Photo upload karein aur AI batayega ke ye kaunsa politician hai."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)