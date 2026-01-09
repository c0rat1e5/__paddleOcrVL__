"""
PaddleOCR-VL Gradio Application
"""

import os
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Configuration - ローカルモデルパスを使用
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "PaddleOCR-VL")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

PROMPTS = {
    "OCR": "OCR:",
    "Formula": "Formula Recognition:",
    "Table": "Table Recognition:",
    "Chart": "Chart Recognition:",
}

# Global model
model = None
processor = None


def load_model():
    global model, processor
    if model is None:
        print(f"Loading model on {DEVICE}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True, torch_dtype=DTYPE
        ).to(DEVICE).eval()
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("Model loaded!")
    return model, processor


def recognize(image, task):
    if image is None:
        return "Please upload an image."
    
    m, p = load_model()
    
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPTS[task]},
    ]}]
    
    inputs = p.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = m.generate(**inputs, max_new_tokens=4096)
    
    result = p.batch_decode(outputs, skip_special_tokens=True)[0]
    if "assistant" in result.lower():
        result = result.split("assistant")[-1].strip()
    
    return result


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 PaddleOCR-VL Demo")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="pil")
            task_select = gr.Radio(
                choices=["OCR", "Formula", "Table", "Chart"],
                value="OCR", label="Recognition Type"
            )
            btn = gr.Button("Recognize", variant="primary")
        
        with gr.Column():
            output = gr.Markdown(label="Result")
    
    btn.click(fn=recognize, inputs=[image_input, task_select], outputs=output)

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    demo.launch(server_name="0.0.0.0", server_port=7860)
