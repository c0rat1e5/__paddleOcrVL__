"""
PaddleOCR-VL Gradio Application
Document Parsing + Element-level Recognition
"""

import os
import tempfile
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "PaddleOCR-VL")
LAYOUT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "PP-DocLayoutV2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

PROMPTS = {
    "OCR": "OCR:",
    "Formula": "Formula Recognition:",
    "Table": "Table Recognition:",
    "Chart": "Chart Recognition:",
}

# Global models
vlm_model = None
vlm_processor = None
doc_parser = None


def load_vlm():
    """Load VLM model for element-level recognition"""
    global vlm_model, vlm_processor
    if vlm_model is None:
        print(f"Loading VLM on {DEVICE}...")
        vlm_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True, torch_dtype=DTYPE
        ).to(DEVICE).eval()
        vlm_processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("VLM loaded!")
    return vlm_model, vlm_processor


def load_doc_parser():
    """Load PaddleOCR Document Parser with local models"""
    global doc_parser
    if doc_parser is None:
        print("Loading Document Parser from local models...")
        os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCRVL
        doc_parser = PaddleOCRVL(
            layout_detection_model_dir=LAYOUT_MODEL_PATH,
            vl_rec_model_dir=MODEL_PATH
        )
        print("Document Parser loaded!")
    return doc_parser


def element_recognize(image, task):
    """Element-level recognition using VLM"""
    if image is None:
        return "Please upload an image."
    
    m, p = load_vlm()
    
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


def document_parse(image):
    """Document parsing with layout detection"""
    if image is None:
        return "Please upload an image.", ""
    
    # Save image to temp file
    temp_path = None
    try:
        if isinstance(image, Image.Image):
            temp_path = tempfile.mktemp(suffix=".png")
            image.save(temp_path)
        elif isinstance(image, str):
            temp_path = image
        else:
            temp_path = tempfile.mktemp(suffix=".png")
            Image.fromarray(image).save(temp_path)
        
        parser = load_doc_parser()
        output = parser.predict(temp_path)
        
        markdown_text = ""
        for res in output:
            # Handle different output formats
            if hasattr(res, 'markdown'):
                md = res.markdown
                if isinstance(md, dict):
                    markdown_text += md.get('text', str(md)) + "\n\n"
                else:
                    markdown_text += str(md) + "\n\n"
            elif hasattr(res, 'text'):
                txt = res.text
                if isinstance(txt, dict):
                    markdown_text += txt.get('text', str(txt)) + "\n\n"
                else:
                    markdown_text += str(txt) + "\n\n"
            elif isinstance(res, dict):
                if 'markdown' in res:
                    md = res['markdown']
                    if isinstance(md, dict):
                        markdown_text += md.get('text', str(md)) + "\n\n"
                    else:
                        markdown_text += str(md) + "\n\n"
                elif 'text' in res:
                    markdown_text += str(res['text']) + "\n\n"
        
        if not markdown_text.strip():
            markdown_text = "No content recognized."
        
        return markdown_text.strip(), markdown_text.strip()
    
    except Exception as e:
        error_msg = f"Error during parsing: {str(e)}"
        return error_msg, error_msg
    finally:
        if temp_path and temp_path != image and os.path.exists(temp_path):
            os.remove(temp_path)


# Gradio UI
with gr.Blocks(title="PaddleOCR-VL Demo") as demo:
    gr.Markdown("# 🔍 PaddleOCR-VL Demo")
    gr.Markdown("**Document Parsing** with layout detection or **Element-level Recognition** for single elements")
    
    with gr.Tabs():
        # Document Parsing Tab
        with gr.Tab("📄 Document Parsing"):
            gr.Markdown("Upload a full document page for automatic layout detection and parsing.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    doc_image = gr.Image(label="Upload Document", type="pil")
                    doc_btn = gr.Button("Parse Document", variant="primary")
                
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab("Markdown Preview"):
                            doc_preview = gr.Markdown(label="Result")
                        with gr.Tab("Raw Output"):
                            doc_raw = gr.Code(label="Markdown Source", language="markdown")
            
            doc_btn.click(fn=document_parse, inputs=[doc_image], outputs=[doc_preview, doc_raw])
        
        # Element Recognition Tab
        with gr.Tab("🔤 Element Recognition"):
            gr.Markdown("Upload a cropped element (text, formula, table, or chart) for recognition.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    elem_image = gr.Image(label="Upload Element", type="pil")
                    task_select = gr.Radio(
                        choices=["OCR", "Formula", "Table", "Chart"],
                        value="OCR", label="Recognition Type"
                    )
                    elem_btn = gr.Button("Recognize", variant="primary")
                
                with gr.Column(scale=1):
                    elem_output = gr.Markdown(label="Result")
            
            elem_btn.click(fn=element_recognize, inputs=[elem_image, task_select], outputs=elem_output)

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("Loading models on first use...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
