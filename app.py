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
        vlm_model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, trust_remote_code=True, torch_dtype=DTYPE
            )
            .to(DEVICE)
            .eval()
        )
        # print(vlm_model)
        # → PaddleOCRVLForConditionalGeneration(
        #     (vision_model): NavitSigLIPModel(...)
        #     (language_model): Ernie4_5MoeForCausalLM(...)
        #   )

        vlm_processor = AutoProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
        # print(vlm_processor)
        # → PaddleOCRVLProcessor(
        #     image_processor=PaddleOCRVLImageProcessor,
        #     tokenizer=PreTrainedTokenizerFast
        #   )
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
            # → "/home/.../models/PP-DocLayoutV2"
            vl_rec_model_dir=MODEL_PATH,
            # → "/home/.../models/PaddleOCR-VL"
        )
        # print(doc_parser)
        # → PaddleOCRVL(
        #     layout_detection_model=PP-DocLayoutV2,
        #     vl_rec_model=PaddleOCR-VL-0.9B
        #   )
        print("Document Parser loaded!")
    return doc_parser


def element_recognize(image, task, progress=gr.Progress()):
    """Element-level recognition using VLM"""
    # print(image)
    # → <PIL.Image.Image mode=RGB size=800x600>

    # print(task)
    # → "OCR" | "Formula" | "Table" | "Chart"

    if image is None:
        return "Please upload an image."

    progress(0.1, desc="Loading model...")
    m, p = load_vlm()
    # print(m)
    # → PaddleOCRVLForConditionalGeneration (VLMモデル本体)
    # print(p)
    # → PaddleOCRVLProcessor (トークナイザー+画像処理)

    progress(0.3, desc="Preparing image...")
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPTS[task]},
            ],
        }
    ]
    # print(messages)
    # → [{"role": "user", "content": [
    #       {"type": "image", "image": <PIL.Image>},
    #       {"type": "text", "text": "OCR:"}
    #    ]}]

    progress(0.5, desc="Processing...")
    inputs = p.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)
    # print(inputs.keys())
    # → dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
    # print(inputs['input_ids'].shape)
    # → torch.Size([1, 256])  # トークン化されたテキスト
    # print(inputs['pixel_values'].shape)
    # → torch.Size([1, 3, 896, 896])  # 画像テンソル

    progress(0.7, desc="Generating output...")
    with torch.no_grad():
        outputs = m.generate(**inputs, max_new_tokens=4096)
    # print(outputs.shape)
    # → torch.Size([1, 512])  # 生成されたトークンID

    progress(0.9, desc="Decoding...")
    result = p.batch_decode(outputs, skip_special_tokens=True)[0]
    # print(result)
    # → "user\n[画像]\nOCR:\nassistant\nHMM INTERESTING,\nTHESE MODELS..."

    if "assistant" in result.lower():
        result = result.split("assistant")[-1].strip()
    # print(result)
    # → "HMM INTERESTING,\nTHESE MODELS...\nMAYBE I CAN BUY ONE..."

    progress(1.0, desc="Done!")
    return result


def document_parse(image, progress=gr.Progress()):
    """Document parsing with layout detection"""
    # print(image)
    # → <PIL.Image.Image mode=RGB size=1200x1600>

    if image is None:
        return "Please upload an image.", ""

    # Save image to temp file
    temp_path = None
    try:
        progress(0.1, desc="Preparing image...")
        if isinstance(image, Image.Image):
            temp_path = tempfile.mktemp(suffix=".png")
            image.save(temp_path)
        elif isinstance(image, str):
            temp_path = image
        else:
            temp_path = tempfile.mktemp(suffix=".png")
            Image.fromarray(image).save(temp_path)
        # print(temp_path)
        # → "/tmp/tmpxyz123.png"

        progress(0.2, desc="Loading Document Parser...")
        parser = load_doc_parser()

        progress(0.4, desc="Detecting layout...")
        print("Starting document parsing...")
        output = parser.predict(temp_path)
        # print(output)
        # → [DocParserResult(
        #       markdown={'text': '# Title\n\nParagraph...', 'images': {}},
        #       layout_results=[{bbox: [...], label: 'text'}, ...]
        #    )]
        print("Parsing complete!")

        progress(0.8, desc="Extracting text...")

        markdown_text = ""
        for res in output:
            # print(res)
            # → DocParserResult object または dict

            # Handle different output formats
            if hasattr(res, "markdown"):
                md = res.markdown
                # print(md)
                # → {'text': '# Title\n\n本文テキスト...', 'images': {}}
                if isinstance(md, dict):
                    markdown_text += md.get("text", str(md)) + "\n\n"
                else:
                    markdown_text += str(md) + "\n\n"
            elif hasattr(res, "text"):
                txt = res.text
                if isinstance(txt, dict):
                    markdown_text += txt.get("text", str(txt)) + "\n\n"
                else:
                    markdown_text += str(txt) + "\n\n"
            elif isinstance(res, dict):
                if "markdown" in res:
                    md = res["markdown"]
                    if isinstance(md, dict):
                        markdown_text += md.get("text", str(md)) + "\n\n"
                    else:
                        markdown_text += str(md) + "\n\n"
                elif "text" in res:
                    markdown_text += str(res["text"]) + "\n\n"

        # print(markdown_text[:200])
        # → "# Document Title\n\nThis is the first paragraph...\n\n| Col1 | Col2 |..."

        if not markdown_text.strip():
            markdown_text = "No content recognized."

        progress(1.0, desc="Done!")
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
    gr.Markdown(
        "**Document Parsing** with layout detection or **Element-level Recognition** for single elements"
    )

    with gr.Tabs():
        # Document Parsing Tab
        with gr.Tab("📄 Document Parsing"):
            gr.Markdown(
                "Upload a full document page for automatic layout detection and parsing."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    doc_image = gr.Image(label="Upload Document", type="pil")
                    doc_status = gr.Markdown("", elem_id="doc_status")
                    doc_btn = gr.Button("📄 Parse Document", variant="primary")

                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab("Markdown Preview"):
                            doc_preview = gr.Markdown(label="Result")
                        with gr.Tab("Raw Output"):
                            doc_raw = gr.Code(
                                label="Markdown Source", language="markdown"
                            )

            def doc_parse_with_status(image):
                yield "⏳ **Processing...**", "", ""
                if image is None:
                    yield "❌ Please upload an image.", "Please upload an image.", ""
                    return

                try:
                    import tempfile

                    temp_path = None
                    if isinstance(image, Image.Image):
                        temp_path = tempfile.mktemp(suffix=".png")
                        image.save(temp_path)
                    elif isinstance(image, str):
                        temp_path = image
                    else:
                        temp_path = tempfile.mktemp(suffix=".png")
                        Image.fromarray(image).save(temp_path)

                    yield "🔄 **Loading models...**", "", ""
                    parser = load_doc_parser()

                    yield "🔍 **Detecting layout & recognizing...**", "", ""
                    print("Starting document parsing...")
                    output = parser.predict(temp_path)
                    print("Parsing complete!")

                    markdown_text = ""
                    for res in output:
                        if hasattr(res, "markdown"):
                            md = res.markdown
                            if isinstance(md, dict):
                                markdown_text += md.get("text", str(md)) + "\n\n"
                            else:
                                markdown_text += str(md) + "\n\n"
                        elif hasattr(res, "text"):
                            txt = res.text
                            if isinstance(txt, dict):
                                markdown_text += txt.get("text", str(txt)) + "\n\n"
                            else:
                                markdown_text += str(txt) + "\n\n"
                        elif isinstance(res, dict):
                            if "markdown" in res:
                                md = res["markdown"]
                                if isinstance(md, dict):
                                    markdown_text += md.get("text", str(md)) + "\n\n"
                                else:
                                    markdown_text += str(md) + "\n\n"
                            elif "text" in res:
                                markdown_text += str(res["text"]) + "\n\n"

                    if not markdown_text.strip():
                        markdown_text = "No content recognized."

                    if temp_path and temp_path != image and os.path.exists(temp_path):
                        os.remove(temp_path)

                    yield "✅ **Done!**", markdown_text.strip(), markdown_text.strip()
                    return

                except Exception as e:
                    yield f"❌ **Error:** {str(e)}", f"Error: {str(e)}", ""
                    return

            doc_btn.click(
                fn=doc_parse_with_status,
                inputs=[doc_image],
                outputs=[doc_status, doc_preview, doc_raw],
            ).then(
                fn=lambda: gr.update(value="📄 Parse Document", interactive=True),
                outputs=[doc_btn],
            )

            # Disable button while processing
            doc_btn.click(
                fn=lambda: gr.update(value="⏳ Processing...", interactive=False),
                outputs=[doc_btn],
                queue=False,
            )

        # Element Recognition Tab
        with gr.Tab("🔤 Element Recognition"):
            gr.Markdown(
                "Upload a cropped element (text, formula, table, or chart) for recognition."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    elem_image = gr.Image(label="Upload Element", type="pil")
                    task_select = gr.Radio(
                        choices=["OCR", "Formula", "Table", "Chart"],
                        value="OCR",
                        label="Recognition Type",
                    )
                    elem_status = gr.Markdown("", elem_id="elem_status")
                    elem_btn = gr.Button("🔤 Recognize", variant="primary")

                with gr.Column(scale=1):
                    elem_output = gr.Markdown(label="Result")

            def elem_recognize_with_status(image, task):
                yield "⏳ **Processing...**", ""
                if image is None:
                    yield "❌ Please upload an image.", "Please upload an image."
                    return

                try:
                    yield "🔄 **Loading model...**", ""
                    m, p = load_vlm()

                    yield "🖼️ **Preparing image...**", ""
                    if isinstance(image, str):
                        image = Image.open(image).convert("RGB")
                    elif not isinstance(image, Image.Image):
                        image = Image.fromarray(image).convert("RGB")

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": PROMPTS[task]},
                            ],
                        }
                    ]

                    yield "🔍 **Recognizing...**", ""
                    inputs = p.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(DEVICE)

                    with torch.no_grad():
                        outputs = m.generate(**inputs, max_new_tokens=4096)

                    result = p.batch_decode(outputs, skip_special_tokens=True)[0]
                    if "assistant" in result.lower():
                        result = result.split("assistant")[-1].strip()

                    yield "✅ **Done!**", result
                    return

                except Exception as e:
                    yield f"❌ **Error:** {str(e)}", f"Error: {str(e)}"
                    return

            elem_btn.click(
                fn=elem_recognize_with_status,
                inputs=[elem_image, task_select],
                outputs=[elem_status, elem_output],
            ).then(
                fn=lambda: gr.update(value="🔤 Recognize", interactive=True),
                outputs=[elem_btn],
            )

            # Disable button while processing
            elem_btn.click(
                fn=lambda: gr.update(value="⏳ Processing...", interactive=False),
                outputs=[elem_btn],
                queue=False,
            )

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("Loading models on first use...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
