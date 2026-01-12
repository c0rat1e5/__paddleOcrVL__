"""
PaddleOCR-VL Batch Folder Processing
フォルダ内の全画像を一括処理
"""

import os
import tempfile
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import json
from datetime import datetime

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "PaddleOCR-VL")
LAYOUT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "PP-DocLayoutV2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

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
        vlm_processor = AutoProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
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
            vl_rec_model_dir=MODEL_PATH,
        )
        print("Document Parser loaded!")
    return doc_parser


def get_image_files(folder_path):
    """Get all image files from a folder"""
    folder = Path(folder_path)
    if not folder.exists():
        return []
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))
    
    return sorted(set(image_files))


def process_single_image_doc(image_path, parser):
    """Process a single image with document parser"""
    try:
        output = parser.predict(str(image_path))
        
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
        
        return markdown_text.strip() if markdown_text.strip() else "No content recognized."
    
    except Exception as e:
        return f"Error: {str(e)}"


def process_single_image_elem(image_path, task, model, processor):
    """Process a single image with element recognition"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPTS[task]},
                ],
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=4096)
        
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        if "assistant" in result.lower():
            result = result.split("assistant")[-1].strip()
        
        return result
    
    except Exception as e:
        return f"Error: {str(e)}"


def batch_process_folder(folder_path, mode, task, save_results):
    """Batch process all images in a folder"""
    
    if not folder_path or not folder_path.strip():
        yield "❌ フォルダパスを入力してください", "", None
        return
    
    folder_path = folder_path.strip()
    
    if not os.path.exists(folder_path):
        yield f"❌ フォルダが見つかりません: {folder_path}", "", None
        return
    
    if not os.path.isdir(folder_path):
        yield f"❌ ディレクトリではありません: {folder_path}", "", None
        return
    
    # Get image files
    image_files = get_image_files(folder_path)
    
    if not image_files:
        yield f"❌ 画像ファイルが見つかりません: {folder_path}", "", None
        return
    
    total = len(image_files)
    yield f"🔍 **{total}個の画像を検出しました。処理を開始します...**", "", None
    
    # Load model
    if mode == "Document Parsing":
        yield f"🔄 **Document Parserを読み込み中...**", "", None
        parser = load_doc_parser()
    else:
        yield f"🔄 **VLMモデルを読み込み中...**", "", None
        model, processor = load_vlm()
    
    results = {}
    all_output = ""
    
    for i, image_path in enumerate(image_files):
        filename = image_path.name
        progress_msg = f"⏳ **処理中: {i+1}/{total}** - `{filename}`"
        yield progress_msg, all_output, None
        
        print(f"Processing [{i+1}/{total}]: {filename}")
        
        if mode == "Document Parsing":
            result = process_single_image_doc(image_path, parser)
        else:
            result = process_single_image_elem(image_path, task, model, processor)
        
        results[filename] = result
        
        # Build cumulative output
        all_output += f"## 📄 {filename}\n\n{result}\n\n---\n\n"
        
        yield progress_msg, all_output, None
    
    # Save results if requested
    output_file = None
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(folder_path) / "ocr_results"
        output_dir.mkdir(exist_ok=True)
        
        # Save as Markdown
        md_file = output_dir / f"results_{timestamp}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# OCR Results\n\n")
            f.write(f"**Folder:** {folder_path}\n\n")
            f.write(f"**Mode:** {mode}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Images:** {total}\n\n")
            f.write("---\n\n")
            f.write(all_output)
        
        # Save as JSON
        json_file = output_dir / f"results_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({
                "folder": folder_path,
                "mode": mode,
                "task": task if mode == "Element Recognition" else None,
                "timestamp": timestamp,
                "total_images": total,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        output_file = str(md_file)
        yield f"✅ **完了! {total}個の画像を処理しました**\n\n📁 結果保存先: `{output_dir}`", all_output, output_file
    else:
        yield f"✅ **完了! {total}個の画像を処理しました**", all_output, None


def batch_process_uploaded_files(files, mode, task):
    """Batch process uploaded folder/files"""
    
    if not files:
        yield "❌ フォルダをアップロードしてください", "", None
        return
    
    # Handle both single file path and list of files
    if isinstance(files, str):
        # Single directory path
        files = [files]
    
    # Filter to only image files
    image_files = []
    for f in files:
        # Handle different input formats
        if hasattr(f, 'name'):
            file_path = f.name
        else:
            file_path = str(f)
        
        ext = Path(file_path).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            image_files.append(file_path)
    
    if not image_files:
        yield "❌ 有効な画像ファイルがありません", "", None
        return
    
    total = len(image_files)
    yield f"🔍 **{total}個の画像を検出しました。処理を開始します...**", "", None
    
    # Load model
    if mode == "Document Parsing":
        yield f"🔄 **Document Parserを読み込み中...**", "", None
        parser = load_doc_parser()
    else:
        yield f"🔄 **VLMモデルを読み込み中...**", "", None
        model, processor = load_vlm()
    
    results = {}
    all_output = ""
    
    for i, file_path in enumerate(image_files):
        filename = Path(file_path).name
        progress_msg = f"⏳ **処理中: {i+1}/{total}** - `{filename}`"
        yield progress_msg, all_output, None
        
        print(f"Processing [{i+1}/{total}]: {filename}")
        
        if mode == "Document Parsing":
            result = process_single_image_doc(file_path, parser)
        else:
            result = process_single_image_elem(file_path, task, model, processor)
        
        results[filename] = result
        
        # Build cumulative output
        all_output += f"## 📄 {filename}\n\n{result}\n\n---\n\n"
        
        yield progress_msg, all_output, None
    
    # Save results to temp file for download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create temp directory for results
    temp_dir = tempfile.mkdtemp()
    
    # Save as Markdown
    md_file = Path(temp_dir) / f"ocr_results_{timestamp}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# OCR Results\n\n")
        f.write(f"**Mode:** {mode}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Images:** {total}\n\n")
        f.write("---\n\n")
        f.write(all_output)
    
    # Save as JSON
    json_file = Path(temp_dir) / f"ocr_results_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "task": task if mode == "Element Recognition" else None,
            "timestamp": timestamp,
            "total_images": total,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    yield f"✅ **完了! {total}個の画像を処理しました**", all_output, str(md_file)


# Gradio UI
with gr.Blocks(title="PaddleOCR-VL Batch Processing") as demo:
    gr.Markdown("# 📁 PaddleOCR-VL Batch Folder Processing")
    gr.Markdown("複数の画像を一括でOCR処理します")
    
    with gr.Tabs():
        # Upload Files Tab
        with gr.Tab("� フォルダアップロード"):
            gr.Markdown("フォルダをドラッグ&ドロップまたは選択してアップロード")
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="📂 フォルダをアップロード",
                        file_count="directory"
                    )
                    
                    upload_mode_select = gr.Radio(
                        choices=["Document Parsing", "Element Recognition"],
                        value="Document Parsing",
                        label="処理モード",
                        info="Document Parsing: レイアウト検出あり / Element Recognition: 単純OCR"
                    )
                    
                    upload_task_select = gr.Radio(
                        choices=["OCR", "Formula", "Table", "Chart"],
                        value="OCR",
                        label="認識タイプ (Element Recognitionの場合)",
                        visible=True
                    )
                    
                    upload_status = gr.Markdown("", elem_id="upload_status")
                    
                    upload_btn = gr.Button("🚀 一括処理開始", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("📝 結果プレビュー"):
                            upload_preview = gr.Markdown(label="Results")
                        with gr.Tab("📥 ダウンロード"):
                            upload_download = gr.File(label="結果ファイル")
            
            # Show/hide task selector based on mode
            def update_upload_task_visibility(mode):
                return gr.update(visible=(mode == "Element Recognition"))
            
            upload_mode_select.change(
                fn=update_upload_task_visibility,
                inputs=[upload_mode_select],
                outputs=[upload_task_select]
            )
            
            # Process button click
            upload_btn.click(
                fn=batch_process_uploaded_files,
                inputs=[file_upload, upload_mode_select, upload_task_select],
                outputs=[upload_status, upload_preview, upload_download]
            )
        
        # Server Folder Tab
        with gr.Tab("📂 サーバーフォルダ指定"):
            gr.Markdown("サーバー上のフォルダパスを直接指定（サーバーに直接アクセスできる場合）")
            
            with gr.Row():
                with gr.Column(scale=1):
                    folder_input = gr.Textbox(
                        label="📂 フォルダパス（サーバー上のパス）",
                        placeholder="/home/user/images",
                        info="サーバー上の画像フォルダのパスを入力"
                    )
                    
                    mode_select = gr.Radio(
                        choices=["Document Parsing", "Element Recognition"],
                        value="Document Parsing",
                        label="処理モード",
                        info="Document Parsing: レイアウト検出あり / Element Recognition: 単純OCR"
                    )
                    
                    task_select = gr.Radio(
                        choices=["OCR", "Formula", "Table", "Chart"],
                        value="OCR",
                        label="認識タイプ (Element Recognitionの場合)",
                        visible=True
                    )
                    
                    save_checkbox = gr.Checkbox(
                        label="📥 結果をファイルに保存",
                        value=True,
                        info="処理結果をMarkdownとJSONで保存します"
                    )
                    
                    status_display = gr.Markdown("", elem_id="status")
                    
                    process_btn = gr.Button("🚀 一括処理開始", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("📝 結果プレビュー"):
                            result_preview = gr.Markdown(label="Results")
                        with gr.Tab("📥 ダウンロード"):
                            download_file = gr.File(label="結果ファイル")
            
            # Show/hide task selector based on mode
            def update_task_visibility(mode):
                return gr.update(visible=(mode == "Element Recognition"))
            
            mode_select.change(
                fn=update_task_visibility,
                inputs=[mode_select],
                outputs=[task_select]
            )
            
            # Process button click
            process_btn.click(
                fn=batch_process_folder,
                inputs=[folder_input, mode_select, task_select, save_checkbox],
                outputs=[status_display, result_preview, download_file]
            )
    
    # Examples
    gr.Markdown("---")
    gr.Markdown("### 💡 使い方")
    gr.Markdown("""
**� フォルダアップロード（推奨）:**
1. 「フォルダをアップロード」をクリック
2. 画像が入っているフォルダを選択
3. 処理モードを選択
4. 「一括処理開始」ボタンをクリック

**📂 サーバーフォルダ指定:**
- サーバーに直接アクセスできる場合のみ使用
- サーバー上のLinuxパスを入力（例: `/home/user/images`）

📌 **対応フォーマット**: PNG, JPG, JPEG, BMP, TIFF, WebP, GIF
    """)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("Loading models on first use...")
    demo.launch(server_name="0.0.0.0", server_port=7861)
