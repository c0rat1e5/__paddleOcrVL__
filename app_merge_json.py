"""
OCR結果JSONマージツール（Gradio版）
複数のJSONファイルからmarkdown_textsを抽出してまとめる
"""

import os
import json
import ast
import tempfile
import gradio as gr
from pathlib import Path
from datetime import datetime


def extract_markdown_text(result_str):
    """結果文字列からmarkdown_textsを抽出"""
    try:
        # 文字列形式のdictをパース
        if isinstance(result_str, str) and result_str.startswith("{"):
            result_dict = ast.literal_eval(result_str)
            if "markdown_texts" in result_dict:
                return result_dict["markdown_texts"], True
            else:
                return None, False
        return result_str, True
    except Exception as e:
        return f"Parse Error: {str(e)}", False


def validate_json_files(json_files):
    """JSONファイルのバリデーション"""
    errors = []
    valid_files = []
    
    for json_file in json_files:
        file_path = json_file if isinstance(json_file, str) else json_file.name
        filename = Path(file_path).name
        
        # JSONファイルかチェック
        if not filename.endswith(".json"):
            continue
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # resultsがあるかチェック
            if "results" not in data:
                errors.append(f"❌ `{filename}`: 'results'キーがありません")
                continue
            
            # 各結果にmarkdown_textsがあるかチェック
            results = data["results"]
            missing_keys = []
            
            for img_name, result in results.items():
                _, has_markdown = extract_markdown_text(result)
                if not has_markdown:
                    missing_keys.append(img_name)
            
            if missing_keys:
                errors.append(f"❌ `{filename}`: 以下の画像に'markdown_texts'がありません: {', '.join(missing_keys[:3])}{'...' if len(missing_keys) > 3 else ''}")
            else:
                valid_files.append((file_path, data))
        
        except json.JSONDecodeError as e:
            errors.append(f"❌ `{filename}`: JSONパースエラー - {str(e)}")
        except Exception as e:
            errors.append(f"❌ `{filename}`: エラー - {str(e)}")
    
    return valid_files, errors


def merge_json_files(files):
    """JSONファイルをマージしてtxtを生成"""
    
    if not files:
        yield "❌ JSONファイルをアップロードしてください", "", None
        return
    
    # ファイルリストを取得
    file_list = files if isinstance(files, list) else [files]
    
    # JSONファイルのみフィルタ
    json_files = []
    for f in file_list:
        file_path = f if isinstance(f, str) else f.name
        if file_path.endswith(".json"):
            json_files.append(file_path)
    
    if not json_files:
        yield "❌ JSONファイルが見つかりません", "", None
        return
    
    yield f"🔍 **{len(json_files)}個のJSONファイルを検出しました。バリデーション中...**", "", None
    
    # バリデーション
    valid_files, errors = validate_json_files(json_files)
    
    if errors:
        error_msg = "## ❌ バリデーションエラー\n\n" + "\n".join(errors)
        error_msg += f"\n\n**有効なファイル:** {len(valid_files)}/{len(json_files)}"
        
        if not valid_files:
            yield error_msg + "\n\n⛔ 有効なファイルがないため、処理を中止しました。", "", None
            return
        else:
            error_msg += "\n\n⚠️ 有効なファイルのみ処理を続行します..."
            yield error_msg, "", None
    
    yield f"✅ **バリデーション完了！{len(valid_files)}個のファイルを処理します...**", "", None
    
    # マージ処理
    all_texts = []
    total_images = 0
    
    for file_path, data in valid_files:
        filename = Path(file_path).name
        results = data["results"]
        
        for img_name, result in sorted(results.items()):
            text, _ = extract_markdown_text(result)
            if text:
                all_texts.append(text)
                total_images += 1
    
    # 結合（改行2つで区切る）
    combined_text = "\n\n".join(all_texts)
    
    # 一時ファイルに保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = tempfile.mkdtemp()
    
    # TXTファイル
    txt_file = Path(temp_dir) / f"merged_texts_{timestamp}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(combined_text)
    
    status = f"✅ **完了！**\n\n"
    status += f"- **処理ファイル数:** {len(valid_files)}\n"
    status += f"- **合計画像数:** {total_images}\n"
    status += f"- **文字数:** {len(combined_text):,}"
    
    yield status, combined_text, str(txt_file)


# Gradio UI
with gr.Blocks(title="OCR Result Merger") as demo:
    gr.Markdown("# 🔗 OCR結果マージツール")
    gr.Markdown("複数のOCR結果JSONファイルから`markdown_texts`を抽出してまとめます")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="📂 JSONファイルをアップロード",
                file_count="directory",
                file_types=[".json"]
            )
            
            status_display = gr.Markdown("", elem_id="status")
            
            merge_btn = gr.Button("🔗 マージ開始", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📝 プレビュー"):
                    preview_text = gr.Textbox(
                        label="結合されたテキスト",
                        lines=20,
                        max_lines=30
                    )
                with gr.Tab("📥 ダウンロード"):
                    download_file = gr.File(label="TXTファイル")
    
    # Process button click
    merge_btn.click(
        fn=merge_json_files,
        inputs=[file_upload],
        outputs=[status_display, preview_text, download_file]
    )
    
    # 使い方
    gr.Markdown("---")
    gr.Markdown("### 💡 使い方")
    gr.Markdown("""
1. 「JSONファイルをアップロード」をクリック
2. OCR結果のJSONファイルが入っているフォルダを選択
3. 「マージ開始」ボタンをクリック
4. バリデーション結果を確認
   - ❌ エラーがある場合：どのファイルに問題があるか表示されます
   - ✅ 全て有効な場合：マージ処理が実行されます
5. 「ダウンロード」タブからTXTファイルを取得

📌 **注意:** すべてのJSONファイルに`markdown_texts`が含まれている必要があります
    """)


if __name__ == "__main__":
    print("Starting OCR Result Merger...")
    demo.launch(server_name="0.0.0.0", server_port=7862)
