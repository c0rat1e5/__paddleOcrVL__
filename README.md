# PaddleOCR-VL Gradio Application

🔍 **PaddleOCR-VL** - Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model

This is a local Gradio application that uses the [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) model directly with the `transformers` library.

## Features

- 📝 **Text Recognition (OCR)**: General text, documents, handwriting in 109 languages
- 📐 **Formula Recognition**: Mathematical formulas and LaTeX equations
- 📊 **Table Recognition**: Structured tables (outputs HTML)
- 📈 **Chart Recognition**: Bar charts, pie charts, line graphs, etc.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (recommended for faster inference)
- ~2GB GPU VRAM (model is 0.9B parameters)

## Installation

1. **Clone or download this project**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install flash-attn for faster inference**
   ```bash
   pip install flash-attn
   ```

## Usage

Run the application:

```bash
python app.py
```

The application will start at `http://localhost:7860`

## Model Details

| Property | Value |
|----------|-------|
| Model | PaddlePaddle/PaddleOCR-VL |
| Parameters | 0.9B |
| Base Model | ERNIE-4.5-0.3B |
| Languages | 109 languages |
| License | Apache 2.0 |

## References

- [GitHub Repository](https://github.com/PaddlePaddle/PaddleOCR)
- [Technical Report (arXiv)](https://arxiv.org/abs/2510.14528)
- [HuggingFace Model](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- [Online Demo](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo)

## License

Apache 2.0 - Same as the original PaddleOCR-VL model.

## Citation

```bibtex
@misc{cui2025paddleocrvlboostingmultilingualdocument,
      title={PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model}, 
      author={Cheng Cui and Ting Sun and Suyin Liang and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Xueqing Wang and Changda Zhou and Hongen Liu and Manhui Lin and Yue Zhang and Yubo Zhang and Handong Zheng and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2510.14528},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14528}, 
}
```
