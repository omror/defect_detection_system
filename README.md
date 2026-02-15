# Casting Product Defect Detection

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/omerfarukor/defect_detection_system/tree/main)


A Streamlit app to detect manufacturing faults in casting products. I fine-tuned a **ResNet18** model to classify images as either "Defect" or "Normal".

## How it Works

It takes an image, processes it, and runs it through the PyTorch model. It outputs the prediction class along with a confidence score.

## Setup

1. **Clone and install dependencies:**
```bash
git clone <https://github.com/omror/defect_detection_system>
cd defect-detection-system

# Setup venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

```


2. **Model & Data (Important):**
Since the trained model (`defect_model.pth`) and the dataset are large, I excluded them from the repo.
* **If you want to run the app:** Place your trained `defect_model.pth` in the root directory.
* **If you want to train from scratch:** Organize your data in a `data/` folder like this:
```
data/
├── defect/
└── normal/

```


Then run: `python src/train.py`



## Usage

Simply run the Streamlit app:

```bash
streamlit run main.py

```

## Project Structure

* `main.py`: Main app interface.
* `src/train.py`: Training script (PyTorch).
* `src/preprocess.py`: Image transforms and loading logic.
