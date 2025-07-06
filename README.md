# Query, Localize, Explain: A Language-Guided Framework for Open-Set Video Event Localization

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-blue)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/paper-in%20progress-brightgreen)](./)

---

This repository contains the official PyTorch implementation for the research paper: **"Query, Localize, Explain: A Language-Guided Framework for Open-Set Event Localization in Medical Video"**. Our work introduces a new paradigm for video analysis that moves beyond rigid, "closed-set" detectors, enabling users to find any temporal event that can be described with natural language.

## 🌟 Overview

Traditional video analysis models are limited to a predefined list of event classes. Our framework breaks this limitation by learning the deep semantic relationship between visual data and text. It can localize events based on free-form text queries (e.g., "a polyp being removed by a snare") and, crucially, provides visual explanations for its predictions, addressing the critical need for trust and transparency in clinical AI.


## ✨ Key Features

* **Open-Set Localization**: Find anything you can describe with text, not just a fixed list of classes. This creates a flexible and future-proof tool for clinical review and research.
* **Explainable AI (XAI)**: The model generates visual heatmaps that highlight the specific pixels it "looked at" to make a decision, providing crucial insight and building trust.
* **State-of-the-Art Architecture**: A synergistic pipeline combining a powerful frozen vision backbone (M²CRL), a PEFT-tuned text encoder, a true cross-attention fusion head, and a temporal transformer for robust context modeling.
* **Demonstrated Generalization**: The framework is designed to learn a generalizable skill, with evaluation protocols to test its zero-shot transfer capabilities from the medical domain to general-purpose benchmarks.

---

## 🏗️ Architectural Pipeline

Our framework processes video through four distinct, specialized stages:

1.  **Vision Backbone (`M²CRL`)**: A frozen M²CRL model extracts powerful, general-purpose feature maps from each video frame.
    ```
    [Frame] -> [Backbone] -> [Visual Feature Map]
    ```
2.  **Language-Guided Head (`Cross-Attention`)**: Fuses the visual feature map with the text query features. It performs **spatial attention** to identify relevant regions within the frame.
    ```
    ([Visual Features], [Text Features]) -> [Cross-Attention] -> [Frame Relevance Score + XAI Heatmap]
    ```
3.  **Temporal Head (`Temporal Transformer`)**: Receives the sequence of raw scores for all frames in a clip. It performs **temporal attention** to model the global event dynamics and outputs a smoothed, contextually-aware score sequence.
    ```
    [Score Sequence] -> [Temporal Transformer] -> [Refined Score Sequence]
    ```
4.  **Final Prediction (`Thresholding`)**: A simple final module that converts the refined score sequence into concrete `[start_time, end_time]` predictions.

---

## 🚀 Getting Started

### 1. Setup Environment

It is recommended to use a Conda environment.

```bash
# Create and activate the environment
conda create -n loc_env python=3.9
conda activate loc_env

# Install dependencies from the requirements file
pip install -r requirements.txt

2. Prepare Data
This project uses a two-step data preparation process for training on the server:

Pre-processing Script: First, run the prepare_data.py script on the server. This script will iterate through your datasets and generate a single master CSV file containing (frame_path, text_query, relevance_label) triplets.

Dataloader: The dataset.py script reads this pre-processed CSV file during training, making the training process highly efficient.

3. Training on the Server
The train.py script is designed to be run on a multi-GPU server like the Aston EPS ML Server.

# 1. Start a persistent terminal session
tmux

# 2. Set the GPUs you have booked (e.g., GPUs 2 and 3)
export CUDA_VISIBLE_DEVICES=2,3

# 3. Launch the training script
# The script will read all settings from config.py
python train.py

# 4. Detach from the session (Ctrl+B then D) and let it run

4. Local Testing & Inference
Use the inference.py script to test a trained model checkpoint on a single video file. This is perfect for local testing and visualization.

python inference.py \
    --video_path "/path/to/your/sample_video.mp4" \
    --text_query "a polyp being removed" \
    --checkpoint_path "./checkpoints/model_epoch_10.pth"

📜 Citation
If you find this work useful in your research, please consider citing our paper:

@article{jafari2025query,
  title={Query, Localize, Explain: A Language-Guided Framework for Open-Set Event Localization in Medical Video},
  author={Jafari, Soheil and Dr. Zhuangzhuang Dai},
  journal={arXiv preprint arXiv:25XX.XXXXX},
  year={2025}
}
