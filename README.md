# An Explainable, Language-Guided Framework for Open-Set Temporal Localization in Endoscopic Videos

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-blue)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the dissertation titled: *“An Explainable, Language-Guided Framework for Open-Set Temporal Localization on Endoscopic Videos”* by Soheil Jafarifard Bidgoli (MSc Computer Science, Aston University).

The project introduces a novel framework for localizing arbitrary, language-described events in long-form surgical videos, moving beyond the limitations of traditional closed-set recognition models.

## 🚀 Overview

Surgical and endoscopic procedures generate vast amounts of video data. Clinicians often need to find specific moments, but traditional AI models can only recognize a fixed, predefined set of events (e.g., "Phase 1," "Phase 2"). This framework breaks that limitation by enabling **open-vocabulary temporal localization**. Users can query long, untrimmed videos with free-form natural language to find relevant events (e.g., *“find when the grasper retracts the gallbladder”*).

Our framework is built on three pillars:
1.  **Open-Vocabulary Localization**: Leverages powerful vision-language models to understand and locate events described by arbitrary text queries, not just fixed labels.
2.  **Architectural Scalability**: Employs a hybrid Transformer and Structured State Space Model (SSM) architecture to efficiently process long-form surgical videos, overcoming the quadratic complexity of traditional attention mechanisms.
3.  **Trustworthiness & Explainability**: Integrates Evidential Deep Learning to quantify model uncertainty and provides visual attention maps to explain its predictions, fostering clinical trust and safety.

## ✨ Key Features

* **End-to-End Open-Vocabulary TAL**: A complete pipeline from data preprocessing to language-guided inference for surgical video analysis.
* **Flexible Vision Backbones**: Supports both a powerful **M²CRL** pretrained video transformer and a highly efficient **EndoMamba** (SSM) backbone for long-sequence modeling.
* **Parameter-Efficient Fine-Tuning (PEFT)**: Uses Low-Rank Adaptation (**LoRA**) to efficiently adapt a pretrained CLIP text encoder to the surgical domain with minimal computational cost.
* **Advanced Temporal Modeling**: Features a state-of-the-art **Mamba-based Temporal Head** that scales linearly with sequence length, making it ideal for hour-long procedural videos.
* **Bi-Level Consistency Loss**: A novel training objective that enforces temporal consistency at both the semantic and spatial levels using optical flow (**RAFT**) to regularize the model.
* **Uncertainty Quantification**: Implements **Evidential Deep Learning (EDL)** to allow the model to express its own confidence, reliably identifying out-of-distribution or ambiguous events.
* **Built-in Explainability (XAI)**: Generates cross-modal attention maps to visualize which parts of a frame the model focused on to make its decision, a critical feature for clinical validation.
* **Comprehensive Baseline Suite**: Includes code and instructions to benchmark against canonical baselines like **CLIP**, **X-CLIP**, **Moment-DETR**, and **TeCNO**.

## 🏗️ Framework Architecture

The system is a multi-stage pipeline designed to process spatial, semantic, and temporal information through specialized components:

1.  **Vision Backbone**: A pretrained M²CRL model extracts a grid of powerful visual feature vectors from each video frame.
2.  **Text Encoder**: A LoRA-adapted CLIP text encoder processes the natural language query into a semantic feature vector.
3.  **Language-Guided Fusion Head**: A cross-modal transformer uses attention to fuse the visual and textual features. It identifies relevant spatial regions in the frame corresponding to the query, outputting initial `raw_scores`, intermediate features for the consistency loss, and `attention_weights` for XAI.
4.  **Temporal Head (SSM/Mamba)**: This highly efficient head analyzes the sequence of fused features from the entire clip. It models long-range context to smooth predictions and fill gaps, producing final, contextually-aware `refined_scores`.
5.  **Uncertainty & Prediction Head**: In its SOTA configuration, this head uses Evidential Deep Learning to output not just a final score but also the parameters of a Beta distribution (`evidential_output`), allowing for robust uncertainty quantification.

## 📂 Repository Structure
    Language-Guided-Endoscopy-Localization/
    │
    ├── backbone/                         # Vision backbones
    │   ├── endomamba.py                  # EndoMamba (SSM-based backbone)
    │   └── vision_transformer.py         # ViT-based backbone (M²CRL)
    │
    ├── checkpoints/                      # Saved checkpoints and logs
    │
    ├── comparison_models/                # Baseline and benchmark models
    │   ├── clip_baseline/
    │   │   └── clip_baseline.py          # CLIP zero-shot / linear probe
    │   │
    │   ├── Moment-DETR/                  # Moment-DETR temporal grounding
    │   │   ├── run_evaluation.py
    │   │   ├── run_feature_extraction.py
    │   │   ├── run_preprocessing.py
    │   │   ├── run_training.py
    │   │   └── moment_detr_module/
    │   │       ├── __init__.py
    │   │       ├── configs.py
    │   │       ├── dataset.py
    │   │       ├── engine.py
    │   │       ├── loss.py
    │   │       ├── matcher.py
    │   │       ├── modeling.py
    │   │       ├── position_encoding.py
    │   │       ├── transformer.py
    │   │       ├── utils.py
    │   │       └── README.md
    │   │
    │   └── xclip_baseline/               # X-CLIP video-language baseline
    │       ├── train_xclip.py
    │       ├── eval_xclip.py
    │       ├── infer_xclip.py
    │       ├── requirements.txt
    │       ├── project_config.py
    │       ├── README_XCLIP.md
    │       └── xclip_package/
    │           └── xclip/
    │               ├── __init__.py
    │               ├── data.py
    │               ├── losses.py
    │               ├── metrics.py
    │               ├── model.py
    │               └── utils.py
    │
    ├── dataset_preprocessing/            # Preprocessing for Cholec80 dataset
    │   ├── create_splits.py
    │   ├── extract_cholec80_frames.py
    │   └── prepare_cholec80.py
    │
    ├── pretrained/                       # Pretrained model weights (M2CRL)
    │   └── checkpoint.pth
    │
    ├── dataset.py                        # Dataset wrapper
    ├── inference.py                      # Inference script (language-guided)
    ├── models.py                         # Main model components
    ├── project_config.py                 # Config file for project settings
    ├── train.py                          # Training entry point
    │
    ├── README.md                         # Project documentation
    └── .gitignore
## 🧑‍⚕️ Dataset: Cholec80

This framework is developed and evaluated on the **Cholec80 dataset**, which contains 80 videos of laparoscopic cholecystectomy procedures. Our preprocessing pipeline transforms this dataset into a format suitable for open-vocabulary learning.

### Preprocessing Pipeline

1.  **Frame Extraction**: Videos are decoded into individual frames at a specified sampling rate.
    ```bash
    python dataset_preprocessing/extract_cholec80_frames.py --cholec80_videos_dir /path/to/videos --output_frames_dir /path/to/frames
    ```
2.  **Create Data Splits**: The 80 videos are randomly partitioned into training, validation, and test sets to ensure fair evaluation.
    ```bash
    python dataset_preprocessing/create_splits.py --video_dir /path/to/videos
    ```
3.  **Generate Language Triplets**: The core preprocessing step. This script reads the official phase and tool annotations and generates a CSV file of `(frame_path, text_query, relevance_label)` triplets. This creates positive and negative examples for training the vision-language alignment.
    ```bash
    # Run for each split
    python dataset_preprocessing/prepare_cholec80.py --split train
    python dataset_preprocessing/prepare_cholec80.py --split val
    python dataset_preprocessing/prepare_cholec80.py --split test
    ```

## ⚙️ Setup and Usage

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/soheil-jafari/language-guided-endoscopy-localization.git
    cd language-guided-endoscopy-localization
    ```
2.  Create a Python environment and install dependencies. We recommend using Conda.
    ```bash
    conda create -n endo-tal python=3.9 -y
    conda activate endo-tal
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt # A requirements.txt would need to be created
    ```

3.  **Configuration**: Before running any scripts, review and update the paths in `project_config.py` to match your system's directory structure.

### Training

The main training script `train.py` handles model training with support for various configurations controlled by `project_config.py` and command-line arguments.

* **To train the model from scratch or fine-tune:**
    ```bash
    python train.py
    ```
* **To fine-tune from an existing checkpoint:**
    ```bash
    python train.py --finetune_from /path/to/your/checkpoint.pth
    ```
* **To run in debug mode on a small data subset:**
    ```bash
    python train.py --debug
    ```
* **To adjust the training subset size (e.g., 50% of the data):**
    ```bash
    python train.py --subset 0.5
    ```

### Inference

Use `inference.py` to run a trained model on a video to localize a specific language query.

```bash
python inference.py \
    --video_path /path/to/your/video.mp4 \
    --text_query "a grasper is present" \
    --checkpoint_path /path/to/your/best_model.pth
📊 Baselines and Comparisons
This repository includes the necessary code and instructions to benchmark our framework against three key families of models:

General Vision-Language Models: For open-set, text-driven evaluation.

CLIP: Zero-shot and linear-probe per-frame relevance scoring.

X-CLIP: A powerful video-language model for scoring short clips.

Temporal Grounding Models: For the direct task of localizing events from text.

Moment-DETR: Predicts start/end boundaries from a language query.

Surgical Specialist Models: Closed-set baselines trained specifically for Cholec80.

TeCNO: A temporal convolutional network for surgical phase recognition.

The code for these baselines can be found in the comparison_models/ directory. Each subfolder contains a README with specific instructions for running that model.

📚 Citation
If you use this framework or ideas from our work in your research, please cite the following dissertation:

Code snippet

@mastersthesis{jafarifard2025,
  title={An Explainable, Language-Guided Framework for Open-Set Temporal Localization on Endoscopic Videos},
  author={Soheil Jafarifard Bidgoli},
  school={Aston University},
  year={2025}
}
🤝 Acknowledgements
This work was completed as part of the CS4700 Dissertation for the MSc in Computer Science at Aston University.

Supervisor: Dr. Zhuangzhuang Dai.

This project builds upon the foundational work of the Cholec80 dataset creators and the authors of M²CRL, VideoMamba, CLIP, Moment-DETR, and other referenced works.
