# Query, Localize, Explain: A Language-Guided Framework for Open-Set Video Event Localization

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-blue)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/paper-in%20progress-brightgreen)](./)

---
This repository contains the official implementation for the dissertation titled: “An Explainable, Language-Guided Framework for Open-Set Temporal Localization on Endoscopic Videos” by Soheil Jafarifard Bidgoli (MSc Computer Science, Aston University).

The project introduces a novel framework for localizing arbitrary, language-described events in long-form surgical videos, moving beyond the limitations of traditional closed-set recognition models.

🚀 Overview
Surgical and endoscopic procedures generate vast amounts of video data. Clinicians often need to find specific moments, but traditional AI models can only recognize a fixed, predefined set of events (e.g., "Phase 1," "Phase 2"). This framework breaks that limitation by enabling open-vocabulary temporal localization. Users can query long, untrimmed videos with free-form natural language to find relevant events (e.g., “find when the grasper retracts the gallbladder”).

Our framework is built on three pillars:


Open-Vocabulary Localization: Leverages powerful vision-language models to understand and locate events described by arbitrary text queries, not just fixed labels.



Architectural Scalability: Employs a hybrid Transformer and Structured State Space Model (SSM) architecture to efficiently process long-form surgical videos, overcoming the quadratic complexity of traditional attention mechanisms.



Trustworthiness & Explainability: Integrates Evidential Deep Learning to quantify model uncertainty and provides visual attention maps to explain its predictions, fostering clinical trust and safety.




✨ Key Features
End-to-End Open-Vocabulary TAL: A complete pipeline from data preprocessing to language-guided inference for surgical video analysis.


Flexible Vision Backbones: Supports both a powerful M²CRL pretrained video transformer and a highly efficient EndoMamba (SSM) backbone for long-sequence modeling.




Parameter-Efficient Fine-Tuning (PEFT): Uses Low-Rank Adaptation (LoRA) to efficiently adapt a pretrained CLIP text encoder to the surgical domain with minimal computational cost.



Advanced Temporal Modeling: Features a state-of-the-art Mamba-based Temporal Head that scales linearly with sequence length, making it ideal for hour-long procedural videos.



Bi-Level Consistency Loss: A novel training objective that enforces temporal consistency at both the semantic and spatial levels using optical flow (RAFT) to regularize the model.



Uncertainty Quantification: Implements Evidential Deep Learning (EDL) to allow the model to express its own confidence, reliably identifying out-of-distribution or ambiguous events.



Built-in Explainability (XAI): Generates cross-modal attention maps to visualize which parts of a frame the model focused on to make its decision, a critical feature for clinical validation.



Comprehensive Baseline Suite: Includes code and instructions to benchmark against canonical baselines like CLIP, X-CLIP, Moment-DETR, and TeCNO.

🏗️ Framework Architecture
The system is a multi-stage pipeline designed to process spatial, semantic, and temporal information through specialized components:


Vision Backbone: A pretrained M²CRL model extracts a grid of powerful visual feature vectors from each video frame.



Text Encoder: A LoRA-adapted CLIP text encoder processes the natural language query into a semantic feature vector.

Language-Guided Fusion Head: A cross-modal transformer uses attention to fuse the visual and textual features. It identifies relevant spatial regions in the frame corresponding to the query, outputting initial 

raw_scores, intermediate features for the consistency loss, and attention_weights for XAI.


Temporal Head (SSM/Mamba): This highly efficient head analyzes the sequence of fused features from the entire clip. It models long-range context to smooth predictions and fill gaps, producing final, contextually-aware 

refined_scores.



Uncertainty & Prediction Head: In its SOTA configuration, this head uses Evidential Deep Learning to output not just a final score but also the parameters of a Beta distribution (evidential_output), allowing for robust uncertainty quantification.

📂 Repository Structure
Language-Guided-Endoscopy-Localization/
│
├── backbone/
│   ├── endomamba.py                  # EndoMamba (SSM-based backbone)
│   └── vision_transformer.py         # ViT-based backbone (M²CRL)
│
├── checkpoints/                      # Saved model checkpoints
│
├── comparison_models/                # Baseline and benchmark models
│   ├── clip_baseline/                # CLIP zero-shot / linear probe
│   ├── Moment-DETR/                  # Moment-DETR temporal grounding
│   └── xclip_baseline/               # X-CLIP video-language baseline
│
├── dataset_preprocessing/            # Scripts for Cholec80 dataset
│   ├── create_splits.py              # Creates train/val/test video splits
│   ├── extract_cholec8o_frames.py    # Extracts frames from videos
│   └── prepare_cholec80.py           # Generates (frame, query, label) triplets
│
├── pretrained/                       # Pretrained model weights
│   └── checkpoint.pth
│
├── dataset.py                        # PyTorch Dataset and DataLoader
├── inference.py                      # Inference script for localization
├── models.py                         # Core architectural components
├── project_config.py                 # Centralized configuration file
├── train.py                          # Main training script
│
└── README.md
🧑‍⚕️ Dataset: Cholec80
This framework is developed and evaluated on the 

Cholec80 dataset, which contains 80 videos of laparoscopic cholecystectomy procedures. Our preprocessing pipeline transforms this dataset into a format suitable for open-vocabulary learning.

Preprocessing Pipeline
Frame Extraction: Videos are decoded into individual frames at a specified sampling rate.

Bash

python dataset_preprocessing/extract_cholec80_frames.py --cholec80_videos_dir /path/to/videos --output_frames_dir /path/to/frames
Create Data Splits: The 80 videos are randomly partitioned into training, validation, and test sets to ensure fair evaluation.

Bash

python dataset_preprocessing/create_splits.py --video_dir /path/to/videos
Generate Language Triplets: The core preprocessing step. This script reads the official phase and tool annotations and generates a CSV file of (frame_path, text_query, relevance_label) triplets. This creates positive and negative examples for training the vision-language alignment.


Bash

# Run for each split
python dataset_preprocessing/prepare_cholec80.py --split train
python dataset_preprocessing/prepare_cholec80.py --split val
python dataset_preprocessing/prepare_cholec80.py --split test
⚙️ Setup and Usage
Installation
Clone the repository:

Bash

git clone https://github.com/soheil-jafari/language-guided-endoscopy-localization.git
cd language-guided-endoscopy-localization
Create a Python environment and install dependencies. We recommend using Conda.

Bash

conda create -n endo-tal python=3.9 -y
conda activate endo-tal
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt # A requirements.txt would need to be created
Configuration: Before running any scripts, review and update the paths in project_config.py to match your system's directory structure.

Training
The main training script train.py handles model training with support for various configurations controlled by project_config.py and command-line arguments.

To train the model from scratch or fine-tune:

Bash

python train.py
To fine-tune from an existing checkpoint:

Bash

python train.py --finetune_from /path/to/your/checkpoint.pth
To run in debug mode on a small data subset:

Bash

python train.py --debug
To adjust the training subset size (e.g., 50% of the data):

Bash

python train.py --subset 0.5
Inference
Use inference.py to run a trained model on a video to localize a specific language query.

Bash

python inference.py \
    --video_path /path/to/your/video.mp4 \
    --text_query "a grasper is present" \
    --checkpoint_path /path/to/your/best_model.pth
The script will output frame-by-frame relevance probabilities and identify the most relevant moment in the video.

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

@article{jafari2025query,
  title={Query, Localize, Explain: A Language-Guided Framework for Open-Set Event Localization in Medical Video},
  author={Jafari, Soheil and Dr. Zhuangzhuang Dai},
  journal={arXiv preprint arXiv:25XX.XXXXX},
  year={2025}
}
