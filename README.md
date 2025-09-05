# Query, Localize, Explain: A Language-Guided Framework for Open-Set Video Event Localization

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-blue)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/paper-in%20progress-brightgreen)](./)

---

An explainable, language-guided framework for open-set temporal action localization (TAL) on endoscopic videos.
This repository contains the official implementation accompanying the dissertation “An Explainable, Language-Guided Framework for Open-Set Temporal Localization on Endoscopic Videos” by Soheil Jafarifard Bidgoli (MSc Computer Science, Aston University).

🚀 Overview

Surgical and endoscopic procedures generate long, complex videos where clinicians may wish to localize arbitrary events described in natural language (e.g., “when the clipper is applied”).
Traditional methods are closed-set: they only recognize fixed phases or tools. Our framework goes further:

Open-vocabulary localization – Query with free-form clinical language, not just predefined labels.

Efficient temporal modeling – Supports long untrimmed videos via Transformer and State-Space (Mamba) backbones.

Explainability & trustworthiness – Produces attention heatmaps and uncertainty estimates for clinical safety.

🏗️ Framework Architecture

The system is a multi-stage pipeline combining computer vision, natural language processing, and temporal reasoning:

Vision Backbone

Default: M²CRL pretrained video transformer for robust spatio-temporal features.

Alternative: EndoMamba backbone (structured state-space model) for efficient long-video scaling.

Text Encoder

CLIP-based transformer encoder.

Fine-tuned with LoRA (parameter-efficient tuning) for medical terminology.

Language-Guided Fusion Head

Cross-modal transformer aligns video features with text queries.

Outputs frame-level relevance scores + attention heatmaps.

Temporal Head

Baseline: Lightweight Transformer for sequence modeling.

SOTA Upgrade: Mamba-based SSM for linear-complexity long-sequence analysis.

Uncertainty & Explainability

Evidential Deep Learning (EDL) adds calibrated uncertainty scores.

Cross-attention maps provide visual explanations.

Language-Guided-Endoscopy-Localization/
│
├── backbone/                         # Vision backbones
│   ├── endomamba.py                  # EndoMamba (SSM-based backbone)
│   └── vision_transformer.py         # ViT-based backbone (M²CRL, etc.)
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
├── pretrained/                       # Pretrained model weights
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

📊 Baselines & Comparisons

We benchmark against three families of baselines
:

Vision–Language Models (open-vocabulary)

CLIP (zero-shot & linear-probe)

VideoCLIP / X-CLIP

Temporal Grounding Models (language → time interval)

Moment-DETR

Surgical Specialists (closed-set baselines)

TeCNO

📈 Evaluation Protocol

Frame-level Relevance (main task): AUROC, AUPRC, mAP@frame

Temporal Grounding: mAP@tIoU=0.3/0.5/0.7, R@1/R@5

Phase & Tool Recognition: Accuracy, Macro-F1, mAP

Uncertainty calibration and explainability are also reported via EDL and attention maps.

🧑‍⚕️ Dataset

Cholec80 laparoscopic cholecystectomy dataset (80 full-length surgeries).

Preprocessing pipeline generates:

Frame sequences

Language queries (phases + tools)

Positive/negative triplets for training

⚙️ Training

Frame Extraction

python datasets/extract_cholec80_frames.py --input data/raw --output data/frames


Prepare Metadata & Splits

python datasets/prepare_cholec80.py
python datasets/create_splits.py


Train Model

python training/train.py --config configs/m2crl_baseline.yaml

🔬 Key Features

Bi-Level Consistency Loss: Aligns semantic + motion features using optical flow
.

Evidential Deep Learning: Outputs both predictions & uncertainty estimates.

Flexible Backbones: Swap between Transformer and Mamba-based models.

Parameter-Efficient Tuning: LoRA adapters minimize fine-tuning cost.

Cross-modal Explainability: Visualize attention maps for clinician review.

📚 Citation

If you use this framework in your research, please cite:

@mastersthesis{jafarifard2025,
  title={An Explainable, Language-Guided Framework for Open-Set Temporal Localization on Endoscopic Videos},
  author={Soheil Jafarifard Bidgoli},
  school={Aston University},
  year={2025}
}

🤝 Acknowledgements

Supervisor: Dr. Zhuangzhuang Dai

Aston University – MSc Computer Science, CS4700 Dissertation

OpenAI CLIP, M²CRL, VideoMamba, Moment-DETR, and the Cholec80 community

✨ This repo aims to push forward trustworthy, explainable AI for surgery by combining vision, language, and temporal reasoning.

@article{jafari2025query,
  title={Query, Localize, Explain: A Language-Guided Framework for Open-Set Event Localization in Medical Video},
  author={Jafari, Soheil and Dr. Zhuangzhuang Dai},
  journal={arXiv preprint arXiv:25XX.XXXXX},
  year={2025}
}
