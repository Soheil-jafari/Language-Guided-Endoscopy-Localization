🎯 Moment-DETR for Custom Datasets

A clean and streamlined implementation of Moment-DETR for language-based temporal video grounding on custom datasets.
The task: given a natural language query, the model predicts the [start, end] timestamps in a video where the described event occurs.

This repo refactors the official Moment-DETR into a modular, easy-to-use pipeline, making it simple to adapt for new datasets.

🚀 Key Features

🔄 End-to-end pipeline (Preprocess → Feature Extraction → Training → Evaluation)

🖼️ ResNet-50 feature extractor for efficient training

⚡ Multi-GPU support with torch.distributed

📊 Standard metrics: mAP@tIoU, Recall@k

🛠️ Plug-and-play configs for quick dataset integration

📌 Overview & Pipeline

<img width="300" height="400" alt="ChatGPT Image Aug 29, 2025, 11_25_38 PM" src="https://github.com/user-attachments/assets/3ea1e7a2-8956-4f37-937c-e4a649fad517" />

The workflow consists of four stages, each handled by a dedicated script:

graph TD;
    A[📑 Preprocessing <br> run_preprocessing.py] --> B[🖼️ Feature Extraction <br> run_feature_extraction.py];
    B --> C[🧠 Training <br> run_training.py];
    C --> D[📊 Evaluation <br> run_evaluation.py];


👉 This modular design means you can preprocess, extract features, train, and evaluate independently.

⚙️ Setup & Installation

We recommend a dedicated Conda environment.

# 1. Clone the repository
git clone https://github.com/yourname/moment-detr-custom.git
cd moment_detr_baseline

# 2. Create and activate environment
conda create -n detr python=3.9 -y
conda activate detr

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

🛠️ Usage: Step-by-Step
🔧 Step 0: Data Configuration

Define dataset paths in your config file.

Update run_preprocessing.py and run_feature_extraction.py with either:

sys.path.append(...) → config file, OR

Replace with direct paths (proj_config.TRAIN_TRIPLETS_CSV_PATH, etc.)

👉 Annotation CSV format:

video_id	text	start_frame	end_frame
📑 Step 1: Preprocess Annotations

Convert CSV annotations → JSONL format.

python run_preprocessing.py


✅ Outputs: train.jsonl, val.jsonl, test.jsonl → preprocessed_data/

🖼️ Step 2: Extract Visual Features

Extract features using ResNet-50.

python run_feature_extraction.py


Resumable (skips completed videos)

Saves features → extracted_features_resnet50/

🧠 Step 3: Train the Model

Train with multi-GPU support.

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 run_training.py


Checkpoints saved → checkpoints/

Customize GPUs via nproc_per_node

📊 Step 4: Evaluate the Model

Evaluate trained checkpoint.

python run_evaluation.py --resume /path/to/checkpoints/run_name/best_checkpoint.ckpt


Results printed + saved in logs

Metrics: mAP@tIoU, Recall@k

🖼️ Example Results

Here you can showcase results with GIFs or sample visualizations. For example:

Pipeline GIF (data → features → training → predictions)

Sample grounding outputs: query + predicted [start, end] overlayed on video timeline

![Pipeline Overview](docs/pipeline.gif)  
![Sample Result](docs/result_example.png)  

📚 Citation

This code adapts from the original Moment-DETR:

End-to-End Video Instance Segmentation with Transformers
Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chun-Guang Li, Yong-Qiang Yao, Yue-Meng Li, Gaofeng Meng
CVPR 2021

If you use this repo, please cite the paper.

@inproceedings{momentdetr2021,
  title={End-to-End Video Instance Segmentation with Transformers},
  author={Wang, Yuqing and Xu, Zhaoliang and Wang, Xinlong and Li, Chun-Guang and Yao, Yong-Qiang and Li, Yue-Meng and Meng, Gaofeng},
  booktitle={CVPR},
  year={2021}
}


✨ To-Do / Extensions

 Add support for more backbones (e.g., Swin Transformer)

 Provide pretrained weights on benchmark datasets

 Add demo notebooks for quick inference

🔥 With this repo, you can train Moment-DETR on any dataset with just a CSV file of annotations and raw videos.
