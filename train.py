import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import our project components
import config
from models import LocalizationFramework
from dataset import create_dataloader


# --- 1. Define the Loss Functions ---

class DualObjectiveLoss(nn.Module):
    """
    A custom loss function that combines the primary relevance loss
    with a temporal consistency loss.
    """

    def __init__(self, temporal_weight=config.TEMPORAL_LOSS_WEIGHT):
        super().__init__()
        self.relevance_loss = nn.BCEWithLogitsLoss()
        self.temporal_weight = temporal_weight

    def forward(self, refined_scores, raw_scores, ground_truth_relevance):
        """
        Calculates the combined loss.
        Args:
            refined_scores (torch.Tensor): The output from the Temporal Head [B, T, 1].
            raw_scores (torch.Tensor): The output from the Language-Guided Head [B, T, 1].
            ground_truth_relevance (torch.Tensor): The target labels [B, T, 1].
        """
        # --- Primary Loss ---
        # Compare the refined scores with the ground truth labels.
        primary_loss = self.relevance_loss(refined_scores, ground_truth_relevance)

        # --- Temporal Consistency Loss ---
        # This loss encourages the scores of adjacent frames to be similar.
        # We calculate the L1 difference between consecutive frames in the refined scores sequence.
        # We use refined_scores here to directly penalize a non-smooth final output.
        if self.temporal_weight > 0 and refined_scores.shape[1] > 1:  # Only if sequence has more than 1 frame
            # Sliced tensors for adjacent frames
            scores_t = refined_scores[:, 1:, :]  # All frames except the first
            scores_t_minus_1 = refined_scores[:, :-1, :]  # All frames except the last

            # Calculate the mean absolute difference
            temporal_loss = torch.mean(torch.abs(scores_t - scores_t_minus_1))
        else:
            temporal_loss = 0.0

        # --- Combined Loss ---
        total_loss = primary_loss + (self.temporal_weight * temporal_loss)

        return total_loss, primary_loss, temporal_loss


# --- 2. The Main Training Function ---

def train_model():
    """
    The main function to orchestrate the model training process.
    """
    # --- Setup ---
    print("--- Starting Training Setup ---")
    device = torch.device(config.DEVICE)

    # Create the directory for saving model checkpoints if it doesn't exist
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --- Initialize Model, Dataloader, Optimizer, and Loss ---
    print("Initializing model...")
    model = LocalizationFramework().to(device)

    print("Creating dataloader...")

    train_loader = create_dataloader(config.TRIPLETS_CSV_PATH)

    # Optimizer: AdamW is a good default for transformer-based models.
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # The custom dual-objective loss function
    criterion = DualObjectiveLoss()

    print("--- Setup Complete. Starting Training... ---")

    # --- Training Loop ---
    for epoch in range(config.EPOCHS):
        model.train()

        total_loss_epoch = 0.0
        total_primary_loss_epoch = 0.0
        total_temporal_loss_epoch = 0.0

        # Use tqdm for a progress bar over the training data
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", unit="batch")

        for batch in progress_bar:
            # Unpack the batch and move tensors to the configured device (GPU)
            video_clip, input_ids, attention_mask, relevance_labels = [b.to(device) for b in batch]

            # Zero the gradients from the previous step
            optimizer.zero_grad()

            # --- Forward Pass ---
            # The model returns both raw and refined scores
            raw_scores, refined_scores = model(video_clip, input_ids, attention_mask)

            # --- Loss Calculation ---
            # Calculate the combined loss using our custom loss function
            loss, primary_loss, temporal_loss = criterion(refined_scores, raw_scores, relevance_labels)

            # --- Backward Pass & Optimization ---
            loss.backward()
            optimizer.step()

            # --- Logging ---
            total_loss_epoch += loss.item()
            total_primary_loss_epoch += primary_loss.item() if isinstance(primary_loss, torch.Tensor) else primary_loss
            total_temporal_loss_epoch += temporal_loss.item() if isinstance(temporal_loss,
                                                                            torch.Tensor) else temporal_loss

            # Update the progress bar with the current average loss
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                primary=f"{primary_loss.item():.4f}",
                temporal=f"{temporal_loss.item():.4f}"
            )

        # --- End of Epoch ---
        avg_loss = total_loss_epoch / len(train_loader)
        avg_primary_loss = total_primary_loss_epoch / len(train_loader)
        avg_temporal_loss = total_temporal_loss_epoch / len(train_loader)

        print(f"\nEnd of Epoch {epoch + 1}: Avg Loss: {avg_loss:.4f}, "
              f"Avg Primary Loss: {avg_primary_loss:.4f}, "
              f"Avg Temporal Loss: {avg_temporal_loss:.4f}\n")

        # --- Save Model Checkpoint ---
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


# --- 3. Script Execution ---

if __name__ == '__main__':
    # --- Instructions for Running on the EPS ML Server ---
    print("=" * 60)
    print("INSTRUCTIONS FOR RUNNING ON THE EPS ML SERVER:")
    print("1. Start a tmux session: `tmux`")
    print("2. Book a GPU group and find your assigned GPU IDs (e.g., 2, 3).")
    print("3. Set the CUDA_VISIBLE_DEVICES environment variable:")
    print("   `export CUDA_VISIBLE_DEVICES=2,3`")
    print("4. Run this script:")
    print("   `python train.py`")
    print("5. Detach from the tmux session by pressing `Ctrl+B` then `D`.")
    print("=" * 60)

    # Make sure the required path is in the config file
    if not hasattr(config, 'TRIPLETS_CSV_PATH') or not config.TRIPLETS_CSV_PATH:
        raise ValueError("Please define TRIPLETS_CSV_PATH in config.py to point to your pre-processed CSV file.")

    train_model()

