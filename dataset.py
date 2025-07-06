import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
import os
import cv2
from torchvision import transforms
import transformers
from transformers import AutoTokenizer

import config


class EndoscopyLocalizationDataset(Dataset):
    """
    Custom PyTorch dataset for loading pre-generated (frame_path, text_query, relevance_label) triplets.
    """

    def __init__(self, triplets_csv_path, tokenizer,
                 img_size=config.IMG_SIZE):  # Assuming IMG_SIZE is defined in config
        """
        Args:
            triplets_csv_path (str): Path to the CSV file containing the generated triplets.
                                     Expected columns: 'frame_path', 'text_query', 'relevance_label'.
            tokenizer: The pre-initialized text tokenizer (e.g., from CLIP).
            img_size (int): The target size for image resizing (e.g., 224 for ViT).
        """
        self.triplets_df = pd.read_csv(triplets_csv_path)
        self.tokenizer = tokenizer
        self.img_size = img_size


        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert OpenCV BGR numpy array to PIL Image for transforms
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # --- UPDATED NORMALIZATION FOR M2CRL ---
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

    def __len__(self):
        return len(self.triplets_df)

    def __getitem__(self, idx):
        triplet = self.triplets_df.iloc[idx]

        frame_path = triplet['frame_path']
        text_query = str(triplet['text_query'])  # Ensure text is string
        relevance = float(triplet['relevance_label'])

        # --- Frame Loading and Preprocessing ---
        try:
            # Load the image with OpenCV (reads as BGR numpy array)
            frame = cv2.imread(frame_path)
            if frame is None:
                raise FileNotFoundError(f"Could not read frame: {frame_path}")

            # Convert from BGR (OpenCV default) to RGB for torchvision transforms
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply defined image transformations (resize, ToTensor, Normalize)
            frame = self.image_transform(frame)

        except Exception as e:
            print(f"Warning: Error loading/processing frame {frame_path}: {e}. Returning a dummy sample.",
                  file=sys.stderr)
            # If a frame is missing or corrupt, return a dummy sample or re-sample
            # For simplicity, we'll return a dummy zero tensor and re-sample in practice
            # In a real training loop, you might want to handle this more robustly
            dummy_frame = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
            dummy_input_ids = torch.zeros(config.MAX_TEXT_LENGTH, dtype=torch.long)
            dummy_attention_mask = torch.zeros(config.MAX_TEXT_LENGTH, dtype=torch.long)
            dummy_relevance = torch.tensor(0.0, dtype=torch.float32)
            return dummy_frame, dummy_input_ids, dummy_attention_mask, dummy_relevance

        # --- Text Tokenization ---
        text_inputs = self.tokenizer(
            text_query,
            padding='max_length',
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH,
            return_tensors="pt"
        )
        input_ids = text_inputs['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = text_inputs['attention_mask'].squeeze(0)

        relevance = torch.tensor(relevance, dtype=torch.float32)

        return frame, input_ids, attention_mask, relevance


def create_dataloader(triplets_csv_path):
    """
    A helper function to set up the dataset and dataloader.
    """
    print(f"Loading triplets from {triplets_csv_path}...")

    # Initialize the tokenizer (needs to be done here or passed in)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    # Create the dataset instance
    dataset = EndoscopyLocalizationDataset(
        triplets_csv_path=triplets_csv_path,
        tokenizer=tokenizer,
        img_size=config.IMG_SIZE  # Pass img_size from config
    )

    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Shuffle for training
        num_workers=config.NUM_WORKERS,  # Set based on your server's core allocation
        pin_memory=True  # Speeds up data transfer to GPU
    )

    print(f"DataLoader created successfully with {len(dataset)} samples.")
    return dataloader


if __name__ == '__main__':
    # This block allows you to test the dataloader independently
    print("Testing the dataloader...")

    # --- IMPORTANT: Replace this with the actual path to your generated triplets CSV ---
    # This path will be the OUTPUT_TRIPLETS_FILE from prepare_real_colon.py
    TEST_TRIPLETS_CSV = config.OUTPUT_TRIPLETS_FILE  # Assuming config.py has this path

    # Ensure config.py has IMG_SIZE defined, e.g., IMG_SIZE = 224
    if not hasattr(config, 'IMG_SIZE'):
        print("Warning: config.IMG_SIZE not found. Setting to default 224.", file=sys.stderr)
        config.IMG_SIZE = 224

    # Ensure config.py has OUTPUT_TRIPLETS_FILE defined
    if not hasattr(config, 'OUTPUT_TRIPLETS_FILE'):
        print("Error: config.OUTPUT_TRIPLETS_FILE not found. Please define it in config.py.", file=sys.stderr)
        sys.exit(1)

    # Ensure the test CSV actually exists
    if not os.path.exists(TEST_TRIPLETS_CSV):
        print(f"Error: Test triplets CSV not found at {TEST_TRIPLETS_CSV}. Please run prepare_real_colon.py first.",
              file=sys.stderr)
        sys.exit(1)

    train_loader = create_dataloader(TEST_TRIPLETS_CSV)

    # Fetch one batch of data to see if it works
    try:
        frame_batch, ids_batch, mask_batch, relevance_batch = next(iter(train_loader))

        print("\n--- Batch Test Successful ---")
        print(f"Frame batch shape: {frame_batch.shape}")
        print(f"Input IDs batch shape: {ids_batch.shape}")
        print(f"Attention mask batch shape: {mask_batch.shape}")
        print(f"Relevance batch shape: {relevance_batch.shape}")
        print(f"Example relevance values: {relevance_batch[:5]}")
        print("---------------------------\n")

    except Exception as e:
        print(f"\n--- Error testing dataloader ---")
        import traceback

        traceback.print_exc()  # Print full traceback for debugging
        print(e)
        print("------------------------------\n")

