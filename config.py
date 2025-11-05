# config.py

import os
import torch

# =========================
# Dataset Configuration
# =========================
# Updated path to dataset downloaded via Kaggle API
DATA_PATH = os.path.expanduser('~/ERA_Session9_MiniCapstone/IMAGENET/')
NUM_CLASSES = 1000  # ImageNet has 1000 classes

# =========================
# Model Configuration
# =========================
MODEL_NAME = "resnet50_imagenet"

# =========================
# Training Configuration
# =========================
BATCH_SIZE = 64             # Adjust as per GPU memory (32/64/128 typical for g4dn.2xlarge)
LEARNING_RATE = 0.01        # Suitable starting point for ImageNet from scratch
NUM_EPOCHS = 100
STEP_LR_GAMMA = 0.1
STEP_LR_STEP_SIZE = 30      # Step LR every 30 epochs

# =========================
# Device Configuration
# =========================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# Paths for saving model and logs
# =========================
MODEL_SAVE_PATH = 'models/' # Directory in your project to save trained models
LOG_DIR = 'logs/'           # Directory for logs (if any)

# =========================
# Optional/Advanced: Hugging Face or experiment tracking (update if/when needed)
# =========================
# HF_MODEL_REPO_ID = "your_username/resnet50_imagenet" # Uncomment if using Hugging Face Hub