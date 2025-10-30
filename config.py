# config.py

import torch

# Dataset Configuration
DATA_PATH = '/path/to/full/imagenet'  # This path needs to be updated with the actual ImageNet path
NUM_CLASSES = 1000  # Full ImageNet has 1000 classes

# Model Configuration
MODEL_NAME = "resnet50_imagenet_custom"
# Note: Achieving 80%+ accuracy with <150k parameters on full ImageNet is highly unlikely.
# The current model modifications are designed to meet the parameter constraint,
# but a larger model would be necessary for high accuracy.

# Training Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100 # This might need to be adjusted for full ImageNet
STEP_LR_GAMMA = 0.1
STEP_LR_STEP_SIZE = 7

# Device Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths for saving model and logs
MODEL_SAVE_PATH = 'models/'
LOG_DIR = 'logs/'

# Hugging Face Configuration
HF_MODEL_REPO_ID = "your_username/resnet50_imagenet_custom" # Update with your Hugging Face username and desired repo name