# Custom ResNet50 for ImageNet Classification

This repository contains a modularized PyTorch implementation of a ResNet50 model, modified for a reduced parameter count, and trained from scratch on the full ImageNet dataset. The goal is to demonstrate a training pipeline for large-scale image classification, with considerations for deployment.

## Features

-   **Modular Codebase**: Organized into `config.py`, `dataset.py`, `model.py`, `train.py`, and `predict.py` for clarity and reusability.
-   **ImageNet Dataset**: Configured to work with the full ImageNet (ILSVRC2012) dataset.
-   **Custom ResNet50**: A ResNet50 architecture with significantly reduced parameters (<150k) to demonstrate model customization. **Note: Achieving high accuracy (80%+) on ImageNet with this parameter constraint and without pre-trained weights is extremely challenging and may not be feasible.**
-   **Training Pipeline**: Includes a complete training and validation loop.
-   **Prediction Script**: A utility to load a trained model and predict on new images.
-   **Deployment Ready**: Instructions for pushing the model to Hugging Face and deploying on AWS EC2.

## Setup

### Prerequisites

-   Python 3.x
-   PyTorch
-   Access to the full ImageNet (ILSVRC2012) dataset.

1.  Clone the repository:
    ```bash
    git clone https://github.com/your_username/resnet50_imagenet_custom.git
    cd resnet50_imagenet_custom
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### ImageNet Dataset

The full ImageNet (ILSVRC2012) dataset is required. Please ensure it is organized in the following structure:

/path/to/full/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   ├── n01443537_10042.JPEG
│   │   └── ...
│   └── ... (all 1000 training classes)
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000001.JPEG
    │   └── ...
    ├── n01443537/
    │   ├── ILSVRC2012_val_00000002.JPEG
    │   └── ...
    └── ... (all 1000 validation classes)

    python -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activateeNet dataset.

You will also need an `imagenet_class_index.json` file for mapping prediction indices to human-readable class names. A common version of this file is available [here](https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json) (though you might need to adapt its format to match what `predict.py` expects, which is a dictionary mapping integer string to a list of `[synset_id, class_name]`).

## Usage

### Training the Model

To train the model:

```bash
python train.py
```

Training on the full ImageNet dataset without pre-trained weights and with a reduced model will take a very long time and require significant GPU resources.

### Making Predictions

To make a prediction on a new image:

1.  Ensure you have a trained model saved in the `models/` directory (e.g., `resnet50_imagenet_custom_final.pth`).
2.  Provide the path to your image in the `predict.py` script (update `example_image_path`).
3.  Run the prediction script:
    ```bash
    python predict.py
    ```

## Deployment

### GitHub

This repository is designed to be hosted on GitHub. Simply push your changes to your remote repository.

### Hugging Face Model Hub

To publish your trained model to Hugging Face:

1.  Install the `huggingface_hub` library (included in `requirements.txt`).
2.  Log in to Hugging Face:
    ```bash
    huggingface-cli login
    ```
    You will be prompted to enter your Hugging Face token.
3.  Create a model repository on Hugging Face (e.g., `your_username/resnet50_imagenet_custom`).
4.  Use the `huggingface_hub` library to push your model. You would add a script for this, or integrate it into your `train.py` after saving the best model.

    Example (can be added to `train.py` or a separate `push_to_hub.py`):
    ```python
    from huggingface_hub import HfApi, create_repo, upload_file
    from config import HF_MODEL_REPO_ID, MODEL_SAVE_PATH, MODEL_NAME
    import os

    def push_to_huggingface(model_path, repo_id):
        api = HfApi()
        # Create repo if it doesn't exist
        create_repo(repo_id=repo_id, exist_ok=True, private=False) # Set private=True if desired

        # Upload model weights
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"pytorch_model.bin", # Standard filename for PyTorch models
            repo_id=repo_id,
        )
        print(f"Model {os.path.basename(model_path)} uploaded to Hugging Face: {repo_id}")

    # Example usage:
    # if __name__ == '__main__':
    #    final_model_path = os.path.join(MODEL_SAVE_PATH, f'{MODEL_NAME}_final.pth')
    #    push_to_huggingface(final_model_path, HF_MODEL_REPO_ID)
    ```
    Remember to update `HF_MODEL_REPO_ID` in `config.py`.

### AWS EC2 Deployment

**1. Launch an EC2 Instance:**

*   **Instance Type**: Choose a GPU instance type (e.g., `g4dn.xlarge`, `p3.2xlarge`, or `p2.xlarge` for cost-effectiveness).
*   **AMI**: Use a Deep Learning AMI that comes pre-configured with CUDA, cuDNN, and PyTorch (e.g., "Deep Learning AMI (Ubuntu 18.04) Version XX.X").
*   **Storage**: Ensure sufficient storage (e.g., 100-200GB SSD) for the ImageNet dataset and model checkpoints.
*   **Security Group**: Configure a security group to allow SSH access.

**2. Connect to the Instance:**

```bash
ssh -i /path/to/your/key.pem ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com
```

**3. Set up Environment:**

The Deep Learning AMI usually has PyTorch installed. If not, or if you need a specific version:

```bash
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```

**4. Transfer Code and Data:**

*   **Code**: Use `git clone` on the EC2 instance to get your repository:
    ```bash
    git clone https://github.com/your_username/resnet50_imagenet_custom.git
    cd resnet50_imagenet_custom
    ```
*   **ImageNet Dataset**: Transfer your ImageNet dataset to the EC2 instance. This can be done via AWS S3 (recommended for large datasets) or `scp`.
    *   **S3 (Recommended)**:
        ```bash
        aws s3 sync s3://your-imagenet-bucket/imagenet /path/to/full/imagenet
        ```
        Ensure your EC2 instance has an IAM role with S3 read access.
    *   **scp**:
        ```bash
        scp -i /path/to/your/key.pem -r /path/to/local/imagenet_data ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com:/path/on/ec2/
        ```
    *   **Update `config.py`**: Make sure `DATA_PATH` in `config.py` points to the correct location of the dataset on the EC2 instance.

**5. Run Training/Prediction:**

*   **Training**:
    ```bash
    python train.py
    ```
*   **Prediction**:
    ```bash
    python predict.py
    ```

## Accuracy Expectations

**Please note:** The model has been modified to significantly reduce the parameter count to under 150,000, and it is trained from scratch without pre-trained weights. Achieving an accuracy of 80% or above on the full ImageNet dataset under these constraints is **highly improbable**. A standard ResNet50 has approximately 25 million parameters. This customized model serves primarily as a demonstration of a highly constrained model architecture and a full training/deployment pipeline, rather than a high-performance classifier. To reach 80%+ accuracy, a much larger model, often pre-trained on ImageNet, would be required.

## Example Demonstration

To demonstrate the model's functionality:

1.  **Train the model** as described above.
2.  **Download a sample image** from the internet (e.g., an image of a "golden retriever").
3.  **Update `predict.py`** with the path to your downloaded image in the `example_image_path` variable.
4.  **Run `predict.py`**:
    ```bash
    python predict.py
    ```
    The output will show the predicted class (e.g., "golden retriever") and the confidence score.
