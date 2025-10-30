# predict.py

import torch
from torchvision import transforms
from PIL import Image
import os
import json

from config import DEVICE, NUM_CLASSES, MODEL_SAVE_PATH, MODEL_NAME
from model import build_resnet50_standard

def load_model(model_path, num_classes, device):
    """
    Loads a trained model from a specified path.
    """
    model = build_resnet50_standard(num_classes, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode
    return model

def preprocess_image(image_path):
    """
    Preprocesses an image for model inference.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    return image.unsqueeze(0) # Add batch dimension

def get_class_labels(label_mapping_file='imagenet_class_index.json'):
    """
    Loads ImageNet class labels.
    """
    # This file maps ImageNet synset IDs to readable class names
    # You would typically download or create this file
    # Example structure: {"0": ["n01440764", "tench"], ...}
    if os.path.exists(label_mapping_file):
        with open(label_mapping_file, 'r') as f:
            idx_to_label = json.load(f)
            # Assuming the JSON format is { "index": ["synset_id", "class_name"] }
            # We want to return a list of class names ordered by index
            sorted_labels = [None] * len(idx_to_label)
            for idx_str, details in idx_to_label.items():
                sorted_labels[int(idx_str)] = details[1]
            return sorted_labels
    else:
        print(f"Warning: {label_mapping_file} not found. Using generic class names.")
        return [f"class_{i}" for i in range(NUM_CLASSES)]


def predict_image(model, image_path, class_labels):
    """
    Predicts the class of a given image.
    """
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_idx = torch.topk(probabilities, 1)

    predicted_label = class_labels[top_idx.item()]
    confidence = top_prob.item()

    return predicted_label, confidence

if __name__ == '__main__':
    # Example usage:
    # 1. Ensure you have a trained model saved in MODEL_SAVE_PATH
    # 2. Ensure you have an ImageNet class index file (e.g., imagenet_class_index.json)
    #    You can find one online, e.g., in torchvision's models hub or a similar resource.
    #    For a quick start, I'll provide a placeholder for `imagenet_class_index.json`.

    # Placeholder for imagenet_class_index.json
    # In a real scenario, you'd download or generate this based on your ImageNet dataset.
    sample_imagenet_class_index = {str(i): [f"n{i:09}", f"class_name_{i}"] for i in range(NUM_CLASSES)}
    imagenet_class_index_file = 'imagenet_class_index.json'
    with open(imagenet_class_index_file, 'w') as f:
        json.dump(sample_imagenet_class_index, f, indent=4)


    model_path = os.path.join(MODEL_SAVE_PATH, f'{MODEL_NAME}_final.pth') # Or your best_model_*.pth
    # Check if a model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
    else:
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, NUM_CLASSES, DEVICE)
        print("Model loaded.")

        class_labels = get_class_labels(imagenet_class_index_file)

        # Example image to predict
        # Replace 'path/to/your/image.jpg' with a real image path for demonstration
        example_image_path = 'path/to/your/image.jpg'
        if not os.path.exists(example_image_path):
            print(f"Error: Example image not found at {example_image_path}. Please provide a valid image path.")
        else:
            print(f"\nPredicting for image: {example_image_path}")
            predicted_label, confidence = predict_image(model, example_image_path, class_labels)
            print(f"Predicted Class: {predicted_label}")
            print(f"Confidence: {confidence:.4f}")
