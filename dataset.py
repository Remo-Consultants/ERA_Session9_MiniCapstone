# dataset.py

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import json

class ImagenetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.classes = []
        self.class_to_idx = {}

        self._load_dataset()

    def _load_dataset(self):
        # Assuming ImageNet structure: root_dir/class_name/image.JPEG
        # Or root_dir/train/class_name/image.JPEG and root_dir/val/class_name/image.JPEG

        # Try to find a 'synset_words.txt' or similar for class names,
        # otherwise infer from directory names.
        synset_file = os.path.join(self.root_dir, 'synset_words.txt')
        if os.path.exists(synset_file):
            with open(synset_file, 'r') as f:
                synsets = [line.strip().split(' ', 1) for line in f]
                # Assuming synsets are in the format: "nXXXXXXXXX class_description"
                # We need to map these to the actual directory names
                synset_to_desc = {s[0]: s[1] for s in synsets}
        else:
            synset_to_desc = {} # Fallback if no synset file

        class_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)) and d.startswith('n')]
        self.classes = sorted(class_dirs)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.endswith(('.JPEG', '.png', '.jpg')): # ImageNet typically uses .JPEG
                        self.image_files.append(os.path.join(class_path, img_file))
                        self.labels.append(self.class_to_idx[class_name])
        print(f"Found {len(self.image_files)} images in {self.root_dir}")
        print(f"Found {len(self.classes)} classes.")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None # Return None for error handling in DataLoader

def get_dataloaders(data_path, batch_size, num_workers=4):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dir = os.path.join(data_path, 'TRAIN')
    val_dir = os.path.join(data_path, 'VAL')

    image_datasets = {
        'train': ImagenetDataset(train_dir, data_transforms['train']),
        'val': ImagenetDataset(val_dir, data_transforms['val'])
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True) # drop_last for potentially uneven batches
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
