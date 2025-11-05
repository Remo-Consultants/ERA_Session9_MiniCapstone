

```markdown
# ERA Session 9 Mini Capstone:  ResNet50 for ImageNet

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Remo-Consultants/ERA_Session9_MiniCapstone)](https://github.com/Remo-Consultants/ERA_Session9_MiniCapstone/commits/main)
[![Issues](https://img.shields.io/github/issues/Remo-Consultants/ERA_Session9_MiniCapstone)](https://github.com/Remo-Consultants/ERA_Session9_MiniCapstone/issues)
[![Stars](https://img.shields.io/github/stars/Remo-Consultants/ERA_Session9_MiniCapstone?style=social)](https://github.com/Remo-Consultants/ERA_Session9_MiniCapstone/stargazers)

---

> **A modular PyTorch ResNet50 pipeline for full ImageNet, robust for AWS EC2 and cloud training**

---

## 📑 Table of Contents

- [Features](#features)
- [Setup](#setup)
- [How Each File Works](#how-each-file-works)
- [Model Architecture & Parameters](#model-architecture--parameters)
- [Training and Inference Workflow](#training-and-inference-workflow)
- [Tips](#tips)
- [Dataset Source & Credits](#dataset-source--credits)

---

## 🚀 Features

- **Easy EC2/cloud setup**
- **Academic Torrents ImageNet download**
- Modular: `config.py`, `dataset.py`, `model.py`, `train.py`, `predict.py`
- Robust CSV logging and automatic plots per run
- Secure, failure-resistant (use `nohup`)
- Hugging Face deployment instructions included

---

## ⚡ Setup

### 1. Launch an EC2 GPU Instance

- **AMI:** AWS Deep Learning AMI (Ubuntu, CUDA)
- **Instance:** `g4dn.xlarge` or better (100GB+ SSD recommended)

### 2. Connect & System Prep

```bash
ssh -i "/path/to/your/key.pem" ubuntu@<ec2-public-dns>
sudo apt-get update
sudo apt-get install aria2 python3-venv python3-pip -y
```

### 3. Download ImageNet (Academic Torrents)

```bash
aria2c http://academictorrents.com/download/a306397ccf9c2ead27155983c254227c0fd938e2.torrent
aria2c http://academictorrents.com/download/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.torrent
# Unpack to:
# ~/ERA_Session9_MiniCapstone/IMAGENET/TRAIN/
# ~/ERA_Session9_MiniCapstone/IMAGENET/VAL/
```

### 4. Clone This Repo

```bash
git clone https://github.com/Remo-Consultants/ERA_Session9_MiniCapstone.git
cd ERA_Session9_MiniCapstone
```

### 5. Environment & Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6. Data Path Config

Edit `config.py` to:
```python
DATA_PATH = os.path.expanduser('~/ERA_Session9_MiniCapstone/IMAGENET/')
```

---

## 🗂️ How Each File Works

| File           | Description                                                                                                                      |
|----------------|----------------------------------------------------------------------------------------------------------------------------------|
| `config.py`    | Central control: data paths, hyperparameters, output directories, device selection.                                              |
| `dataset.py`   | Scans your data folders, applies ImageNet-specific transforms, and creates efficient PyTorch `DataLoaders`.                      |
| `model.py`     | Constructs a ResNet50 (from torchvision) with customizable output layer and parameter counting for compliance demonstration.     |
| `train.py`     | Loads configs/data/model, implements the train/val loop, **automatically logs** every epoch and produces loss/accuracy plots.    |
| `predict.py`   | Loads a trained model and predicts classes for new images, using a class-name mapping.                                           |
| `requirements.txt` | All required Python packages.                                                                                                 |
| `logs/`        | Training logs (.csv), plots (`training_plot.png`, `lr_plot.png`).                                                               |
| `models/`      | Saved model files (e.g., `best_model_*.pth`).                                                                                    |

---

## 🏗️ Model Architecture & Parameters

- Core: [torchvision.models.resnet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)  
- No pretrained weights, 1000 output classes (full ImageNet)
- Parameter count shown at training start

**Key config values:**  
- `DATA_PATH` (`config.py`): must point to a folder with `TRAIN/` and `VAL/`
- `MODEL_NAME`, `NUM_CLASSES`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, `STEP_LR_GAMMA`, `STEP_LR_STEP_SIZE`, `DEVICE`

---

## 🔨 Training and Inference Workflow

1. **Set config:** Edit `config.py` for paths, batch, learning rate, etc.
2. **Start training:**  
   ```bash
   nohup python train.py > logs/train_nohup.out 2>&1 &
   ```
3. **Monitor logs:**  
   ```bash
   tail -f logs/train_nohup.out
   ```
4. **Run prediction:**  
   ```bash
   nohup python predict.py --image path_to_image.jpg > logs/predict_nohup.out 2>&1 &
   ```
5. **Outputs:**  
   - Models in `models/`
   - Logs and training/validation plots in `logs/`

---

## 💡 Tips

- **Resume & manage jobs:** `ps aux | grep python` and `kill <pid>`
- **Harden against SSH disconnects:** Use `nohup`, `tmux`, or `screen`
- **Visualize progress:** See `logs/training_plot.png` & `logs/lr_plot.png`
- **Compliance:** Always check that your dataset structure matches the expected directory format.
  
---

## 📦 Dataset Source & Credits

- **ImageNet (ILSVRC2012)** via [Academic Torrents](http://academictorrents.com/collection/imagenet)  
- Please ensure your use complies with the [ImageNet License Terms](https://www.image-net.org/download)

---

## 📚 Further Resources
- [Project Home](https://github.com/Remo-Consultants/ERA_Session9_MiniCapstone)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision](https://pytorch.org/vision/stable/index.html)

---

*Happy Training! 🚀*

