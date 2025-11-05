# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

from config import DEVICE, LEARNING_RATE, NUM_EPOCHS, STEP_LR_GAMMA, STEP_LR_STEP_SIZE, MODEL_SAVE_PATH, LOG_DIR
from model import build_resnet50_standard, count_parameters
from dataset import get_dataloaders # Will pass data_path and batch_size here

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Ensure save paths exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Prepare logging
    log_path = os.path.join(LOG_DIR, 'training_log.csv')
    csv_log = open(log_path, 'w', newline='')
    log_writer = csv.writer(csv_log)
    log_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate']) 

    # For plotting
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, lr_hist = [], [], [], [], []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_stats = {}
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            processed_samples = 0

            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase], desc=f"Batch Progress ({phase})", leave=False)):
                if inputs is None or labels is None:
                    print(f"Skipping a batch in {phase} due to None values (image loading error).")
                    continue

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                processed_samples += inputs.size(0)

            if phase == 'train':
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Learning Rate: {current_lr:.6f}')
            else:
                current_lr = optimizer.param_groups[0]['lr']

            epoch_loss = running_loss / processed_samples if processed_samples > 0 else 0.0
            epoch_acc = running_corrects.double() / processed_samples if processed_samples > 0 else 0.0

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            epoch_stats[f"{phase}_loss"] = epoch_loss
            epoch_stats[f"{phase}_acc"] = epoch_acc

            if phase == 'train':
                train_loss_hist.append(epoch_loss)
                train_acc_hist.append(epoch_acc.item())
                lr_hist.append(current_lr)
            else:
                val_loss_hist.append(epoch_loss)
                val_acc_hist.append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'best_model_{best_acc:.4f}.pth'))

        # Write to log per epoch [after both phases]
        log_writer.writerow([
            epoch,
            epoch_stats.get('train_loss', 0),
            epoch_stats.get('train_acc', 0),
            epoch_stats.get('val_loss', 0),
            epoch_stats.get('val_acc', 0),
            current_lr,
        ])
        csv_log.flush()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Save training curves
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    x_range = np.arange(len(train_loss_hist))
    axs[0].plot(x_range, train_loss_hist, label='Train Loss')
    axs[0].plot(x_range, val_loss_hist, label='Val Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(x_range, train_acc_hist, label='Train Acc')
    axs[1].plot(x_range, val_acc_hist, label='Val Acc')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plot_file = os.path.join(LOG_DIR, 'training_plot.png')
    plt.savefig(plot_file)
    print(f"Training and validation curves saved to {plot_file}")

    # Optional: plot LR schedule
    plt.figure()
    plt.plot(x_range, lr_hist, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.savefig(os.path.join(LOG_DIR, 'lr_plot.png'))
    print(f"LR schedule saved to {os.path.join(LOG_DIR, 'lr_plot.png')}")

    csv_log.close()

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    from config import DATA_PATH, BATCH_SIZE, NUM_CLASSES, MODEL_NAME

    # 1. Prepare Data
    print("Preparing data...")
    dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_PATH, BATCH_SIZE)
    print("Data loading complete.")
    print("Dataset sizes:", dataset_sizes)

    # 2. Build Model
    print("Building model...")
    model = build_resnet50_standard(NUM_CLASSES, DEVICE)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")
    if total_params > 150000:
        print("WARNING: Parameter count exceeds 150,000!")
    print("Model architecture:")
    print(model)

    # 3. Define Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA)

    # 4. Train Model (with epoch-by-epoch logs/plotting)
    print("\nStarting training...")
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, NUM_EPOCHS)
    print("Training complete.")

    # Save final model
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'{MODEL_NAME}_final.pth'))
    print(f"Final model saved to {os.path.join(MODEL_SAVE_PATH, f'{MODEL_NAME}_final.pth')}")