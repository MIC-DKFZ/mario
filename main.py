import json
import os
import sys
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DataClass import MarioData
from utils.augmentations import get_training_augmentation_scheme, get_validation_augmentation_scheme
from utils.settings import freeze_encoder, init_settings, init_optimizer
from utils.utils import set_seed, parse_args, set_environment


def train_step(im_ti: torch.Tensor, im_ti_1: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Performs a single training step.

    Args:
        im_ti (torch.Tensor): Input image at time t.
        im_ti_1 (torch.Tensor): Input image at time t+1.
        label (torch.Tensor): Ground truth label.

    Returns:
        torch.Tensor: Computed loss for the training step.
    """
    optimizer.zero_grad()
    im_ti = im_ti.cuda()
    im_ti_1 = im_ti_1.cuda()
    label = label.cuda()
    with autocast():
        if config.contrastive_pretrained:
            im_ti = im_ti[:, 0, :, :].unsqueeze(1)
            im_ti_1 = im_ti_1[:, 0, :, :].unsqueeze(1)

        out = model(im_ti, im_ti_1)
        loss = criterion(out, label)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss


def val_step(im_ti: torch.Tensor, im_ti_1: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Performs a single validation step.

    Args:
        im_ti (torch.Tensor): Input image at time t.
        im_ti_1 (torch.Tensor): Input image at time t+1.
        label (torch.Tensor): Ground truth label.

    Returns:
        torch.Tensor: Computed loss for the validation step.
    """
    im_ti = im_ti.cuda()
    im_ti_1 = im_ti_1.cuda()
    label = label.cuda()
    if config.contrastive_pretrained:
        im_ti = im_ti[:, 0, :, :].unsqueeze(1)
        im_ti_1 = im_ti_1[:, 0, :, :].unsqueeze(1)
    out = model(im_ti, im_ti_1)
    probabilities = torch.sigmoid(out)
    predictions = (probabilities >= threshold).float()
    all_labels.append(label)
    all_preds.append(predictions)
    loss = criterion(out, label)
    return loss


def log_metrics(val_loss: float, all_labels: List[int], all_preds: List[int]) -> None:
    """
    Logs metrics to W&B and TensorBoard.

    Args:
        val_loss (float): Validation loss.
        all_labels (List[int]): True labels for the dataset.
        all_preds (List[int]): Predicted labels for the dataset.
    """
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    val_loss = val_loss / len(train_loader)
    val_writer.add_scalar('val_Loss', val_loss, e)
    val_writer.add_scalar('precision', precision, e)
    val_writer.add_scalar('recall', recall, e)
    val_writer.add_scalar('f1', f1, e)
    val_writer.add_scalar('f1 micro', f1_micro, e)
    wandb.log({"train_loss": epoch_loss, "val_loss": val_loss, "f1_micro": f1_micro, "f1_macro": f1})


def save_model(model: torch.nn.Module, config: SimpleNamespace, all_labels: List[int], all_preds: List[int]) -> None:
    """
    Saves the best model based on validation loss.

    Args:
        model (torch.nn.Module): Model to be saved.
        config (SimpleNamespace): Configuration parameters.
        all_labels (List[int]): True labels.
        all_preds (List[int]): Predicted labels.
    """
    torch.save(model.state_dict(), os.path.join(results_prefix, str(config.fold), 'best_model.pth'))

    print(f'New best model saved with validation loss: {val_loss:.4f}')

    all_labels_flat = np.argmax(all_labels, axis=1) if len(all_labels[0]) > 1 else all_labels
    all_preds_flat = np.argmax(all_preds, axis=1) if len(all_preds[0]) > 1 else all_preds
    cm = confusion_matrix(all_labels_flat, all_preds_flat)
    cr = classification_report(all_labels_flat, all_preds_flat, output_dict=True)

    # Save
    np.savetxt(os.path.join(results_prefix, str(config.fold), 'best_confusion_matrix.txt'), cm, fmt='%d')
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(os.path.join(results_prefix, str(config.fold), 'best_classification_report.csv'))


def save_final_model(model: torch.nn.Module, config: SimpleNamespace, all_labels: List[int],
                     all_preds: List[int]) -> None:
    """
    Saves the final model at the end of training.

    Args:
        model (torch.nn.Module): Model to be saved.
        config (SimpleNamespace): Configuration parameters.
        all_labels (List[int]): True labels.
        all_preds (List[int]): Predicted labels.
    """
    torch.save(model.state_dict(), os.path.join(results_prefix, str(config.fold), 'final_model.pth'))
    print(f'Final validation loss: {val_loss:.4f}')
    all_labels_flat = np.argmax(all_labels, axis=1) if len(all_labels[0]) > 1 else all_labels
    all_preds_flat = np.argmax(all_preds, axis=1) if len(all_preds[0]) > 1 else all_preds
    cm = confusion_matrix(all_labels_flat, all_preds_flat)
    cr = classification_report(all_labels_flat, all_preds_flat, output_dict=True)

    # Save
    np.savetxt(os.path.join(results_prefix, str(config.fold), 'final_confusion_matrix.txt'), cm, fmt='%d')
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(os.path.join(results_prefix, str(config.fold), 'final_classification_report.csv'))

    # Print metrics nicely
    print(f'Confusion Matrix:\n{cm}')
    print(f'Classification Report:\n{pd.DataFrame(cr).transpose()}')
    f1_value = cr['weighted avg']['f1-score']
    print(f'F1 Score: {f1_value:.4f}')


if __name__ == '__main__':
    # Set the seed
    seed = 42
    set_seed(seed)

    args = parse_args(sys.argv[1:])
    exp = args.experiment

    data_prefix, results_prefix = set_environment()

    config = SimpleNamespace(**vars(args))

    wandb.init(
        project="mario-wandb",
        config=vars(config)
    )

    data_path = os.path.join(data_prefix, 'train_npy/train')
    # When training on whole dataset (also validation split, use splits_train_all.json
    splits = json.load(open('splits.json'))
    frame = pd.read_csv(os.path.join(data_prefix, 'df_task1_all_train.csv'))
    fold_num = config.fold
    fold = splits['Fold_' + str(fold_num)]
    train_cases = fold['Train']
    test_cases = fold['Test']

    train_writer = SummaryWriter(os.path.join(results_prefix, str(config.fold)))
    val_writer = SummaryWriter(os.path.join(results_prefix, str(config.fold)))

    train_transforms = get_training_augmentation_scheme(config)
    val_transforms = get_validation_augmentation_scheme(config)

    train_class = MarioData(train_cases, frame, data_path, transform=train_transforms)
    val_class = MarioData(test_cases, frame, data_path, transform=val_transforms)

    train_loader = DataLoader(train_class, batch_size=config.batch_size, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(val_class, batch_size=48, shuffle=False, drop_last=True)

    model, optimizer, scheduler, class_weights, criterion = init_settings(config, train_loader)

    best_val_loss = float('inf')
    scaler = GradScaler()

    freeze_encoder(model)

    # Start training
    for e in range(0, config.num_epochs):
        if e == config.freeze_epoch:
            # After Warmup, unfreeze model and init new optimizer
            for param in model.encoder.parameters():
                param.requires_grad = True

            optimizer, scheduler = init_optimizer(config)

        epoch_loss = 0.0
        model = model.train()

        for im_ti, im_ti_1, label in tqdm(train_loader):
            loss = train_step(im_ti, im_ti_1, label)
            epoch_loss += loss.item()

        # one epoch finished
        epoch_loss = epoch_loss / len(train_loader)
        train_writer.add_scalar('Loss', epoch_loss, e)
        print(f'Epoch [{e + 1}/{config.num_epochs}], Loss: {epoch_loss:.4f}')
        print(f'Current learning rate: {scheduler.get_last_lr()}')

        # Validation
        val_loss = 0.0
        all_labels = []
        all_preds = []

        threshold = 0.5
        with torch.no_grad():

            model = model.eval()

            for im_ti, im_ti_1, label in tqdm(test_loader):
                loss = val_step(im_ti, im_ti_1, label)
                val_loss += loss.item()

            all_labels = torch.cat(all_labels).cpu().numpy()
            all_preds = torch.cat(all_preds).cpu().numpy()

        # Step the scheduler
        scheduler.step(val_loss)

        log_metrics(val_loss, all_labels, all_preds)

        # Save the model if the validation loss is the best we've seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, config, all_labels, all_preds)

    print("Training done")
    save_final_model(model, config, all_labels, all_preds)

    train_writer.close()
    val_writer.close()
