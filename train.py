import gc
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import CFG
from src.dataset import CustomDataset, get_train_valid_dataset, get_transforms
from src.metrics import calc_cv
from src.model import build_model
from src.sheduler import get_scheduler, scheduler_step
from src.utils import AverageMeter, cfg_init, init_logger


def criterion(y_pred, y_true):
    DiceLoss = smp.losses.DiceLoss(mode='binary')
    BCELoss = smp.losses.SoftBCEWithLogitsLoss(
        smooth_factor=0.01, pos_weight=torch.tensor([0.5]).to(device)
    )

    alpha = 0.5
    beta = 1 - alpha
    TverskyLoss = smp.losses.TverskyLoss(
        mode='binary', log_loss=False, alpha=alpha, beta=beta)
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)


def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()

    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with autocast(CFG.use_amp):
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        pbar.set_postfix({'loss': loss.item()})

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt):
    mask_pred = np.zeros(valid_mask_gt.shape)
    mask_count = np.zeros(valid_mask_gt.shape)

    model.eval()
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(images)
            loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # make whole mask
        y_preds = torch.sigmoid(y_preds).to('cpu').numpy()
        start_idx = step * CFG.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

    print(f'mask_count_min: {mask_count.min()}')
    mask_pred /= mask_count
    return losses.avg, mask_pred


if __name__ == '__main__':
    cfg_init(CFG)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Logger = init_logger(log_file=CFG.log_path)

    Logger.info('\n\n-------- exp_info -----------------')

    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset(CFG)

    train_dataset = CustomDataset(
        train_images, CFG, labels=train_masks,
        transform=get_transforms(data='train', cfg=CFG)
    )
    valid_dataset = CustomDataset(
        valid_images, CFG, labels=valid_masks,
        transform=get_transforms(data='valid', cfg=CFG)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = build_model(CFG)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler = get_scheduler(CFG, optimizer)

    fragment_id = CFG.valid_id

    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

    # train loop
    fold = CFG.valid_id

    best_loss = np.inf
    best_score = np.inf if CFG.metric_direction == 'minimize' else -1

    for epoch in range(CFG.epochs):
        gc.collect()

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, device)

        # eval
        avg_val_loss, mask_pred = valid_fn(
            valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt)

        scheduler_step(scheduler, avg_val_loss, epoch)

        best_dice, best_th = calc_cv(valid_mask_gt, mask_pred)

        # score = avg_val_loss
        score = best_dice

        elapsed = time.time() - start_time

        Logger.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        # Logger.info(f'Epoch {epoch+1} - avgScore: {avg_score:.4f}')
        Logger.info(f'Epoch {epoch + 1} - avgScore: {score:.4f}')

        update_best = (score < best_score) if CFG.metric_direction == 'minimize' else (score > best_score)

        if update_best:
            best_loss = avg_val_loss
            best_score = score

            Logger.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            Logger.info(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')

            torch.save(
                {'model': model.state_dict(), 'preds': mask_pred},
                CFG.model_dir + f'{CFG.model_name}_fold{fold}_best.pth'
            )
