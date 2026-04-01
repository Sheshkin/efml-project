import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.utils.metrics import compute_all_metrics
from src.utils.logging_utils import get_logger, save_json


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_w = dice_weight
        self.bce_w = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_prob = torch.sigmoid(pred)
        eps = 1e-6
        intersection = (pred_prob * target).sum(dim=(2, 3))
        dice_loss = 1.0 - (2 * intersection + eps) / (
            pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
        )
        dice_loss = dice_loss.mean()

        return self.dice_w * dice_loss + self.bce_w * bce_loss


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        lr=1e-4,
        weight_decay=1e-5,
        checkpoint_dir="results/checkpoints",
        log_dir="results/logs",
        experiment_name="finetune",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.criterion = DiceBCELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self.logger = get_logger(f"trainer.{experiment_name}", log_dir=log_dir)
        self.history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
        self.best_dice = 0.0

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(self.train_loader, desc="Train", leave=False):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(imgs)
            loss = self.criterion(preds, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_dice, all_iou, all_acc = [], [], []

        for imgs, masks in tqdm(self.val_loader, desc="Val", leave=False):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            preds = self.model(imgs)
            loss = self.criterion(preds, masks)
            total_loss += loss.item() * imgs.size(0)

            m = compute_all_metrics(preds, masks)
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_acc.append(m["pixel_accuracy"])

        n = len(self.val_loader.dataset)
        return {
            "val_loss": total_loss / n,
            "val_dice": sum(all_dice) / len(all_dice),
            "val_iou": sum(all_iou) / len(all_iou),
            "val_pixel_acc": sum(all_acc) / len(all_acc),
        }

    def fit(self, epochs):
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")
        self.logger.info(f"Parameters: {self.model.count_parameters():,}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_dice"].append(val_metrics["val_dice"])
            self.history["val_iou"].append(val_metrics["val_iou"])

            self.logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"dice={val_metrics['val_dice']:.4f} | "
                f"iou={val_metrics['val_iou']:.4f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

            if val_metrics["val_dice"] > self.best_dice:
                self.best_dice = val_metrics["val_dice"]
                self.save_checkpoint("best_model.pt", epoch, val_metrics)
                self.logger.info(f"  → New best Dice: {self.best_dice:.4f} (saved)")

        self.save_checkpoint("final_model.pt", epochs, val_metrics)
        save_json(self.history, f"results/logs/{self.experiment_name}_history.json")
        self.logger.info(f"Training complete. Best Dice: {self.best_dice:.4f}")
        return self.history

    def save_checkpoint(self, filename, epoch, metrics):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_dice": self.best_dice,
        }, path)

    def load_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_dice = ckpt.get("best_dice", 0.0)
        self.logger.info(f"Loaded checkpoint from {path}, best_dice={self.best_dice:.4f}")
        return ckpt
