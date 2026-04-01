import torch


def dice_coefficient(pred, target, threshold=0.5, eps=1e-6):
    pred = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = (pred_bin + target - pred_bin * target).sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def pixel_accuracy(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    pred_bin = (pred > threshold).float()
    return (pred_bin == target).float().mean().item()


def compute_all_metrics(pred, target, threshold=0.5):
    return {
        "dice": dice_coefficient(pred, target, threshold),
        "iou": iou_score(pred, target, threshold),
        "pixel_accuracy": pixel_accuracy(pred, target, threshold),
    }
