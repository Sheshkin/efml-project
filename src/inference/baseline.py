import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.metrics import compute_all_metrics
from src.benchmark.runner import run_full_benchmark, model_size_mb
from src.utils.logging_utils import get_logger, save_json


def evaluate_model(model, data_loader, device, precision="fp32"):
    model.eval()
    all_dice, all_iou, all_acc = [], [], []

    with torch.no_grad():
        for imgs, masks in tqdm(data_loader, desc="Evaluating", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)
            if precision == "fp16":
                imgs = imgs.half()
                with torch.autocast(device_type=device if device != "mps" else "cpu",
                                    dtype=torch.float16, enabled=(device != "mps")):
                    preds = model(imgs)
            else:
                preds = model(imgs)

            preds = preds.float()
            m = compute_all_metrics(preds, masks)
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_acc.append(m["pixel_accuracy"])

    return {
        "dice": sum(all_dice) / len(all_dice),
        "iou": sum(all_iou) / len(all_iou),
        "pixel_accuracy": sum(all_acc) / len(all_acc),
    }


def run_baseline_benchmark(model, test_loader, device, precision="fp16", warmup=20,
                           latency_runs=100, batch_size_throughput=16,
                           input_size=(256, 256), output_dir="results"):
    logger = get_logger("baseline_benchmark")
    logger.info(f"Running baseline benchmark | device={device} | precision={precision}")

    model.eval()
    model.to(device)

    if precision == "fp16" and device in ("mps", "cuda"):
        model = model.half()
    else:
        precision = "fp32"
        model = model.float()

    logger.info("Evaluating quality...")
    quality = evaluate_model(model, test_loader, device, precision)
    logger.info(f"Dice={quality['dice']:.4f} | IoU={quality['iou']:.4f} | PixAcc={quality['pixel_accuracy']:.4f}")

    logger.info("Running latency benchmark (bs=1)...")
    benchmark = run_full_benchmark(
        model_fn=lambda x: model(x),
        data_loader=test_loader,
        device=device,
        precision=precision,
        warmup=warmup,
        latency_runs=latency_runs,
        batch_size_throughput=batch_size_throughput,
        input_size=input_size,
    )

    result = {
        "method": "Baseline FP16" if precision == "fp16" else "Baseline FP32",
        "precision": precision,
        "device": device,
        "model_size_mb": model_size_mb(model),
        "p50_latency_ms": benchmark["latency"]["median_ms"],
        "p95_latency_ms": benchmark["latency"]["p95_ms"],
        "mean_latency_ms": benchmark["latency"]["mean_ms"],
        "throughput_img_sec": benchmark["throughput"]["throughput_img_sec"],
        "batch_size": batch_size_throughput,
        **quality,
        **{f"memory_{k}": v for k, v in benchmark["memory"].items()},
    }

    save_json(result, os.path.join(output_dir, "logs", "baseline_result.json"))
    logger.info(f"Baseline result: {json.dumps(result, indent=2)}")
    return result
