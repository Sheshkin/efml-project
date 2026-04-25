import copy
import os
from typing import Dict, Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm
from src.benchmark.runner import model_size_mb, run_full_benchmark
from src.inference.baseline import evaluate_model
from src.training.trainer import DiceBCELoss
from src.utils.logging_utils import get_logger, save_json

def _prunable_modules(model):
    modules = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, 'weight'):
            modules.append(m)
    return modules

def compute_weight_sparsity(model):
    total = 0
    zeros = 0
    for m in _prunable_modules(model):
        w = m.weight.detach()
        total += w.numel()
        zeros += (w == 0).sum().item()
    if total == 0:
        return 0.0
    return float(zeros / total)

def _finetune_with_optional_masks(model, train_loader, device, epochs, lr, weight_decay, max_batches=None, param_masks=None, logger=None):
    if epochs <= 0:
        return
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        seen = 0
        for (step, (imgs, masks)) in enumerate(tqdm(train_loader, desc=f'Finetune {epoch}/{epochs}', leave=False)):
            if max_batches is not None and step >= max_batches:
                break
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if param_masks:
                with torch.no_grad():
                    for (param, mask) in param_masks:
                        param.mul_(mask)
            total_loss += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        mean_loss = total_loss / max(1, seen)
        if logger:
            logger.info('Finetune epoch %d/%d | loss=%.4f', epoch, epochs, mean_loss)

def run_unstructured_pruning_benchmark(model, train_loader, test_loader, device, output_dir='results', precision='fp16', prune_amount=0.5, finetune_epochs=2, finetune_lr=5e-05, finetune_weight_decay=1e-05, max_finetune_batches=None, warmup=20, latency_runs=100, batch_size_throughput=16, input_size=(256, 256)):
    logger = get_logger('prune_benchmark')
    logger.info('Running unstructured pruning | amount=%.2f | finetune_epochs=%d', prune_amount, finetune_epochs)
    model = copy.deepcopy(model).float().to(device).eval()
    modules = _prunable_modules(model)
    params_to_prune = [(m, 'weight') for m in modules]
    prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=prune_amount)
    _finetune_with_optional_masks(model=model, train_loader=train_loader, device=device, epochs=finetune_epochs, lr=finetune_lr, weight_decay=finetune_weight_decay, max_batches=max_finetune_batches, param_masks=None, logger=logger)
    for (m, name) in params_to_prune:
        prune.remove(m, name)
    sparsity = compute_weight_sparsity(model)
    model.eval().to(device)
    if precision == 'fp16' and device in ('cuda', 'mps'):
        infer_model = model.half()
    else:
        precision = 'fp32'
        infer_model = model.float()
    quality = evaluate_model(infer_model, test_loader, device=device, precision=precision)
    benchmark = run_full_benchmark(model_fn=lambda x: infer_model(x), data_loader=test_loader, device=device, precision=precision, warmup=warmup, latency_runs=latency_runs, batch_size_throughput=batch_size_throughput, input_size=input_size)
    result = {'method': 'Pruning (unstructured)', 'precision': precision, 'device': device, 'runtime_device': device, 'prune_amount': prune_amount, 'weight_sparsity': sparsity, 'model_size_mb': model_size_mb(infer_model), 'effective_model_size_mb': model_size_mb(infer_model) * (1.0 - sparsity), 'p50_latency_ms': benchmark['latency']['median_ms'], 'p95_latency_ms': benchmark['latency']['p95_ms'], 'mean_latency_ms': benchmark['latency']['mean_ms'], 'throughput_img_sec': benchmark['throughput']['throughput_img_sec'], 'batch_size': batch_size_throughput, **quality, **{f'memory_{k}': v for (k, v) in benchmark['memory'].items()}, 'status': 'ok'}
    save_path = os.path.join(output_dir, 'logs', 'prune_result.json')
    save_json(result, save_path)
    logger.info('Saved prune result -> %s', save_path)
    return result

def _mask_2_4(weight):
    flat = weight.detach().abs().flatten()
    pad = (4 - flat.numel() % 4) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    groups = flat.view(-1, 4)
    keep_idx = torch.topk(groups, k=2, dim=1, largest=True).indices
    mask_groups = torch.zeros_like(groups)
    mask_groups.scatter_(1, keep_idx, 1.0)
    mask_flat = mask_groups.reshape(-1)
    if pad:
        mask_flat = mask_flat[:-pad]
    return mask_flat.view_as(weight).to(weight.dtype)

def run_semi_structured_2_4_benchmark(model, train_loader, test_loader, device, output_dir='results', precision='fp16', finetune_epochs=2, finetune_lr=5e-05, finetune_weight_decay=1e-05, max_finetune_batches=None, warmup=20, latency_runs=100, batch_size_throughput=16, input_size=(256, 256)):
    logger = get_logger('sparsity24_benchmark')
    logger.info('Running 2:4 semi-structured masking + finetuning')
    model = copy.deepcopy(model).float().to(device).train()
    modules = _prunable_modules(model)
    param_masks = []
    with torch.no_grad():
        for m in modules:
            mask = _mask_2_4(m.weight)
            m.weight.mul_(mask)
            param_masks.append((m.weight, mask))
    _finetune_with_optional_masks(model=model, train_loader=train_loader, device=device, epochs=finetune_epochs, lr=finetune_lr, weight_decay=finetune_weight_decay, max_batches=max_finetune_batches, param_masks=param_masks, logger=logger)
    with torch.no_grad():
        for (param, mask) in param_masks:
            param.mul_(mask)
    sparsity = compute_weight_sparsity(model)
    model.eval().to(device)
    if precision == 'fp16' and device in ('cuda', 'mps'):
        infer_model = model.half()
    else:
        precision = 'fp32'
        infer_model = model.float()
    quality = evaluate_model(infer_model, test_loader, device=device, precision=precision)
    benchmark = run_full_benchmark(model_fn=lambda x: infer_model(x), data_loader=test_loader, device=device, precision=precision, warmup=warmup, latency_runs=latency_runs, batch_size_throughput=batch_size_throughput, input_size=input_size)
    result = {'method': '2:4 Semi-structured', 'precision': precision, 'device': device, 'runtime_device': device, 'weight_sparsity': sparsity, 'model_size_mb': model_size_mb(infer_model), 'effective_model_size_mb': model_size_mb(infer_model) * (1.0 - sparsity), 'p50_latency_ms': benchmark['latency']['median_ms'], 'p95_latency_ms': benchmark['latency']['p95_ms'], 'mean_latency_ms': benchmark['latency']['mean_ms'], 'throughput_img_sec': benchmark['throughput']['throughput_img_sec'], 'batch_size': batch_size_throughput, **quality, **{f'memory_{k}': v for (k, v) in benchmark['memory'].items()}, 'status': 'ok'}
    save_path = os.path.join(output_dir, 'logs', 'sparsity24_result.json')
    save_json(result, save_path)
    logger.info('Saved 2:4 result -> %s', save_path)
    return result
