import os
import time
import json
import torch
import torch.nn as nn
from src.inference.baseline import evaluate_model
from src.benchmark.runner import run_full_benchmark, model_size_mb
from src.utils.logging_utils import get_logger, save_json

def run_compile_benchmark(model, test_loader, device, precision='fp16', compile_mode='default', warmup=20, latency_runs=100, batch_size_throughput=16, input_size=(256, 256), output_dir='results'):
    logger = get_logger('compile_benchmark')
    logger.info(f'Running torch.compile benchmark | mode={compile_mode} | device={device}')
    model.eval()
    model.to(device)
    if precision == 'fp16' and device in ('mps', 'cuda'):
        model = model.half()
    else:
        precision = 'fp32'
        model = model.float()
    logger.info(f"Compiling model with mode='{compile_mode}'...")
    t_compile_start = time.perf_counter()
    safe_mode = compile_mode
    if device not in ('cuda',) and compile_mode == 'max-autotune':
        safe_mode = 'default'
        logger.warning("max-autotune requires CUDA; falling back to 'default'")
    compiled_model = torch.compile(model, mode=safe_mode, fullgraph=False)
    dummy = torch.randn(1, 3, *input_size, device=device)
    if precision == 'fp16':
        dummy = dummy.half()
    with torch.no_grad():
        _ = compiled_model(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    compile_time_s = time.perf_counter() - t_compile_start
    logger.info(f'Compile + first inference time: {compile_time_s:.2f}s')
    logger.info('Evaluating quality...')
    quality = evaluate_model(compiled_model, test_loader, device, precision)
    logger.info(f"Dice={quality['dice']:.4f} | IoU={quality['iou']:.4f}")
    benchmark = run_full_benchmark(model_fn=lambda x: compiled_model(x), data_loader=test_loader, device=device, precision=precision, warmup=warmup, latency_runs=latency_runs, batch_size_throughput=batch_size_throughput, input_size=input_size)
    result = {'method': f'torch.compile ({safe_mode})', 'precision': precision, 'device': device, 'runtime_device': device, 'compile_mode': safe_mode, 'compile_time_s': compile_time_s, 'model_size_mb': model_size_mb(model), 'p50_latency_ms': benchmark['latency']['median_ms'], 'p95_latency_ms': benchmark['latency']['p95_ms'], 'mean_latency_ms': benchmark['latency']['mean_ms'], 'throughput_img_sec': benchmark['throughput']['throughput_img_sec'], 'batch_size': batch_size_throughput, **quality, **{f'memory_{k}': v for (k, v) in benchmark['memory'].items()}, 'status': 'ok'}
    save_json(result, os.path.join(output_dir, 'logs', 'compile_result.json'))
    logger.info(f'Compile result: {json.dumps(result, indent=2)}')
    return result
