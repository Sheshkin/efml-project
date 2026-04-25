import time
import gc
import os
import statistics
import torch
import numpy as np

def _sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

def measure_latency(fn, input_tensor, device, warmup=20, runs=100):
    for _ in range(warmup):
        with torch.no_grad():
            _ = fn(input_tensor)
        _sync(device)
    latencies = []
    for _ in range(runs):
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = fn(input_tensor)
        _sync(device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    return {'mean_ms': statistics.mean(latencies), 'median_ms': statistics.median(latencies), 'p95_ms': float(np.percentile(latencies, 95)), 'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 'min_ms': min(latencies), 'max_ms': max(latencies)}

def measure_throughput(fn, input_tensor, batch_size, device, warmup=5, runs=20):
    for _ in range(warmup):
        with torch.no_grad():
            _ = fn(input_tensor)
        _sync(device)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = fn(input_tensor)
        _sync(device)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    return {'throughput_img_sec': batch_size * runs / elapsed, 'batch_size': batch_size, 'total_runs': runs, 'elapsed_s': elapsed}

def measure_memory(device):
    result = {}
    if device == 'cuda':
        result['allocated_mb'] = torch.cuda.memory_allocated() / 1000000.0
        result['reserved_mb'] = torch.cuda.memory_reserved() / 1000000.0
        result['peak_mb'] = torch.cuda.max_memory_allocated() / 1000000.0
    elif device == 'mps':
        result['allocated_mb'] = torch.mps.current_allocated_memory() / 1000000.0
        result['peak_mb'] = result['allocated_mb']
    else:
        import psutil
        proc = psutil.Process(os.getpid()) if _psutil_available() else None
        result['rss_mb'] = proc.memory_info().rss / 1000000.0 if proc else 0.0
    return result

def _psutil_available():
    try:
        import psutil
        return True
    except ImportError:
        return False

def model_size_mb(model):
    total = sum((p.numel() * p.element_size() for p in model.parameters()))
    return total / 1000000.0

def run_full_benchmark(model_fn, data_loader, device, precision='fp32', warmup=20, latency_runs=100, batch_size_throughput=16, input_size=(256, 256)):
    C = 3
    (H, W) = input_size
    x_single = torch.randn(1, C, H, W, device=device)
    if precision == 'fp16':
        x_single = x_single.half()
    lat = measure_latency(model_fn, x_single, device, warmup=warmup, runs=latency_runs)
    x_batch = torch.randn(batch_size_throughput, C, H, W, device=device)
    if precision == 'fp16':
        x_batch = x_batch.half()
    thr = measure_throughput(model_fn, x_batch, batch_size_throughput, device)
    mem = measure_memory(device)
    return {'latency': lat, 'throughput': thr, 'memory': mem, 'device': device, 'precision': precision, 'input_size': list(input_size)}
