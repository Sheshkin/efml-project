import os
import time
from typing import Dict, Iterable, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from src.utils.logging_utils import get_logger, save_json
from src.utils.metrics import compute_all_metrics

def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError('onnxruntime is required for ONNX backend. Install requirements first.') from exc
    return ort

def providers_for_device(device, ort):
    available = set(ort.get_available_providers())
    if device == 'cuda' and 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']

def create_ort_session(onnx_path, device, logger=None):
    ort = _import_onnxruntime()
    providers = providers_for_device(device, ort)
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        return (session, providers)
    except Exception as exc:
        if providers != ['CPUExecutionProvider']:
            if logger:
                logger.warning('Failed to initialize provider %s (%s). Falling back to CPUExecutionProvider.', providers, exc)
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            return (session, ['CPUExecutionProvider'])
        raise

def export_to_onnx(model, onnx_path, input_size=(256, 256), precision='fp32', opset_version=17, dynamic_batch=True):
    model = model.eval().cpu()
    if precision == 'fp16':
        model = model.half()
    (h, w) = input_size
    dummy = torch.randn(1, 3, h, w, device='cpu')
    if precision == 'fp16':
        dummy = dummy.half()
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    torch.onnx.export(model, dummy, onnx_path, export_params=True, opset_version=opset_version, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes)

def _ort_input_dtype(session):
    input_type = session.get_inputs()[0].type
    if 'float16' in input_type:
        return np.float16
    return np.float32

def _evaluate_session(session, data_loader, logger=None):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    in_dtype = _ort_input_dtype(session)
    (all_dice, all_iou, all_acc) = ([], [], [])
    for (imgs, masks) in tqdm(data_loader, desc='ONNX eval', leave=False):
        x = imgs.numpy().astype(in_dtype, copy=False)
        preds = session.run([output_name], {input_name: x})[0]
        preds_t = torch.from_numpy(preds).float()
        m = compute_all_metrics(preds_t, masks.float())
        all_dice.append(m['dice'])
        all_iou.append(m['iou'])
        all_acc.append(m['pixel_accuracy'])
    quality = {'dice': sum(all_dice) / len(all_dice), 'iou': sum(all_iou) / len(all_iou), 'pixel_accuracy': sum(all_acc) / len(all_acc)}
    if logger:
        logger.info('ONNX quality | Dice=%.4f IoU=%.4f PixAcc=%.4f', quality['dice'], quality['iou'], quality['pixel_accuracy'])
    return quality

def _benchmark_session(session, input_size=(256, 256), warmup=20, latency_runs=100, batch_size_throughput=16):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    in_dtype = _ort_input_dtype(session)
    (h, w) = input_size
    x_single = np.random.randn(1, 3, h, w).astype(in_dtype)
    for _ in range(warmup):
        _ = session.run([output_name], {input_name: x_single})
    latencies_ms = []
    for _ in range(latency_runs):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: x_single})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
    x_batch = np.random.randn(batch_size_throughput, 3, h, w).astype(in_dtype)
    thr_warmup = max(3, warmup // 4)
    thr_runs = 20
    for _ in range(thr_warmup):
        _ = session.run([output_name], {input_name: x_batch})
    t0 = time.perf_counter()
    for _ in range(thr_runs):
        _ = session.run([output_name], {input_name: x_batch})
    elapsed = time.perf_counter() - t0
    throughput = batch_size_throughput * thr_runs / elapsed
    return {'mean_latency_ms': float(np.mean(latencies_ms)), 'p50_latency_ms': float(np.percentile(latencies_ms, 50)), 'p95_latency_ms': float(np.percentile(latencies_ms, 95)), 'throughput_img_sec': float(throughput), 'batch_size': int(batch_size_throughput), 'benchmark_runs': int(latency_runs)}

def onnx_model_size_mb(onnx_path):
    return os.path.getsize(onnx_path) / 1000000.0

def run_onnx_benchmark(model, test_loader, device, output_dir='results', precision='fp32', warmup=20, latency_runs=100, batch_size_throughput=16, input_size=(256, 256), onnx_path=None, method_name='ONNXRuntime', result_filename='backend_result.json', export_if_missing=True):
    logger = get_logger('onnx_benchmark')
    if onnx_path is None:
        onnx_path = os.path.join(output_dir, 'artifacts', 'onnx', 'unet_backend.onnx')
    if export_if_missing or not os.path.exists(onnx_path):
        logger.info('Exporting ONNX model -> %s', onnx_path)
        export_to_onnx(model, onnx_path=onnx_path, input_size=input_size, precision=precision)
    logger.info('Creating ONNX Runtime session (device=%s)', device)
    (session, providers) = create_ort_session(onnx_path, device=device, logger=logger)
    logger.info('Evaluating ONNX model quality...')
    quality = _evaluate_session(session, test_loader, logger=logger)
    logger.info('Running ONNX benchmark...')
    perf = _benchmark_session(session, input_size=input_size, warmup=warmup, latency_runs=latency_runs, batch_size_throughput=batch_size_throughput)
    runtime_device = 'cuda' if providers and providers[0] == 'CUDAExecutionProvider' else 'cpu'
    result = {'method': method_name, 'precision': precision, 'device': device, 'runtime_device': runtime_device, 'providers': providers, 'onnx_path': onnx_path, 'model_size_mb': onnx_model_size_mb(onnx_path), **perf, **quality, 'status': 'ok'}
    save_path = os.path.join(output_dir, 'logs', result_filename)
    save_json(result, save_path)
    logger.info('Saved ONNX result -> %s', save_path)
    return result
