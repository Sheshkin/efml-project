import os
import time
import numpy as np
import torch
from tqdm import tqdm
from src.inference.onnx_infer import export_to_onnx
from src.utils.logging_utils import get_logger, save_json
from src.utils.metrics import compute_all_metrics

def _import_tvm_stack():
    try:
        import onnx
        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor
    except Exception as exc:
        raise RuntimeError('TVM stack is not available. Install TVM + ONNX build dependencies first.') from exc
    return (onnx, tvm, relay, graph_executor)

def _build_tvm_module(onnx_path, input_shape, device):
    (onnx, tvm, relay, graph_executor) = _import_tvm_stack()
    model = onnx.load(onnx_path)
    (mod, params) = relay.frontend.from_onnx(model, shape={'input': input_shape})
    if device == 'cuda':
        target = tvm.target.Target('cuda')
        dev = tvm.cuda(0)
        if not dev.exist:
            target = tvm.target.Target('llvm')
            dev = tvm.cpu(0)
    else:
        target = tvm.target.Target('llvm')
        dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    module = graph_executor.GraphModule(lib['default'](dev))
    return (module, dev, str(target))

def _run_tvm(module, dev, batch_np):
    import tvm
    module.set_input('input', tvm.nd.array(batch_np, device=dev))
    module.run()
    return module.get_output(0).numpy()

def _eval_tvm(module, dev, data_loader):
    (all_dice, all_iou, all_acc) = ([], [], [])
    for (imgs, masks) in tqdm(data_loader, desc='TVM eval', leave=False):
        batch = imgs.numpy().astype(np.float32, copy=False)
        preds = _run_tvm(module, dev, batch)
        m = compute_all_metrics(torch.from_numpy(preds).float(), masks.float())
        all_dice.append(m['dice'])
        all_iou.append(m['iou'])
        all_acc.append(m['pixel_accuracy'])
    return {'dice': sum(all_dice) / len(all_dice), 'iou': sum(all_iou) / len(all_iou), 'pixel_accuracy': sum(all_acc) / len(all_acc)}

def _bench_tvm(module, dev, input_size, warmup, latency_runs, batch_size_throughput):
    (h, w) = input_size
    x_single = np.random.randn(1, 3, h, w).astype(np.float32)
    for _ in range(warmup):
        _ = _run_tvm(module, dev, x_single)
    lat_ms = []
    for _ in range(latency_runs):
        t0 = time.perf_counter()
        _ = _run_tvm(module, dev, x_single)
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000)
    x_batch = np.random.randn(batch_size_throughput, 3, h, w).astype(np.float32)
    thr_warmup = max(3, warmup // 4)
    runs = 20
    for _ in range(thr_warmup):
        _ = _run_tvm(module, dev, x_batch)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = _run_tvm(module, dev, x_batch)
    elapsed = time.perf_counter() - t0
    return {'mean_latency_ms': float(np.mean(lat_ms)), 'p50_latency_ms': float(np.percentile(lat_ms, 50)), 'p95_latency_ms': float(np.percentile(lat_ms, 95)), 'throughput_img_sec': float(batch_size_throughput * runs / elapsed), 'batch_size': int(batch_size_throughput)}

def run_tvm_benchmark(model, test_loader, device, output_dir='results', input_size=(256, 256), warmup=20, latency_runs=100, batch_size_throughput=16):
    logger = get_logger('tvm_benchmark')
    onnx_path = os.path.join(output_dir, 'artifacts', 'onnx', 'unet_tvm_fp32.onnx')
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    try:
        logger.info('Exporting model to ONNX for TVM -> %s', onnx_path)
        export_to_onnx(model, onnx_path, input_size=input_size, precision='fp32')
        logger.info('Compiling TVM graph...')
        (module, dev, target) = _build_tvm_module(onnx_path, (1, 3, *input_size), device=device)
        quality = _eval_tvm(module, dev, test_loader)
        perf = _bench_tvm(module, dev, input_size=input_size, warmup=warmup, latency_runs=latency_runs, batch_size_throughput=batch_size_throughput)
        result = {'method': 'TVM (Relax)', 'precision': 'fp32', 'device': device, 'runtime_device': device, 'tvm_target': target, 'onnx_path': onnx_path, 'model_size_mb': os.path.getsize(onnx_path) / 1000000.0, **perf, **quality, 'status': 'ok'}
    except Exception as exc:
        logger.warning('Skipping TVM benchmark: %s', exc)
        result = {'method': 'TVM (Relax)', 'precision': 'fp32', 'device': device, 'runtime_device': device, 'onnx_path': onnx_path, 'status': 'skipped', 'error': str(exc)}
    save_path = os.path.join(output_dir, 'logs', 'tvm_result.json')
    save_json(result, save_path)
    logger.info('Saved TVM result -> %s', save_path)
    return result
