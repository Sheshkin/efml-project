import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from src.utils.logging_utils import get_logger, save_json
from src.utils.metrics import compute_all_metrics


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(
            "onnxruntime is required for ONNX backend. Install requirements first."
        ) from exc
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    return ort


def providers_for_device(device, ort):
    available = set(ort.get_available_providers())
    if device == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _cuda_env_summary(ort):
    cudnn_version = None
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        pass

    return {
        "onnxruntime": getattr(ort, "__version__", "unknown"),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_cudnn": cudnn_version,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ort_available_providers": ort.get_available_providers(),
    }


def _format_cuda_ep_error(ort, requested_providers, exc=None):
    env = _cuda_env_summary(ort)
    parts = [
        "ONNX Runtime CUDAExecutionProvider was requested but could not be used.",
        f"requested_providers={requested_providers}",
        f"env={env}",
    ]
    if exc is not None:
        parts.append(f"original_error={type(exc).__name__}: {exc}")
    return " | ".join(parts)


def create_ort_session(onnx_path, device, logger=None, enable_profiling=False, allow_cpu_fallback=False):
    ort = _import_onnxruntime()
    providers = providers_for_device(device, ort)
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = bool(enable_profiling)

    if logger:
        logger.info("ONNX Runtime available providers: %s", ort.get_available_providers())

    if device == "cuda" and "CUDAExecutionProvider" not in providers:
        message = _format_cuda_ep_error(ort, providers)
        if allow_cpu_fallback:
            if logger:
                logger.warning("%s Falling back to CPUExecutionProvider.", message)
            providers = ["CPUExecutionProvider"]
        else:
            raise RuntimeError(message)

    try:
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )
        return session, session.get_providers()
    except Exception as exc:
        if device == "cuda" and not allow_cpu_fallback:
            raise RuntimeError(_format_cuda_ep_error(ort, providers, exc)) from exc

        if providers != ["CPUExecutionProvider"]:
            if logger:
                logger.warning(
                    "Failed to initialize provider %s (%s). Falling back to CPUExecutionProvider.",
                    providers,
                    exc,
                )
            session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            return session, session.get_providers()
        raise


def export_to_onnx(model, onnx_path, input_size=(256, 256), precision="fp32", opset_version=17, dynamic_batch=True):
    export_device = "cuda" if precision == "fp16" and torch.cuda.is_available() else "cpu"
    model = model.eval().to(export_device)
    if precision == "fp16":
        model = model.half()

    h, w = input_size
    dummy = torch.randn(1, 3, h, w, device=export_device)
    if precision == "fp16":
        dummy = dummy.half()

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "output": {0: "batch"},
        }

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


def _ort_input_dtype(session):
    return _ort_type_to_np(session.get_inputs()[0].type)


def _ort_output_dtype(session):
    return _ort_type_to_np(session.get_outputs()[0].type)


def _ort_type_to_np(type_text):
    if "float16" in type_text:
        return np.float16
    if "double" in type_text:
        return np.float64
    return np.float32


def _np_to_torch_dtype(np_dtype):
    if np_dtype == np.float16:
        return torch.float16
    if np_dtype == np.float64:
        return torch.float64
    return torch.float32


def _evaluate_session(session, data_loader, logger=None):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    in_dtype = _ort_input_dtype(session)

    total_dice, total_iou, total_acc = 0.0, 0.0, 0.0
    total_samples = 0

    for imgs, masks in tqdm(data_loader, desc="ONNX eval", leave=False):
        batch_size = imgs.size(0)
        x = imgs.numpy().astype(in_dtype, copy=False)
        preds = session.run([output_name], {input_name: x})[0]
        preds_t = torch.from_numpy(preds).float()
        m = compute_all_metrics(preds_t, masks.float())
        total_dice += m["dice"] * batch_size
        total_iou += m["iou"] * batch_size
        total_acc += m["pixel_accuracy"] * batch_size
        total_samples += batch_size

    quality = {
        "dice": total_dice / max(1, total_samples),
        "iou": total_iou / max(1, total_samples),
        "pixel_accuracy": total_acc / max(1, total_samples),
        "eval_samples": total_samples,
    }
    if logger:
        logger.info(
            "ONNX quality | Dice=%.4f IoU=%.4f PixAcc=%.4f",
            quality["dice"],
            quality["iou"],
            quality["pixel_accuracy"],
        )
    return quality


def _benchmark_session(session, device="cpu", providers=None, input_size=(256, 256), warmup=20, latency_runs=100, batch_size_throughput=16, logger=None):
    providers = providers or session.get_providers()
    if (
        device == "cuda"
        and "CUDAExecutionProvider" in providers
        and torch.cuda.is_available()
    ):
        try:
            perf = _benchmark_session_cuda_iobinding(
                session,
                input_size=input_size,
                warmup=warmup,
                latency_runs=latency_runs,
                batch_size_throughput=batch_size_throughput,
            )
            perf["io_binding"] = True
            return perf
        except Exception as exc:
            if logger:
                logger.warning("CUDA I/O binding failed; falling back to numpy inputs: %s", exc)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    in_dtype = _ort_input_dtype(session)

    h, w = input_size

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

    throughput = (batch_size_throughput * thr_runs) / elapsed

    return {
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "throughput_img_sec": float(throughput),
        "batch_size": int(batch_size_throughput),
        "benchmark_runs": int(latency_runs),
        "io_binding": False,
    }


def _resolve_output_shape(session, batch_size, input_size):
    shape = session.get_outputs()[0].shape
    if not shape:
        return None

    h, w = input_size
    resolved = []
    for i, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif i == 0:
            resolved.append(batch_size)
        elif i == 2:
            resolved.append(h)
        elif i == 3:
            resolved.append(w)
        else:
            return None
    return tuple(resolved)


def _make_cuda_iobinding_runner(session, input_tensor, output_shape):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_np_dtype = _ort_input_dtype(session)
    output_np_dtype = _ort_output_dtype(session)

    binding = session.io_binding()
    binding.bind_input(
        name=input_name,
        device_type="cuda",
        device_id=input_tensor.device.index or 0,
        element_type=input_np_dtype,
        shape=tuple(input_tensor.shape),
        buffer_ptr=input_tensor.data_ptr(),
    )

    output_tensor = None
    if output_shape is not None:
        output_tensor = torch.empty(
            output_shape,
            device=input_tensor.device,
            dtype=_np_to_torch_dtype(output_np_dtype),
        )
        binding.bind_output(
            name=output_name,
            device_type="cuda",
            device_id=input_tensor.device.index or 0,
            element_type=output_np_dtype,
            shape=tuple(output_tensor.shape),
            buffer_ptr=output_tensor.data_ptr(),
        )
    else:
        binding.bind_output(output_name, "cuda")

    def _run():
        session.run_with_iobinding(binding)

    return _run


def _benchmark_session_cuda_iobinding(session, input_size=(256, 256), warmup=20, latency_runs=100, batch_size_throughput=16):
    in_dtype = _ort_input_dtype(session)
    torch_dtype = _np_to_torch_dtype(in_dtype)
    h, w = input_size

    x_single = torch.randn(1, 3, h, w, device="cuda", dtype=torch_dtype)
    run_single = _make_cuda_iobinding_runner(
        session,
        x_single,
        _resolve_output_shape(session, 1, input_size),
    )

    for _ in range(warmup):
        run_single()
    torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(latency_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_single()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    x_batch = torch.randn(batch_size_throughput, 3, h, w, device="cuda", dtype=torch_dtype)
    run_batch = _make_cuda_iobinding_runner(
        session,
        x_batch,
        _resolve_output_shape(session, batch_size_throughput, input_size),
    )

    thr_warmup = max(3, warmup // 4)
    thr_runs = 20
    for _ in range(thr_warmup):
        run_batch()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(thr_runs):
        run_batch()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "throughput_img_sec": float((batch_size_throughput * thr_runs) / elapsed),
        "batch_size": int(batch_size_throughput),
        "benchmark_runs": int(latency_runs),
    }


def _parse_provider_profile(profile_path):
    if not profile_path or not os.path.exists(profile_path):
        return {}

    try:
        with open(profile_path, "r") as f:
            events = json.load(f)
    except Exception:
        return {}
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    counts = {}
    for event in events:
        args = event.get("args", {})
        provider = args.get("provider")
        if provider:
            counts[provider] = counts.get(provider, 0) + 1
    return counts


def _profile_execution_providers(onnx_path, device, input_size, logger=None, require_cuda=False):
    try:
        session, providers = create_ort_session(
            onnx_path,
            device=device,
            logger=logger,
            enable_profiling=True,
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        in_dtype = _ort_input_dtype(session)
        h, w = input_size
        x = np.random.randn(1, 3, h, w).astype(in_dtype)
        session.run([output_name], {input_name: x})
        profile_path = session.end_profiling()
        return _parse_provider_profile(profile_path), providers
    except Exception as exc:
        if require_cuda:
            raise
        if logger:
            logger.warning("Unable to profile ONNX Runtime execution providers: %s", exc)
        return {}, []


def _assert_cuda_execution(onnx_path, device, input_size, logger=None):
    if device != "cuda":
        return {}, []

    provider_counts, profiled_providers = _profile_execution_providers(
        onnx_path,
        device=device,
        input_size=input_size,
        logger=logger,
        require_cuda=True,
    )
    cuda_ops = int(provider_counts.get("CUDAExecutionProvider", 0) or 0)
    if cuda_ops == 0:
        raise RuntimeError(
            "ONNX Runtime CUDAExecutionProvider is available, but a profiled test "
            "inference executed zero CUDA nodes. "
            f"profiled_providers={profiled_providers}, provider_counts={provider_counts}. "
            "This usually means the model was assigned to CPUExecutionProvider or CUDA EP "
            "failed after session creation."
        )

    if logger:
        logger.info("Verified ONNX Runtime CUDA execution: %s", provider_counts)
    return provider_counts, profiled_providers


def _runtime_device_from_profile(session_providers, provider_counts):
    cuda_count = provider_counts.get("CUDAExecutionProvider", 0)
    cpu_count = provider_counts.get("CPUExecutionProvider", 0)
    if cuda_count and cpu_count:
        return "mixed"
    if cuda_count:
        return "cuda"
    if cpu_count:
        return "cpu"
    if session_providers and session_providers[0] == "CUDAExecutionProvider":
        return "cuda_unverified"
    return "cpu"


def onnx_model_size_mb(onnx_path):
    return os.path.getsize(onnx_path) / 1e6


def run_onnx_benchmark(model, test_loader, device, output_dir="results", precision="fp32", warmup=20, latency_runs=100, batch_size_throughput=16, input_size=(256, 256), onnx_path=None, method_name="ONNXRuntime", result_filename="backend_result.json", export_if_missing=True):
    logger = get_logger("onnx_benchmark")

    if onnx_path is None:
        onnx_path = os.path.join(output_dir, "artifacts", "onnx", "unet_backend.onnx")

    if export_if_missing or not os.path.exists(onnx_path):
        logger.info("Exporting ONNX model -> %s", onnx_path)
        export_to_onnx(
            model,
            onnx_path=onnx_path,
            input_size=input_size,
            precision=precision,
        )

    logger.info("Creating ONNX Runtime session (device=%s)", device)
    session, providers = create_ort_session(onnx_path, device=device, logger=logger)
    provider_counts, profiled_providers = _assert_cuda_execution(
        onnx_path,
        device=device,
        input_size=input_size,
        logger=logger,
    )

    logger.info("Evaluating ONNX model quality...")
    quality = _evaluate_session(session, test_loader, logger=logger)

    logger.info("Running ONNX benchmark...")
    perf = _benchmark_session(
        session,
        device=device,
        providers=providers,
        input_size=input_size,
        warmup=warmup,
        latency_runs=latency_runs,
        batch_size_throughput=batch_size_throughput,
        logger=logger,
    )

    if not provider_counts:
        provider_counts, profiled_providers = _profile_execution_providers(
            onnx_path,
            device=device,
            input_size=input_size,
            logger=logger,
        )
    runtime_device = _runtime_device_from_profile(providers, provider_counts)

    result = {
        "method": method_name,
        "precision": precision,
        "device": device,
        "runtime_device": runtime_device,
        "providers": providers,
        "available_profile_providers": profiled_providers,
        "ort_provider_counts": provider_counts,
        "onnx_path": onnx_path,
        "model_size_mb": onnx_model_size_mb(onnx_path),
        **perf,
        **quality,
        "status": "ok",
    }

    save_path = os.path.join(output_dir, "logs", result_filename)
    save_json(result, save_path)
    logger.info("Saved ONNX result -> %s", save_path)

    return result
