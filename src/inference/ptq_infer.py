import os
import types

import numpy as np

from src.inference.onnx_infer import (
    create_ort_session,
    export_to_onnx,
    run_onnx_benchmark,
)
from src.utils.logging_utils import get_logger, save_json


class _LoaderDataReader:
    def __init__(self, data_loader, input_name, max_batches=32, input_dtype=np.float32):
        self.input_name = input_name
        self._items = []
        self._idx = 0

        for i, (imgs, _) in enumerate(data_loader):
            if i >= max_batches:
                break
            arr = imgs.numpy().astype(input_dtype, copy=False)
            self._items.append({self.input_name: arr})

    def get_next(self):
        if self._idx >= len(self._items):
            return None
        item = self._items[self._idx]
        self._idx += 1
        return item

    def rewind(self):
        self._idx = 0


def _import_quant_tools():
    try:
        _patch_onnx_mapping_compat()
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
    except Exception as exc:
        raise RuntimeError(
            "onnxruntime quantization tools are required for PTQ."
        ) from exc

    return {
        "CalibrationMethod": CalibrationMethod,
        "QuantFormat": QuantFormat,
        "QuantType": QuantType,
        "quantize_dynamic": quantize_dynamic,
        "quantize_static": quantize_static,
    }


def _patch_onnx_mapping_compat():
    try:
        import onnx
    except Exception:
        return

    if hasattr(onnx, "mapping"):
        return
    if not hasattr(onnx, "_mapping"):
        return

    tensor_type_map = getattr(onnx._mapping, "TENSOR_TYPE_MAP", {})
    tensor_to_np = {}
    np_to_tensor = {}

    for tensor_type, dtype_map in tensor_type_map.items():
        np_dtype = getattr(dtype_map, "np_dtype", None)
        if np_dtype is None:
            continue
        np_dtype = np.dtype(np_dtype)
        tensor_to_np[tensor_type] = np_dtype
        np_to_tensor[np_dtype] = tensor_type
        np_to_tensor[np_dtype.type] = tensor_type

    onnx.mapping = types.SimpleNamespace(
        TENSOR_TYPE_TO_NP_TYPE=tensor_to_np,
        NP_TYPE_TO_TENSOR_TYPE=np_to_tensor,
    )


def run_ptq_benchmark(model, calibration_loader, test_loader, device, output_dir="results", input_size=(256, 256), warmup=20, latency_runs=100, batch_size_throughput=16, calibration_batches=32, force_dynamic=False):
    logger = get_logger("ptq_benchmark")
    tools = _import_quant_tools()

    onnx_dir = os.path.join(output_dir, "artifacts", "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    fp32_onnx = os.path.join(onnx_dir, "unet_ptq_fp32.onnx")
    int8_onnx = os.path.join(onnx_dir, "unet_ptq_int8.onnx")

    logger.info("Exporting FP32 ONNX for PTQ -> %s", fp32_onnx)
    export_to_onnx(model, fp32_onnx, input_size=input_size, precision="fp32")
    source_model_size_mb = os.path.getsize(fp32_onnx) / 1e6

    temp_session, _ = create_ort_session(fp32_onnx, device="cpu", logger=logger)
    input_name = temp_session.get_inputs()[0].name
    reader = _LoaderDataReader(
        calibration_loader,
        input_name=input_name,
        max_batches=calibration_batches,
        input_dtype=np.float32,
    )

    quant_mode = "static"
    fallback_reason = None

    if force_dynamic:
        quant_mode = "dynamic"
        logger.info("Running forced dynamic quantization...")
        tools["quantize_dynamic"](
            fp32_onnx,
            int8_onnx,
            weight_type=tools["QuantType"].QInt8,
        )
    else:
        try:
            logger.info("Running static PTQ quantization (QDQ + per-channel)...")
            tools["quantize_static"](
                fp32_onnx,
                int8_onnx,
                reader,
                quant_format=tools["QuantFormat"].QDQ,
                activation_type=tools["QuantType"].QUInt8,
                weight_type=tools["QuantType"].QInt8,
                per_channel=True,
                calibrate_method=tools["CalibrationMethod"].MinMax,
            )
        except Exception as exc:
            logger.warning("Static PTQ failed (%s).", exc)
            quant_mode = "static_failed"
            fallback_reason = str(exc)
            if device == "cuda":
                result = {
                    "method": "PTQ (int8)",
                    "precision": "int8",
                    "device": device,
                    "runtime_device": device,
                    "status": "skipped",
                    "error": (
                        "Static QDQ PTQ failed. Dynamic PTQ is intentionally not used "
                        "on CUDA because it emits ConvInteger nodes that CUDAExecutionProvider "
                        "does not implement for this model."
                    ),
                    "onnx_path": int8_onnx,
                }
                result["quantization_mode"] = quant_mode
                result["calibration_batches"] = calibration_batches
                result["source_model_size_mb"] = source_model_size_mb
                result["source_fp32_onnx"] = fp32_onnx
                result["quantized_onnx"] = int8_onnx
                result["fallback_reason"] = fallback_reason
                save_json(result, os.path.join(output_dir, "logs", "ptq_result.json"))
                return result
            else:
                logger.warning("Falling back to dynamic PTQ on non-CUDA backend.")
                quant_mode = "dynamic"
                tools["quantize_dynamic"](
                    fp32_onnx,
                    int8_onnx,
                    weight_type=tools["QuantType"].QInt8,
                )

    logger.info("Running benchmark on PTQ model -> %s", int8_onnx)
    try:
        result = run_onnx_benchmark(
            model=None,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            precision="int8",
            warmup=warmup,
            latency_runs=latency_runs,
            batch_size_throughput=batch_size_throughput,
            input_size=input_size,
            onnx_path=int8_onnx,
            method_name="PTQ (int8)",
            result_filename="ptq_result.json",
            export_if_missing=False,
        )
    except Exception as exc:
        logger.warning("Skipping PTQ runtime benchmark: %s", exc)
        result = {
            "method": "PTQ (int8)",
            "precision": "int8",
            "device": device,
            "runtime_device": device,
            "status": "skipped",
            "error": str(exc),
            "onnx_path": int8_onnx,
        }

    result["quantization_mode"] = quant_mode
    result["calibration_batches"] = calibration_batches
    result["source_model_size_mb"] = source_model_size_mb
    result["source_fp32_onnx"] = fp32_onnx
    result["quantized_onnx"] = int8_onnx
    if fallback_reason:
        result["fallback_reason"] = fallback_reason

    save_json(result, os.path.join(output_dir, "logs", "ptq_result.json"))
    return result
