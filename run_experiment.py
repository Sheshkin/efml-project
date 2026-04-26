import argparse
import hashlib
import json
import os
import traceback
from datetime import datetime

from src.utils.config import get_device, load_config
from src.utils.logging_utils import get_logger, save_json
from src.utils.seed import set_seed


CORE_COMMANDS = ["baseline", "compile", "backend", "ptq", "prune", "semi24"]
OPTIONAL_COMMANDS = ["qat", "tvm"]
RESULT_FILES = {
    "baseline": "baseline_result.json",
    "compile": "compile_result.json",
    "backend": "backend_result.json",
    "ptq": "ptq_result.json",
    "qat": "qat_result.json",
    "prune": "prune_result.json",
    "semi24": "sparsity24_result.json",
    "tvm": "tvm_result.json",
}
_CHECKPOINT_SHA256_CACHE = {}


def _config_path(args, name):
    return os.path.join(args.config_dir, f"{name}.yaml")


def _load_command_config(args, name):
    path = _config_path(args, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found for '{name}': {path}")
    return load_config(path)


def _resolve_output_dir(cfg, args):
    if getattr(args, "output_dir", None):
        return args.output_dir
    return cfg.get("paths", {}).get("output_dir", "results")


def _default_output_dir(args):
    if getattr(args, "output_dir", None):
        return args.output_dir
    try:
        cfg = _load_command_config(args, "baseline")
        return cfg.get("paths", {}).get("output_dir", "results")
    except Exception:
        return "results"


def _checkpoint_path(cfg, args=None):
    configured = cfg["model"]["checkpoint"]
    output_dir = getattr(args, "output_dir", None) if args is not None else None
    if output_dir:
        candidate = os.path.join(output_dir, "checkpoints", os.path.basename(configured))
        if os.path.exists(candidate):
            return candidate
    return configured


def _require_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "Run 'python run_experiment.py train' first."
        )


def _checkpoint_sha256(path):
    path = os.path.abspath(path)
    cached = _CHECKPOINT_SHA256_CACHE.get(path)
    if cached:
        return cached

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _CHECKPOINT_SHA256_CACHE[path] = digest
    return digest


def _data_signature(cfg):
    data_cfg = cfg.get("data", {})
    return {
        "root": data_cfg.get("root"),
        "split": "test",
        "image_size": data_cfg.get("image_size"),
        "batch_size": data_cfg.get("batch_size"),
        "num_workers": data_cfg.get("num_workers", 0),
        "test_max_samples": data_cfg.get("test_max_samples"),
        "seed": data_cfg.get("seed", 42),
    }


def _attach_common_metadata(result, command, cfg, checkpoint_path):
    result["experiment"] = command
    result["checkpoint_path"] = checkpoint_path
    result["checkpoint_sha256"] = _checkpoint_sha256(checkpoint_path)
    result["data_signature"] = _data_signature(cfg)
    result["model_config"] = cfg.get("model", {})
    result["benchmark_config"] = cfg.get("benchmark", {})
    return result


def _finalize_result(result, args, command, cfg, checkpoint_path):
    result = _attach_common_metadata(result, command, cfg, checkpoint_path)
    output_dir = _resolve_output_dir(cfg, args)
    save_json(result, os.path.join(output_dir, "logs", RESULT_FILES[command]))
    return result


def load_model(checkpoint_path, device, base_filters=64, bilinear=True):
    import torch

    from src.models.unet import build_unet

    _require_checkpoint(checkpoint_path)

    model = build_unet(base_filters=base_filters, bilinear=bilinear)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _make_test_loader(cfg):
    from src.data.dataset import build_test_loader

    return build_test_loader(
        data_root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        max_samples=cfg["data"].get("test_max_samples"),
        seed=cfg["data"].get("seed", 42),
    )


def _make_train_val_loaders(cfg):
    from src.data.dataset import build_dataloaders

    return build_dataloaders(
        data_root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        val_fraction=cfg["data"].get("val_fraction", 0.15),
        num_workers=cfg["data"].get("num_workers", 0),
        seed=cfg["data"].get("seed", 42),
        train_max_samples=cfg["data"].get("train_max_samples"),
        val_max_samples=cfg["data"].get("val_max_samples"),
    )


def run_train(args):
    cfg = _load_command_config(args, "train")
    set_seed(cfg["training"]["seed"])
    device = get_device()

    logger = get_logger("train")
    logger.info("Training on device: %s", device)

    train_loader, val_loader = _make_train_val_loaders(cfg)

    from src.models.unet import build_unet

    model = build_unet(
        base_filters=cfg["model"]["base_filters"],
        bilinear=cfg["model"].get("bilinear", True),
    )
    logger.info("UNet parameters: %s", f"{model.count_parameters():,}")

    from src.training.trainer import Trainer

    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    log_dir = cfg["paths"]["log_dir"]
    if getattr(args, "output_dir", None):
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        log_dir = os.path.join(args.output_dir, "logs")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        experiment_name=cfg["experiment_name"],
    )
    trainer.fit(epochs=cfg["training"]["epochs"])
    logger.info("Training complete.")
    return {"status": "ok", "method": "train"}


def run_baseline(args):
    cfg = _load_command_config(args, "baseline")
    set_seed(42)
    device = get_device()

    test_loader = _make_test_loader(cfg)
    checkpoint_path = _checkpoint_path(cfg, args)
    model = load_model(
        checkpoint_path,
        device,
        cfg["model"]["base_filters"],
        cfg["model"].get("bilinear", True),
    )

    from src.inference.baseline import run_baseline_benchmark

    result = run_baseline_benchmark(
        model=model,
        test_loader=test_loader,
        device=device,
        precision=cfg["benchmark"]["precision"],
        warmup=cfg["benchmark"]["warmup_runs"],
        latency_runs=cfg["benchmark"]["latency_runs"],
        batch_size_throughput=cfg["benchmark"]["batch_size_throughput"],
        input_size=tuple(cfg["benchmark"]["input_size"]),
        output_dir=_resolve_output_dir(cfg, args),
    )
    result = _finalize_result(result, args, "baseline", cfg, checkpoint_path)
    print("\nBaseline result:")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))
    return result


def run_compile(args):
    cfg = _load_command_config(args, "compile")
    set_seed(42)
    device = get_device()

    test_loader = _make_test_loader(cfg)
    checkpoint_path = _checkpoint_path(cfg, args)
    model = load_model(
        checkpoint_path,
        device,
        cfg["model"]["base_filters"],
        cfg["model"].get("bilinear", True),
    )

    from src.inference.compile_infer import run_compile_benchmark

    result = run_compile_benchmark(
        model=model,
        test_loader=test_loader,
        device=device,
        precision=cfg["benchmark"]["precision"],
        compile_mode=cfg["compile"]["mode"],
        warmup=cfg["benchmark"]["warmup_runs"],
        latency_runs=cfg["benchmark"]["latency_runs"],
        batch_size_throughput=cfg["benchmark"]["batch_size_throughput"],
        input_size=tuple(cfg["benchmark"]["input_size"]),
        output_dir=_resolve_output_dir(cfg, args),
    )
    result = _finalize_result(result, args, "compile", cfg, checkpoint_path)
    print("\ntorch.compile result:")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))
    return result


def run_backend(args):
    cfg = _load_command_config(args, "backend")
    set_seed(cfg["benchmark"].get("seed", 42))
    device = get_device()

    test_loader = _make_test_loader(cfg)
    checkpoint_path = _checkpoint_path(cfg, args)
    model = load_model(
        checkpoint_path,
        device,
        cfg["model"]["base_filters"],
        cfg["model"].get("bilinear", True),
    )

    from src.inference.onnx_infer import run_onnx_benchmark

    result = run_onnx_benchmark(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=_resolve_output_dir(cfg, args),
        precision=cfg["benchmark"].get("precision", "fp32"),
        warmup=cfg["benchmark"]["warmup_runs"],
        latency_runs=cfg["benchmark"]["latency_runs"],
        batch_size_throughput=cfg["benchmark"]["batch_size_throughput"],
        input_size=tuple(cfg["benchmark"]["input_size"]),
        method_name="ONNX Runtime",
        result_filename="backend_result.json",
    )
    result = _finalize_result(result, args, "backend", cfg, checkpoint_path)
    print("\nONNX backend result:")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))
    return result


def run_ptq(args):
    cfg = _load_command_config(args, "ptq")
    set_seed(cfg["ptq"].get("seed", 42))
    device = get_device()

    _, calibration_loader = _make_train_val_loaders(cfg)
    test_loader = _make_test_loader(cfg)
    checkpoint_path = _checkpoint_path(cfg, args)

    model = load_model(
        checkpoint_path,
        device,
        cfg["model"]["base_filters"],
        cfg["model"].get("bilinear", True),
    )

    from src.inference.ptq_infer import run_ptq_benchmark

    result = run_ptq_benchmark(
        model=model,
        calibration_loader=calibration_loader,
        test_loader=test_loader,
        device=device,
        output_dir=_resolve_output_dir(cfg, args),
        input_size=tuple(cfg["benchmark"]["input_size"]),
        warmup=cfg["benchmark"]["warmup_runs"],
        latency_runs=cfg["benchmark"]["latency_runs"],
        batch_size_throughput=cfg["benchmark"]["batch_size_throughput"],
        calibration_batches=cfg["ptq"]["calibration_batches"],
        force_dynamic=cfg["ptq"].get("force_dynamic", False),
    )
    result = _finalize_result(result, args, "ptq", cfg, checkpoint_path)
    print("\nPTQ result:")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))
    return result


def run_qat(args):
    import torch

    cfg = _load_command_config(args, "qat")
    set_seed(cfg["qat"].get("seed", 42))

    infer_device = get_device()
    if torch.cuda.is_available() and infer_device == "cuda":
        qat_train_device = "cuda"
    else:
        qat_train_device = "cpu"

    train_loader, _ = _make_train_val_loaders(cfg)
    test_loader = _make_test_loader(cfg)
    checkpoint_path = _checkpoint_path(cfg, args)

    model = load_model(
        checkpoint_path,
        infer_device,
        cfg["model"]["base_filters"],
        cfg["model"].get("bilinear", True),
    )

    from src.inference.qat_infer import run_qat_benchmark

    result = run_qat_benchmark(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        output_dir=_resolve_output_dir(cfg, args),
        train_device=qat_train_device,
        qat_epochs=cfg["qat"]["epochs"],
        qat_lr=cfg["qat"]["lr"],
        qat_weight_decay=cfg["qat"]["weight_decay"],
        max_qat_batches=cfg["qat"].get("max_qat_batches"),
        warmup=cfg["benchmark"]["warmup_runs"],
        latency_runs=cfg["benchmark"]["latency_runs"],
        batch_size_throughput=cfg["benchmark"]["batch_size_throughput"],
        input_size=tuple(cfg["benchmark"]["input_size"]),
    )
    result = _finalize_result(result, args, "qat", cfg, checkpoint_path)
    print("\nQAT result:")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))
    return result


def run_prune(args):
    cfg = _load_command_config(args, "prune")
    set_seed(cfg["prune"].get("seed", 42))
    device = get_device()

    train_loader, _ = _make_train_val_loaders(cfg)
    test_loader = _make_test_loader(cfg)
    checkpoint_path = _checkpoint_path(cfg, args)

    model = load_model(
        checkpoint_path,
        device,
        cfg["model"]["base_filters"],
        cfg["model"].get("bilinear", True),
    )

    from src.inference.prune_infer import run_unstructured_pruning_benchmark

    result = run_unstructured_pruning_benchmark(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        output_dir=_resolve_output_dir(cfg, args),
        precision=cfg["benchmark"]["precision"],
        prune_amount=cfg["prune"]["amount"],
        finetune_epochs=cfg["prune"]["finetune_epochs"],
        finetune_lr=cfg["prune"]["finetune_lr"],
        finetune_weight_decay=cfg["prune"]["finetune_weight_decay"],
        max_finetune_batches=cfg["prune"].get("max_finetune_batches"),
        warmup=cfg["benchmark"]["warmup_runs"],
        latency_runs=cfg["benchmark"]["latency_runs"],
        batch_size_throughput=cfg["benchmark"]["batch_size_throughput"],
        input_size=tuple(cfg["benchmark"]["input_size"]),
    )
    result = _finalize_result(result, args, "prune", cfg, checkpoint_path)
    print("\nPruning result:")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))
    return result


def run_semi24(args):
    cfg = _load_command_config(args, "sparsity24")
    set_seed(cfg["sparsity24"].get("seed", 42))
    device = get_device()

    train_loader, _ = _make_train_val_loaders(cfg)
    test_loader = _make_test_loader(cfg)
    checkpoint_path = _checkpoint_path(cfg, args)

    model = load_model(
        checkpoint_path,
        device,
        cfg["model"]["base_filters"],
        cfg["model"].get("bilinear", True),
    )

    from src.inference.prune_infer import run_semi_structured_2_4_benchmark

    result = run_semi_structured_2_4_benchmark(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        output_dir=_resolve_output_dir(cfg, args),
        precision=cfg["benchmark"]["precision"],
        finetune_epochs=cfg["sparsity24"]["finetune_epochs"],
        finetune_lr=cfg["sparsity24"]["finetune_lr"],
        finetune_weight_decay=cfg["sparsity24"]["finetune_weight_decay"],
        max_finetune_batches=cfg["sparsity24"].get("max_finetune_batches"),
        warmup=cfg["benchmark"]["warmup_runs"],
        latency_runs=cfg["benchmark"]["latency_runs"],
        batch_size_throughput=cfg["benchmark"]["batch_size_throughput"],
        input_size=tuple(cfg["benchmark"]["input_size"]),
    )
    result = _finalize_result(result, args, "semi24", cfg, checkpoint_path)
    print("\n2:4 result:")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))
    return result


def run_tvm(args):
    cfg = _load_command_config(args, "tvm")
    set_seed(cfg["benchmark"].get("seed", 42))
    device = get_device()

    test_loader = _make_test_loader(cfg)
    checkpoint_path = _checkpoint_path(cfg, args)
    model = load_model(
        checkpoint_path,
        device,
        cfg["model"]["base_filters"],
        cfg["model"].get("bilinear", True),
    )

    from src.inference.tvm_infer import run_tvm_benchmark

    result = run_tvm_benchmark(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=_resolve_output_dir(cfg, args),
        input_size=tuple(cfg["benchmark"]["input_size"]),
        warmup=cfg["benchmark"]["warmup_runs"],
        latency_runs=cfg["benchmark"]["latency_runs"],
        batch_size_throughput=cfg["benchmark"]["batch_size_throughput"],
    )
    result = _finalize_result(result, args, "tvm", cfg, checkpoint_path)
    print("\nTVM result:")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))
    return result


def run_report(args, result_files=None):
    from src.benchmark.reporter import (
        build_reports_from_logs,
        format_quality_table,
        format_results_table,
    )

    output_dir = _default_output_dir(args)
    results = build_reports_from_logs(
        log_dir=os.path.join(output_dir, "logs"),
        table_dir=os.path.join(output_dir, "tables"),
        plot_dir=os.path.join(output_dir, "plots"),
        result_files=result_files,
    )

    print("\n=== Performance Benchmark ===")
    print(format_results_table(results))

    print("\n=== Quality Metrics ===")
    print(format_quality_table(results))

    validation_path = os.path.join(output_dir, "tables", "validation.md")
    if os.path.exists(validation_path):
        with open(validation_path, "r") as f:
            print("\n=== Validation ===")
            print(f.read())

    return {"status": "ok", "method": "report", "results_count": len(results)}


def _failure_result(method_name, error_text):
    return {
        "method": method_name,
        "status": "failed",
        "error": error_text,
    }


def run_future_all(args):
    logger = get_logger("future_pipeline")

    method_order = list(CORE_COMMANDS)
    if getattr(args, "include_optional", False):
        method_order.extend(OPTIONAL_COMMANDS)

    continue_on_error = not getattr(args, "fail_fast", False)
    output_dir = _default_output_dir(args)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "continue_on_error": continue_on_error,
        "output_dir": output_dir,
        "runs": [],
    }
    executed_result_files = []

    for command in method_order:
        logger.info("[future_all] Running '%s'", command)
        fn = COMMANDS[command]
        result_file = RESULT_FILES[command]
        try:
            result = fn(args)
            step_status = "ok"
            if isinstance(result, dict):
                step_status = result.get("status", "ok")
            executed_result_files.append(result_file)
            manifest["runs"].append(
                {
                    "command": command,
                    "status": step_status,
                    "result_file": result_file,
                }
            )
        except Exception as exc:
            err_text = f"{type(exc).__name__}: {exc}"
            logger.error("[future_all] %s failed: %s", command, err_text)
            traceback.print_exc()

            failure = _failure_result(command, err_text)
            save_json(failure, os.path.join(logs_dir, result_file))
            executed_result_files.append(result_file)

            manifest["runs"].append(
                {
                    "command": command,
                    "status": "failed",
                    "error": err_text,
                    "result_file": result_file,
                }
            )
            if not continue_on_error:
                break

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(manifest, os.path.join(logs_dir, "future_pipeline_manifest.json"))

    run_report(args, result_files=executed_result_files)
    return {"status": "ok", "method": "future_all", "manifest": manifest}


def run_all(args):
    run_train(args)
    run_future_all(args)
    return {"status": "ok", "method": "all"}


COMMANDS = {
    "train": run_train,
    "baseline": run_baseline,
    "compile": run_compile,
    "backend": run_backend,
    "ptq": run_ptq,
    "qat": run_qat,
    "prune": run_prune,
    "semi24": run_semi24,
    "tvm": run_tvm,
    "future_all": run_future_all,
    "report": run_report,
    "all": run_all,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Acceleration Benchmark")
    parser.add_argument("command", choices=list(COMMANDS.keys()), help="Experiment to run")
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory with command YAML configs (default: configs).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for logs/tables/plots/results.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop future_all pipeline on first failed experiment.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also run optional QAT and TVM experiments after the six core experiments.",
    )
    args = parser.parse_args()
    COMMANDS[args.command](args)
