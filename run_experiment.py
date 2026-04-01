import argparse
import os
import json
import torch

from src.utils.config import load_config, get_device
from src.utils.seed import set_seed
from src.utils.logging_utils import get_logger
from src.models.unet import build_unet
from src.data.dataset import build_dataloaders, build_test_loader


def load_model(checkpoint_path, device, base_filters=64):
    model = build_unet(base_filters=base_filters)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_train(args):
    cfg = load_config("configs/train.yaml")
    set_seed(cfg["training"]["seed"])
    device = get_device()

    logger = get_logger("train")
    logger.info(f"Training on device: {device}")

    train_loader, val_loader = build_dataloaders(
        data_root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        val_fraction=cfg["data"]["val_fraction"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["data"]["seed"],
    )

    model = build_unet(
        base_filters=cfg["model"]["base_filters"],
        bilinear=cfg["model"]["bilinear"],
    )
    logger.info(f"UNet parameters: {model.count_parameters():,}")

    from src.training.trainer import Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        checkpoint_dir=cfg["paths"]["checkpoint_dir"],
        log_dir=cfg["paths"]["log_dir"],
        experiment_name=cfg["experiment_name"],
    )
    trainer.fit(epochs=cfg["training"]["epochs"])
    logger.info("Training complete.")


def run_baseline(args):
    cfg = load_config("configs/baseline.yaml")
    set_seed(42)
    device = get_device()

    test_loader = build_test_loader(
        data_root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
    )
    model = load_model(cfg["model"]["checkpoint"], device, cfg["model"]["base_filters"])

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
        output_dir=cfg["paths"]["output_dir"],
    )
    print("\nBaseline result:")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))


def run_compile(args):
    cfg = load_config("configs/compile.yaml")
    set_seed(42)
    device = get_device()

    test_loader = build_test_loader(
        data_root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
    )
    model = load_model(cfg["model"]["checkpoint"], device, cfg["model"]["base_filters"])

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
        output_dir=cfg["paths"]["output_dir"],
    )
    print("\ntorch.compile result:")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))


def run_report(args):
    from src.benchmark.reporter import format_results_table, format_quality_table

    result_files = {
        "baseline": "results/logs/baseline_result.json",
        "compile": "results/logs/compile_result.json",
    }

    perf_results = []
    quality_results = []

    for name, path in result_files.items():
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        with open(path) as f:
            r = json.load(f)
        perf_results.append(r)
        quality_results.append(r)

    os.makedirs("results/tables", exist_ok=True)

    if perf_results:
        print("\n=== Performance Benchmark ===")
        print(format_results_table(perf_results))
        with open("results/tables/performance.md", "w") as f:
            f.write("## Performance Benchmark\n\n")
            f.write(format_results_table(perf_results))

    if quality_results:
        print("\n=== Quality Metrics ===")
        print(format_quality_table(quality_results))
        with open("results/tables/quality.md", "w") as f:
            f.write("## Quality Metrics\n\n")
            f.write(format_quality_table(quality_results))


def run_all(args):
    run_train(args)
    run_baseline(args)
    run_compile(args)
    run_report(args)


COMMANDS = {
    "train": run_train,
    "baseline": run_baseline,
    "compile": run_compile,
    "report": run_report,
    "all": run_all,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Acceleration Benchmark")
    parser.add_argument("command", choices=list(COMMANDS.keys()), help="Experiment to run")
    args = parser.parse_args()
    COMMANDS[args.command](args)
