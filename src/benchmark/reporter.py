import json
import os
from glob import glob

PERF_HEADERS = ["Method", "Precision", "P50 (ms)", "P95 (ms)", "Throughput", "Size (MB)", "Device", "Status"]
QUALITY_HEADERS = ["Method", "Dice", "IoU", "Pixel Acc", "Sparsity", "Status"]
VALIDATION_HEADERS = ["Check", "Status", "Details"]


def _fmt_float(value, digits=4):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _method(row):
    return row.get("method") or row.get("name") or "unknown"


def _status(row):
    return row.get("status", "ok")


def _ok_results(results):
    return [r for r in results if _status(r) == "ok"]


def _as_markdown_table(rows, headers):
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def _format_table(rows, headers):
    try:
        from tabulate import tabulate

        return tabulate(rows, headers=headers, tablefmt="github")
    except Exception:
        return _as_markdown_table(rows, headers)


def format_results_table(results):
    rows = []
    for r in results:
        rows.append([
            _method(r),
            r.get("precision", "-"),
            _fmt_float(r.get("p50_latency_ms"), 2),
            _fmt_float(r.get("p95_latency_ms"), 2),
            _fmt_float(r.get("throughput_img_sec"), 2),
            _fmt_float(r.get("model_size_mb"), 2),
            r.get("runtime_device", r.get("device", "-")),
            _status(r),
        ])
    return _format_table(rows, headers=PERF_HEADERS)


def format_quality_table(results):
    rows = []
    for r in results:
        rows.append([
            _method(r),
            _fmt_float(r.get("dice"), 4),
            _fmt_float(r.get("iou"), 4),
            _fmt_float(r.get("pixel_accuracy"), 4),
            _fmt_float(r.get("weight_sparsity"), 4),
            _status(r),
        ])
    return _format_table(rows, headers=QUALITY_HEADERS)


def load_result_files(log_dir="results/logs", result_files=None):
    if result_files:
        paths = [os.path.join(log_dir, name) for name in result_files]
        result_files = [path for path in paths if os.path.exists(path)]
    else:
        result_files = sorted(glob(os.path.join(log_dir, "*_result.json")))

    loaded = []
    for path in result_files:
        try:
            with open(path, "r") as f:
                row = json.load(f)
            row.setdefault("result_file", os.path.basename(path))
            loaded.append(row)
        except Exception as exc:
            loaded.append({
                "method": os.path.basename(path),
                "status": "failed",
                "error": f"Failed to read result file: {exc}",
                "result_file": os.path.basename(path),
            })
    return loaded


def _find_method(results, needle):
    needle = needle.lower()
    for row in results:
        if needle in _method(row).lower():
            return row
    return None


def _ratio(numerator, denominator):
    try:
        denominator = float(denominator)
        if denominator == 0.0:
            return None
        return float(numerator) / denominator
    except (TypeError, ValueError):
        return None


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _checkpoint_details(rows):
    if any(not r.get("checkpoint_sha256") or not r.get("data_signature") for r in rows):
        return "warning", "Some core rows are missing checkpoint hash or test data signature metadata."

    hashes = {r.get("checkpoint_sha256") for r in rows if r and r.get("checkpoint_sha256")}
    data = {
        json.dumps(r.get("data_signature"), sort_keys=True)
        for r in rows
        if r and r.get("data_signature")
    }
    if len(hashes) <= 1 and len(data) <= 1:
        return "ok", "Core rows use the same checkpoint hash and test data signature."
    return "warning", "Core rows do not share one checkpoint/data signature."


def validate_results(results):
    baseline = _find_method(results, "baseline")
    compile_row = _find_method(results, "torch.compile")
    onnx = _find_method(results, "onnx runtime")
    ptq = _find_method(results, "ptq")
    prune = _find_method(results, "pruning")
    semi24 = _find_method(results, "2:4")

    checks = []
    core_rows = [r for r in [baseline, compile_row, onnx, ptq, prune, semi24] if r]
    if len(core_rows) >= 2:
        status, details = _checkpoint_details(core_rows)
        checks.append({"check": "checkpoint/data consistency", "status": status, "details": details})

    if baseline and compile_row and onnx:
        base_dice = _to_float(baseline.get("dice"))
        compile_dice = _to_float(compile_row.get("dice"))
        onnx_dice = _to_float(onnx.get("dice"))
        if None not in (base_dice, compile_dice, onnx_dice):
            compile_delta = abs(compile_dice - base_dice)
            onnx_delta = abs(onnx_dice - base_dice)
            status = "ok" if max(compile_delta, onnx_delta) <= 0.02 else "warning"
            checks.append({
                "check": "baseline/compile/onnx Dice parity",
                "status": status,
                "details": (
                    f"baseline={_fmt_float(base_dice)}, "
                    f"compile_delta={compile_delta:.4f}, onnx_delta={onnx_delta:.4f}"
                ),
            })

            onnx_gain = onnx_dice - base_dice
            if onnx_gain > 0.02:
                checks.append({
                    "check": "unexpected Dice gain without retraining",
                    "status": "warning",
                    "details": f"ONNX Dice is higher than baseline by {onnx_gain:.4f}.",
                })
        else:
            checks.append({
                "check": "baseline/compile/onnx Dice parity",
                "status": "warning",
                "details": "One or more core Dice values are missing.",
            })

    if baseline and onnx:
        latency_ratio = _ratio(onnx.get("p50_latency_ms"), baseline.get("p50_latency_ms"))
        counts = onnx.get("ort_provider_counts") or {}
        cuda_ops = int(counts.get("CUDAExecutionProvider", 0) or 0)
        cpu_ops = int(counts.get("CPUExecutionProvider", 0) or 0)
        status = "ok"
        details = f"p50_ratio={latency_ratio:.2f}x" if latency_ratio else "p50_ratio=n/a"
        details += f", provider_counts={counts}"
        if latency_ratio and latency_ratio > 20.0:
            status = "warning"
        if onnx.get("device") == "cuda" and cuda_ops == 0:
            status = "warning"
            details += ", CUDA provider did not execute profiled kernels"
        elif cpu_ops and cuda_ops:
            details += ", mixed CUDA/CPU execution"
        checks.append({"check": "ONNX Runtime backend", "status": status, "details": details})

    if ptq:
        size_ratio = _ratio(ptq.get("model_size_mb"), ptq.get("source_model_size_mb"))
        latency_ratio = _ratio(ptq.get("p50_latency_ms"), onnx.get("p50_latency_ms") if onnx else None)
        status = "ok" if size_ratio is not None and size_ratio < 0.5 else "warning"
        details = f"size_ratio={size_ratio:.2f}" if size_ratio is not None else "size_ratio=n/a"
        if latency_ratio is not None:
            details += f", latency_vs_onnx={latency_ratio:.2f}x"
            if latency_ratio >= 0.9:
                details += ", no measured INT8 speedup"
        details += f", runtime_device={ptq.get('runtime_device', '-')}"
        checks.append({"check": "PTQ size/backend", "status": status, "details": details})

    if prune:
        status = "ok" if prune.get("storage_format") == "dense" and not prune.get("speedup_valid") else "warning"
        checks.append({
            "check": "unstructured pruning speedup claim",
            "status": status,
            "details": prune.get("speedup_note", "No sparse-kernel metadata recorded."),
        })

    if semi24:
        status = "ok" if not semi24.get("sparse_speedup_supported") else "ok"
        checks.append({
            "check": "2:4 sparsity backend support",
            "status": status,
            "details": semi24.get("speedup_note", "No 2:4 backend metadata recorded."),
        })

    return checks


def format_validation_table(checks):
    rows = []
    for check in checks:
        rows.append([
            check.get("check", "-"),
            check.get("status", "-"),
            check.get("details", "-"),
        ])
    return _format_table(rows, headers=VALIDATION_HEADERS)


def save_tables(results, table_dir="results/tables"):
    os.makedirs(table_dir, exist_ok=True)

    perf_md = format_results_table(results)
    qual_md = format_quality_table(results)

    with open(os.path.join(table_dir, "performance.md"), "w") as f:
        f.write("## Performance Benchmark\n\n")
        f.write(perf_md)
        f.write("\n")

    with open(os.path.join(table_dir, "quality.md"), "w") as f:
        f.write("## Quality Metrics\n\n")
        f.write(qual_md)
        f.write("\n")

    checks = validate_results(results)
    with open(os.path.join(table_dir, "validation.md"), "w") as f:
        f.write("## Validation Checks\n\n")
        f.write(format_validation_table(checks))
        f.write("\n")


def _save_bar_plot(methods, values, ylabel, title, out_path, color):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.bar(methods, values, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_plots(results, plot_dir="results/plots"):
    try:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    os.makedirs(plot_dir, exist_ok=True)

    ok = _ok_results(results)
    if not ok:
        return

    methods = [_method(r) for r in ok]

    p50_values = [float(r.get("p50_latency_ms", 0.0) or 0.0) for r in ok]
    p95_values = [float(r.get("p95_latency_ms", 0.0) or 0.0) for r in ok]
    thr_values = [float(r.get("throughput_img_sec", 0.0) or 0.0) for r in ok]
    dice_values = [float(r.get("dice", 0.0) or 0.0) for r in ok]
    iou_values = [float(r.get("iou", 0.0) or 0.0) for r in ok]

    _save_bar_plot(
        methods,
        p50_values,
        "ms",
        "Latency P50 by Method",
        os.path.join(plot_dir, "latency_p50.png"),
        color="#4E79A7",
    )
    _save_bar_plot(
        methods,
        p95_values,
        "ms",
        "Latency P95 by Method",
        os.path.join(plot_dir, "latency_p95.png"),
        color="#F28E2B",
    )
    _save_bar_plot(
        methods,
        thr_values,
        "images/sec",
        "Throughput by Method",
        os.path.join(plot_dir, "throughput.png"),
        color="#59A14F",
    )

    plt.figure(figsize=(10, 5))
    x = range(len(methods))
    plt.plot(x, dice_values, marker="o", label="Dice", color="#E15759")
    plt.plot(x, iou_values, marker="s", label="IoU", color="#76B7B2")
    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("score")
    plt.title("Quality Metrics by Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "quality_scores.png"), dpi=160)
    plt.close()


def save_summary_json(results, path="results/logs/summary_results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def build_reports_from_logs(log_dir="results/logs", table_dir="results/tables", plot_dir="results/plots", result_files=None):
    results = load_result_files(log_dir=log_dir, result_files=result_files)
    save_tables(results, table_dir=table_dir)
    try:
        save_plots(results, plot_dir=plot_dir)
    except Exception:
        pass
    save_summary_json(results, path=os.path.join(log_dir, "summary_results.json"))
    return results
