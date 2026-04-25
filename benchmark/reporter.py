import json
import os
from glob import glob
from typing import Dict, Iterable, List
PERF_HEADERS = ['Method', 'Precision', 'P50 (ms)', 'P95 (ms)', 'Throughput', 'Size (MB)', 'Device', 'Status']
QUALITY_HEADERS = ['Method', 'Dice', 'IoU', 'Pixel Acc', 'Sparsity', 'Status']

def _fmt_float(value, digits=4):
    if value is None:
        return '-'
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)

def _method(row):
    return row.get('method') or row.get('name') or 'unknown'

def _status(row):
    return row.get('status', 'ok')

def _ok_results(results):
    return [r for r in results if _status(r) == 'ok']

def _as_markdown_table(rows, headers):
    header_line = '| ' + ' | '.join(headers) + ' |'
    sep_line = '|' + '|'.join(['---'] * len(headers)) + '|'
    body = ['| ' + ' | '.join((str(v) for v in row)) + ' |' for row in rows]
    return '\n'.join([header_line, sep_line, *body])

def _format_table(rows, headers):
    try:
        from tabulate import tabulate
        return tabulate(rows, headers=headers, tablefmt='github')
    except Exception:
        return _as_markdown_table(rows, headers)

def format_results_table(results):
    rows = []
    for r in results:
        rows.append([_method(r), r.get('precision', '-'), _fmt_float(r.get('p50_latency_ms'), 2), _fmt_float(r.get('p95_latency_ms'), 2), _fmt_float(r.get('throughput_img_sec'), 2), _fmt_float(r.get('model_size_mb'), 2), r.get('runtime_device', r.get('device', '-')), _status(r)])
    return _format_table(rows, headers=PERF_HEADERS)

def format_quality_table(results):
    rows = []
    for r in results:
        rows.append([_method(r), _fmt_float(r.get('dice'), 4), _fmt_float(r.get('iou'), 4), _fmt_float(r.get('pixel_accuracy'), 4), _fmt_float(r.get('weight_sparsity'), 4), _status(r)])
    return _format_table(rows, headers=QUALITY_HEADERS)

def load_result_files(log_dir='results/logs'):
    result_files = sorted(glob(os.path.join(log_dir, '*_result.json')))
    loaded = []
    for path in result_files:
        try:
            with open(path, 'r') as f:
                row = json.load(f)
            row.setdefault('result_file', os.path.basename(path))
            loaded.append(row)
        except Exception as exc:
            loaded.append({'method': os.path.basename(path), 'status': 'failed', 'error': f'Failed to read result file: {exc}', 'result_file': os.path.basename(path)})
    return loaded

def save_tables(results, table_dir='results/tables'):
    os.makedirs(table_dir, exist_ok=True)
    perf_md = format_results_table(results)
    qual_md = format_quality_table(results)
    with open(os.path.join(table_dir, 'performance.md'), 'w') as f:
        f.write('## Performance Benchmark\n\n')
        f.write(perf_md)
        f.write('\n')
    with open(os.path.join(table_dir, 'quality.md'), 'w') as f:
        f.write('## Quality Metrics\n\n')
        f.write(qual_md)
        f.write('\n')

def _save_bar_plot(methods, values, ylabel, title, out_path, color):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.bar(methods, values, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_plots(results, plot_dir='results/plots'):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(plot_dir, exist_ok=True)
    ok = _ok_results(results)
    if not ok:
        return
    methods = [_method(r) for r in ok]
    p50_values = [float(r.get('p50_latency_ms', 0.0) or 0.0) for r in ok]
    p95_values = [float(r.get('p95_latency_ms', 0.0) or 0.0) for r in ok]
    thr_values = [float(r.get('throughput_img_sec', 0.0) or 0.0) for r in ok]
    dice_values = [float(r.get('dice', 0.0) or 0.0) for r in ok]
    iou_values = [float(r.get('iou', 0.0) or 0.0) for r in ok]
    _save_bar_plot(methods, p50_values, 'ms', 'Latency P50 by Method', os.path.join(plot_dir, 'latency_p50.png'), color='#4E79A7')
    _save_bar_plot(methods, p95_values, 'ms', 'Latency P95 by Method', os.path.join(plot_dir, 'latency_p95.png'), color='#F28E2B')
    _save_bar_plot(methods, thr_values, 'images/sec', 'Throughput by Method', os.path.join(plot_dir, 'throughput.png'), color='#59A14F')
    plt.figure(figsize=(10, 5))
    x = range(len(methods))
    plt.plot(x, dice_values, marker='o', label='Dice', color='#E15759')
    plt.plot(x, iou_values, marker='s', label='IoU', color='#76B7B2')
    plt.xticks(x, methods, rotation=20, ha='right')
    plt.ylim(0.0, 1.0)
    plt.ylabel('score')
    plt.title('Quality Metrics by Method')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'quality_scores.png'), dpi=160)
    plt.close()

def save_summary_json(results, path='results/logs/summary_results.json'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

def build_reports_from_logs(log_dir='results/logs', table_dir='results/tables', plot_dir='results/plots'):
    results = load_result_files(log_dir=log_dir)
    save_tables(results, table_dir=table_dir)
    save_plots(results, plot_dir=plot_dir)
    save_summary_json(results, path=os.path.join(log_dir, 'summary_results.json'))
    return results
