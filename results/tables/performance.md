## Performance Benchmark

| Method | Precision | P50 (ms) | P95 (ms) | Throughput | Size (MB) | Device | Status |
|---|---|---|---|---|---|---|---|
| Baseline FP16 | fp16 | 13.84 | 20.66 | 47.52 | 26.78 | mps | ok |
| torch.compile (reduce-overhead) | fp16 | 13.59 | 13.87 | 84.91 | 26.78 | mps | ok |
