## Performance Benchmark

| Method                          | Precision   |   P50 (ms) |   P95 (ms) |   Throughput |   Size (MB) | Device   | Status   |
|---------------------------------|-------------|------------|------------|--------------|-------------|----------|----------|
| ONNX Runtime                    | fp16        |       3.1  |       3.24 |       587.16 |       26.81 | mixed    | ok       |
| Baseline FP16                   | fp16        |       4.5  |       4.77 |       488.34 |       26.78 | cuda     | ok       |
| torch.compile (reduce-overhead) | fp16        |       3    |       3.21 |       735.98 |       26.78 | cuda     | ok       |
| Pruning (unstructured)          | fp16        |       4.73 |       5.14 |       487.41 |       26.78 | cuda     | ok       |
| PTQ (int8)                      | int8        |       6.17 |       6.27 |       236.85 |       13.53 | mixed    | ok       |
| 2:4 Semi-structured             | fp16        |       4.76 |       5.21 |       488.63 |       26.78 | cuda     | ok       |
