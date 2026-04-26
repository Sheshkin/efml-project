## Quality Metrics

| Method                          |   Dice |    IoU |   Pixel Acc | Sparsity   | Status   |
|---------------------------------|--------|--------|-------------|------------|----------|
| ONNX Runtime                    | 0.8943 | 0.8212 |      0.9447 | -          | ok       |
| Baseline FP16                   | 0.8944 | 0.8212 |      0.9447 | -          | ok       |
| torch.compile (reduce-overhead) | 0.8944 | 0.8212 |      0.9447 | -          | ok       |
| Pruning (unstructured)          | 0.8973 | 0.826  |      0.946  | 0.5000     | ok       |
| PTQ (int8)                      | 0.894  | 0.8206 |      0.9445 | -          | ok       |
| 2:4 Semi-structured             | 0.8852 | 0.8076 |      0.9395 | 0.5000     | ok       |
