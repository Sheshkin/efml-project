## Validation Checks

| Check                              | Status   | Details                                                                                                              |
|------------------------------------|----------|----------------------------------------------------------------------------------------------------------------------|
| checkpoint/data consistency        | ok       | Core rows use the same checkpoint hash and test data signature.                                                      |
| baseline/compile/onnx Dice parity  | ok       | baseline=0.8944, compile_delta=0.0000, onnx_delta=0.0000                                                             |
| ONNX Runtime backend               | ok       | p50_ratio=0.69x, provider_counts={'CUDAExecutionProvider': 61, 'CPUExecutionProvider': 80}, mixed CUDA/CPU execution |
| PTQ size/backend                   | ok       | size_ratio=0.25, latency_vs_onnx=1.99x, no measured INT8 speedup, runtime_device=mixed                               |
| unstructured pruning speedup claim | ok       | Weights are zeroed in dense Conv2d tensors; no sparse kernels are used.                                              |
| 2:4 sparsity backend support       | ok       | sm_70 does not support NVIDIA 2:4 sparse tensor core speedups (V100 is sm_70).                                       |
