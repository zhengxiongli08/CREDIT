# NVIDIA GeForce RTX 5090 DSMEM Evaluation

- DSMEM beats the fastest baseline on 22/30 workload-size points.
- The calibrated model predicts 27/30 CUDA profitability signs correctly.
- At N=65536, DSMEM has a 1.466x geometric-mean speedup over the best baseline.

## Results at N=65536

| Workload | D/torch.compile | D/Triton | D/CUDA | D/best | Best baseline | P |
|---|---:|---:|---:|---:|---|---:|
| LayerNorm backward | 2.558x | 1.961x | 2.007x | 1.961x | triton | 8 |
| Weighted variance backward | 2.404x | 2.401x | 2.590x | 2.401x | triton | 8 |
| Pearson backward | 2.130x | 1.926x | 1.540x | 1.540x | cuda_nodsmem | 8 |
| Softmax-logits backward | 1.308x | 1.992x | 2.371x | 1.308x | torch.compile | 8 |
| LARS momentum | 1.014x | 1.578x | 1.626x | 1.014x | torch.compile | 8 |
| Row-wise int8 quantization | 1.029x | 1.851x | 1.953x | 1.029x | torch.compile | 4 |

## Crossover Validation

Crossovers are relative to the best non-DSMEM CUDA path.

| Workload | Predicted | Measured |
|---|---:|---:|
| LayerNorm backward | 8192 | 8192 |
| Weighted variance backward | 4096 | 8192 |
| Pearson backward | 4096 | 4096 |
| Softmax-logits backward | 4096 | 8192 |
| LARS momentum | 4096 | 4096 |
| Row-wise int8 quantization | 8192 | 16384 |
