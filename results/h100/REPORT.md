# NVIDIA H100 80GB HBM3 DSMEM Evaluation

- DSMEM beats the fastest baseline on 9/30 workload-size points.
- The calibrated model predicts 28/30 CUDA profitability signs correctly.
- At N=65536, DSMEM has a 1.318x geometric-mean speedup over the best baseline.

## Results at N=65536

| Workload | D/torch.compile | D/Triton | D/CUDA | D/best | Best baseline | P |
|---|---:|---:|---:|---:|---|---:|
| LayerNorm backward | 1.954x | 1.452x | 1.503x | 1.452x | triton | 8 |
| Weighted variance backward | 1.339x | 1.420x | 1.438x | 1.339x | torch.compile | 8 |
| Pearson backward | 2.068x | 1.860x | 1.457x | 1.457x | cuda_nodsmem | 8 |
| Softmax-logits backward | 1.459x | 1.533x | 1.809x | 1.459x | torch.compile | 8 |
| LARS momentum | 1.045x | 1.325x | 1.367x | 1.045x | torch.compile | 8 |
| Row-wise int8 quantization | 1.216x | 1.615x | 1.666x | 1.216x | torch.compile | 8 |

## Crossover Validation

Crossovers are relative to the best non-DSMEM CUDA path.

| Workload | Predicted | Measured |
|---|---:|---:|
| LayerNorm backward | 8192 | 8192 |
| Weighted variance backward | 8192 | 8192 |
| Pearson backward | 16384 | 8192 |
| Softmax-logits backward | 16384 | 16384 |
| LARS momentum | 8192 | 4096 |
| Row-wise int8 quantization | 32768 | 32768 |
