# Cross-GPU DSMEM Evaluation

Largest width shared by every result bundle: N=65536.

## Performance

| Device | Wins/all points | Geomean at common N |
|---|---:|---:|
| RTX 5090 | 22/30 | 1.466x |
| H100 | 9/30 | 1.318x |

## Cost Model

| Device | Peak/additive | Revised | Revised accuracy |
|---|---:|---:|---:|
| RTX 5090 | 27/30 | 27/30 | 90.0% |
| H100 | 16/30 | 28/30 | 93.3% |

## Device Calibration

| Device | HBM GB/s | L2 GB/s | Local store B/cycle/CTA | DSMEM store B/cycle/CTA | Max SMEM/CTA |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | 1487.6 | 6764.9 | 33.01 | 2.14 | 101376 |
| H100 | 2687.4 | 7269.2 | 27.67 | 21.37 | 232448 |
