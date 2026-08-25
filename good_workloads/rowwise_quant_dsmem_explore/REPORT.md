# Row-wise Dynamic Quantization DSMEM Exploration

## Workload

This folder benchmarks per-row dynamic int8 quantization:

```text
max_abs = max(abs(x))
scale = max_abs / 127
q = clamp(round(x * 127 / max_abs), -127, 127).to(int8)
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` once to reduce `max(abs(x))`,
  then rereads `x` to write the int8 output and one scale per row.
- `block_smem`: one CTA per row with local shared-memory staging of the whole
  row when it fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x`, uses DSMEM only for the scalar max-abs reduction, then writes its local
  int8 output slice.
- `torch.compile`: compiled PyTorch expression for the same quantization
  formula.

Throughput uses a 9 B/element model for the logical no-DSMEM read path:
two float input reads plus one int8 output write. The per-row float scale is
included in the timing and throughput model, but is negligible for wide rows.

## Timing Results

Command:

```bash
bash rowwise_quant_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv rowwise_quant_dsmem_explore/results/rowwise_quant_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.016288 | 0.201082 | 0.063840 | 0.081x | 0.317x |
| 8192 | block_smem | 0.087974 | 0.243194 | 0.100141 | 0.362x | 0.412x |
| 16384 | block_smem | 0.202618 | 0.264531 | 0.210944 | 0.766x | 0.797x |
| 32768 | block_read | 0.767494 | 0.402432 | 0.418291 | 1.907x | 1.039x |
| 65536 | block_read | 1.574160 | 0.826278 | 1.282470 | 1.905x | 1.552x |

The DSMEM cluster kernel beats `torch.compile` on 2 of 5 shapes. The first
three shapes fit the one-CTA `block_smem` strategy well, so the cluster launch
and synchronization overhead is not worthwhile. DSMEM becomes useful once rows
are too wide for whole-row local shared-memory staging.

## Nsight Compute

Command:

```bash
bash rowwise_quant_dsmem_explore/ncu_rowwise_quant_metrics.sh
python3 rowwise_quant_dsmem_explore/parse_rowwise_quant_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 1588.512 | 9.003 | 8.994 | 7.981 | 1.012 |
| cluster | 825.600 | 5.247 | 5.018 | 4.000 | 1.018 |

The DSMEM path reduces DRAM traffic from `9.0` to `5.0 B/element` and reduces
NCU kernel time by `1.924x` on the profiled shape. Output traffic is unchanged;
DSMEM removes the second global read of the float input row.

## Verdict

Row-wise dynamic quantization is a positive DSMEM case only for very wide rows.
It is a useful addition because the output is much smaller than the input:
DSMEM can still help when the main opportunity is avoiding a second float input
read, but local shared-memory staging is better while the whole row fits inside
one CTA.

Artifacts:

- `results/rowwise_quant_torch_compare.csv`
- `results/rowwise_quant_ncu_summary.csv`
- `plots/rowwise_quant_speedup.png`
- `plots/rowwise_quant_runtime_ms.png`
- `plots/rowwise_quant_gbps.png`
- `plots/rowwise_quant_ncu_metrics.png`
