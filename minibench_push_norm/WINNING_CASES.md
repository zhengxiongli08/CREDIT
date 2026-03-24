# Winning cases on RTX 5090

These results come from `./sweep.sh` in this folder.

## Why this benchmark works

Your hardware benchmark shows:

- DSMEM write bandwidth is very high: about `8587 B/cycle`.
- DSMEM read latency (`220 cycles`) is much lower than L2 (`360 cycles`) and DRAM (`931 cycles`), but still not free.
- Cluster barriers cost about `992 cycles`, so the cluster handoff has to save substantial work.

This benchmark uses those facts directly:

- it avoids DSMEM reads on the critical path,
- it uses rank 0 to **push** the full producer tile into rank 1's shared memory,
- it makes the consumer require the **entire** tile before continuing,
- it forces the single-CTA local fusion path to chunk and recompute producer work.

## Sweep summary

Command:

```bash
./sweep.sh
```

Key winning region for `--groups 512 --tile-k 12288`:

| rows | producer_iters | baseline ms | local ms | cluster ms | cluster vs baseline | cluster vs local |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 32 | 0.549 | 0.130 | 0.117 | 4.698x | 1.110x |
| 4 | 64 | 0.556 | 0.190 | 0.135 | 4.131x | 1.410x |
| 8 | 8  | 0.556 | 0.156 | 0.133 | 4.176x | 1.171x |
| 8 | 16 | 0.557 | 0.185 | 0.133 | 4.199x | 1.392x |
| 8 | 32 | 0.561 | 0.232 | 0.135 | 4.141x | 1.711x |
| 8 | 64 | 0.566 | 0.334 | 0.154 | 3.668x | 2.163x |

## Interpretation

This is the scenario where inter-SM fusion helps:

- The producer output is expensive enough that recomputing it in a chunked local fused kernel hurts.
- The consumer needs the full tile, so ordinary single-CTA fusion cannot cheaply stream it chunk-by-chunk.
- Pushing the tile directly into the consumer block's shared memory makes the inter-SM network act like an on-chip handoff buffer.

That is a much stronger and more realistic argument than claiming cluster fusion should always beat a well-fitting single-CTA fused kernel.
