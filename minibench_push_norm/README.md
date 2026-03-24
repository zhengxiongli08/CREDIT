# minibench_push_norm

This benchmark is designed around the measured hardware asymmetry on your RTX 5090:

- DSMEM write bandwidth is very high.
- DSMEM read latency is better than L2/DRAM, but still not free.
- Cluster barriers are expensive enough that the handoff must save something substantial.

On your machine, [WINNING_CASES.md](WINNING_CASES.md) records a real win region where `fused_cluster` beats both `baseline` and `fused_local`.

## Pattern

The consumer needs the full producer tile before it can continue, because it computes an RMS-like normalization over the entire tile and then uses the normalized values in a second pass.

- `baseline`
  - Kernel 1 materializes the producer tile in global memory.
  - A cache-flush kernel makes that handoff colder.
  - Kernel 2 reloads the full tile, computes the normalization, and finishes the consumer stage.
- `fused_local`
  - A single CTA cannot keep the whole producer tile and the full consumer working set live at once.
  - It processes the tile in chunks, computes normalization in one pass, then recomputes the producer work in a second pass to finish the consumer stage.
- `fused_cluster`
  - Rank 0 produces the full tile once.
  - It pushes the full tile directly into rank 1's shared memory via DSMEM writes.
  - Rank 1 consumes the tile locally, with no DSMEM reads on the critical path.

This is the most faithful toy benchmark in the workspace to the idea "use the inter-SM network to keep a full producer-consumer handoff on-chip when a single block cannot do so cheaply." 

## Useful commands

```bash
./run.sh --mode all --groups 512 --tile-k 12288 --rows 8 --producer-iters 32
./sweep.sh
./profile.sh
```
