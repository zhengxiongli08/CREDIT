# 30-Second Pitch

NVIDIA DSMEM lets neighboring GPU thread blocks access one another's shared memory, but remote accesses and cluster barriers are expensive. CREDIT identifies the case where DSMEM pays: a wide row is reduced and then reread, while each CTA can keep its own slice local and exchange only scalar partials. A cost model screens candidates without timing a DSMEM workload kernel. Across six workloads on RTX 5090 and H100, it predicts 55 of 60 profitability signs; at width 65,536, DSMEM beats the fastest PyTorch, Triton, or CUDA baseline on every workload.

# Likely Questions

- **Why not use DSMEM for general kernel fusion?** Bulk remote traffic and synchronization usually erase the saved HBM time. CREDIT keeps bulk values owner-local and communicates only compact statistics.
- **What does the model require?** One non-DSMEM timing, independent primitive measurements, and static byte/statistic counts. It does not benchmark a DSMEM workload candidate.
- **Why do larger rows help?** Avoided reread traffic grows with row width, while compact DSMEM communication and cluster startup grow much more slowly.
- **Why is H100 different from RTX 5090?** H100 has a higher measured cluster synchronization cost and different remote-store behavior, so profitable crossovers occur later.
