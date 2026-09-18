from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadMetadata:
    label: str
    category: str
    modeled_bytes_per_element: float
    reread_bytes_per_element: float
    staged_bytes_per_element: int
    reductions_per_stage: tuple[int, ...]


WORKLOAD_METADATA = {
    "layernorm_backward": WorkloadMetadata(
        "LayerNorm backward", "normalization", 32.0, 12.0, 4, (2, 2)
    ),
    "weighted_var_backward": WorkloadMetadata(
        "Weighted variance backward", "normalization", 40.0, 24.0, 12, (2, 1, 2)
    ),
    "pearson_backward": WorkloadMetadata(
        "Pearson backward", "pairwise statistics", 24.0, 8.0, 8, (2, 3)
    ),
    "softmax_logits_backward": WorkloadMetadata(
        "Softmax-logits backward", "softmax", 28.0, 16.0, 8, (1, 1, 1)
    ),
    "lars_momentum": WorkloadMetadata(
        "LARS momentum", "optimizer", 32.0, 12.0, 12, (2,)
    ),
    "rowwise_quant": WorkloadMetadata(
        "Row-wise int8 quantization", "quantization", 9.0, 4.0, 4, (1,)
    ),
}


def model_fields(name: str) -> dict[str, object]:
    metadata = WORKLOAD_METADATA[name]
    return {
        "label": metadata.label,
        "category": metadata.category,
        "modeled_bytes_per_element": metadata.modeled_bytes_per_element,
        "reread_bytes_per_element": metadata.reread_bytes_per_element,
        "staged_bytes_per_element": metadata.staged_bytes_per_element,
        "reductions_per_stage": metadata.reductions_per_stage,
    }

