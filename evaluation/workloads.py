from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


TensorTuple = tuple[torch.Tensor, ...]
TorchFunction = Callable[..., torch.Tensor | TensorTuple]
InputFactory = Callable[[int, int, torch.device], TensorTuple]


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    label: str
    category: str
    rows: int
    run_script: Path
    binary: Path
    torch_function: TorchFunction
    input_factory: InputFactory
    modeled_bytes_per_element: float
    reread_bytes_per_element: float
    staged_bytes_per_element: int
    reductions_per_stage: tuple[int, ...]
    atol: float
    rtol: float

    @property
    def barrier_count(self) -> int:
        return 2 * len(self.reductions_per_stage)

    @property
    def partial_count(self) -> int:
        return sum(self.reductions_per_stage)


def _uniform(shape, low: float, high: float, device: torch.device) -> torch.Tensor:
    out = torch.empty(shape, device=device, dtype=torch.float32)
    return out.uniform_(low, high)


def make_layernorm_inputs(rows: int, cols: int, device: torch.device) -> TensorTuple:
    return (
        _uniform((rows, cols), -1.0, 1.0, device),
        _uniform((rows, cols), -1.0, 1.0, device),
        _uniform((cols,), 0.75, 1.25, device),
    )


def layernorm_backward(x: torch.Tensor, dy: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    xmu = x - mean
    inv_std = torch.rsqrt((xmu * xmu).mean(dim=-1, keepdim=True) + 1.0e-5)
    xhat = xmu * inv_std
    dyg = dy * gamma
    mean_dyg = dyg.mean(dim=-1, keepdim=True)
    mean_dyg_xhat = (dyg * xhat).mean(dim=-1, keepdim=True)
    return (dyg - mean_dyg - xhat * mean_dyg_xhat) * inv_std


def make_weighted_var_inputs(rows: int, cols: int, device: torch.device) -> TensorTuple:
    return (
        _uniform((rows, cols), -2.0, 2.0, device),
        _uniform((rows, cols), 0.5, 1.5, device),
        _uniform((rows, cols), -1.0, 1.0, device),
    )


def weighted_var_backward(x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    sum_w = weight.sum(dim=-1, keepdim=True)
    inv_sum_w = 1.0 / (sum_w + 1.0e-5)
    mean = (weight * x).sum(dim=-1, keepdim=True) * inv_sum_w
    diff = x - mean
    variance = (weight * diff * diff).sum(dim=-1, keepdim=True) * inv_sum_w
    inv_std = torch.rsqrt(variance + 1.0e-5)
    sum_dy = dy.sum(dim=-1, keepdim=True)
    sum_dy_centered = (dy * diff).sum(dim=-1, keepdim=True)
    return inv_std * (
        dy
        - weight * inv_sum_w * sum_dy
        - weight * diff * inv_std * inv_std * inv_sum_w * sum_dy_centered
    )


def make_pearson_inputs(rows: int, cols: int, device: torch.device) -> TensorTuple:
    return (
        _uniform((rows, cols), -1.0, 1.0, device),
        _uniform((rows, cols), -1.0, 1.0, device),
        _uniform((rows, 1), 0.75, 1.25, device),
    )


def pearson_backward(
    x: torch.Tensor, y: torch.Tensor, grad: torch.Tensor
) -> TensorTuple:
    xc = x - x.mean(dim=-1, keepdim=True)
    yc = y - y.mean(dim=-1, keepdim=True)
    var_x = (xc * xc).sum(dim=-1, keepdim=True)
    var_y = (yc * yc).sum(dim=-1, keepdim=True)
    covariance = (xc * yc).sum(dim=-1, keepdim=True)
    inv_x = torch.rsqrt(var_x + 1.0e-6)
    inv_y = torch.rsqrt(var_y + 1.0e-6)
    inv_xy = inv_x * inv_y
    coeff_x = covariance * inv_x * inv_x * inv_x * inv_y
    coeff_y = covariance * inv_x * inv_y * inv_y * inv_y
    return (
        grad * (yc * inv_xy - xc * coeff_x),
        grad * (xc * inv_xy - yc * coeff_y),
    )


def make_softmax_inputs(rows: int, cols: int, device: torch.device) -> TensorTuple:
    return (
        _uniform((rows, cols), -4.0, 4.0, device),
        _uniform((rows, cols), -1.0, 1.0, device),
    )


def softmax_logits_backward(logits: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    dot = (probabilities * dy).sum(dim=-1, keepdim=True)
    return probabilities * (dy - dot)


def make_lars_inputs(rows: int, cols: int, device: torch.device) -> TensorTuple:
    return (
        _uniform((rows, cols), -1.0, 1.0, device),
        _uniform((rows, cols), -1.0, 1.0, device),
        _uniform((rows, cols), -1.0, 1.0, device),
    )


def lars_momentum(
    weight: torch.Tensor, grad: torch.Tensor, momentum: torch.Tensor
) -> TensorTuple:
    update = 0.9 * momentum + grad + 0.01 * weight
    weight_norm = torch.sqrt((weight * weight).sum(dim=-1, keepdim=True))
    update_norm = torch.sqrt((update * update).sum(dim=-1, keepdim=True))
    trust = 0.02 * weight_norm / (update_norm + 1.0e-6)
    return weight - 1.0e-3 * trust * update, update


def make_quant_inputs(rows: int, cols: int, device: torch.device) -> TensorTuple:
    return (_uniform((rows, cols), -1.0, 1.0, device),)


def rowwise_quant(x: torch.Tensor) -> TensorTuple:
    max_abs = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    inv_scale = torch.where(max_abs > 0.0, 127.0 / max_abs, torch.zeros_like(max_abs))
    quantized = torch.clamp(torch.round(x * inv_scale), -127.0, 127.0).to(torch.int8)
    return quantized, (max_abs / 127.0).squeeze(-1)


ROOT = Path(__file__).resolve().parents[1]
GOOD = ROOT / "good_workloads"


WORKLOADS = {
    "layernorm_backward": WorkloadSpec(
        name="layernorm_backward",
        label="LayerNorm backward",
        category="normalization",
        rows=2048,
        run_script=GOOD / "layernorm_backward_dsmem_explore/run_layernorm_backward.sh",
        binary=GOOD / "layernorm_backward_dsmem_explore/bin/layernorm_backward_bench",
        torch_function=layernorm_backward,
        input_factory=make_layernorm_inputs,
        modeled_bytes_per_element=32.0,
        reread_bytes_per_element=12.0,
        staged_bytes_per_element=4,
        reductions_per_stage=(2, 2),
        atol=2.0e-2,
        rtol=2.0e-2,
    ),
    "weighted_var_backward": WorkloadSpec(
        name="weighted_var_backward",
        label="Weighted variance backward",
        category="normalization",
        rows=4096,
        run_script=GOOD / "weighted_var_norm_backward_dsmem_explore/run_weighted_var_norm_backward.sh",
        binary=GOOD / "weighted_var_norm_backward_dsmem_explore/bin/weighted_var_norm_backward_bench",
        torch_function=weighted_var_backward,
        input_factory=make_weighted_var_inputs,
        modeled_bytes_per_element=40.0,
        reread_bytes_per_element=24.0,
        staged_bytes_per_element=12,
        reductions_per_stage=(2, 1, 2),
        atol=3.0e-2,
        rtol=3.0e-2,
    ),
    "pearson_backward": WorkloadSpec(
        name="pearson_backward",
        label="Pearson backward",
        category="pairwise statistics",
        rows=4096,
        run_script=GOOD / "pearson_backward_dsmem_explore/run_pearson_backward.sh",
        binary=GOOD / "pearson_backward_dsmem_explore/bin/pearson_backward_bench",
        torch_function=pearson_backward,
        input_factory=make_pearson_inputs,
        modeled_bytes_per_element=24.0,
        reread_bytes_per_element=8.0,
        staged_bytes_per_element=8,
        reductions_per_stage=(2, 3),
        atol=3.0e-3,
        rtol=3.0e-3,
    ),
    "softmax_logits_backward": WorkloadSpec(
        name="softmax_logits_backward",
        label="Softmax-logits backward",
        category="softmax",
        rows=4096,
        run_script=GOOD / "softmax_logits_backward_dsmem_explore/run_softmax_logits_backward.sh",
        binary=GOOD / "softmax_logits_backward_dsmem_explore/bin/softmax_logits_backward_bench",
        torch_function=softmax_logits_backward,
        input_factory=make_softmax_inputs,
        modeled_bytes_per_element=28.0,
        reread_bytes_per_element=16.0,
        staged_bytes_per_element=8,
        reductions_per_stage=(1, 1, 1),
        atol=3.0e-3,
        rtol=3.0e-3,
    ),
    "lars_momentum": WorkloadSpec(
        name="lars_momentum",
        label="LARS momentum",
        category="optimizer",
        rows=4096,
        run_script=GOOD / "lars_momentum_dsmem_explore/run_lars_momentum.sh",
        binary=GOOD / "lars_momentum_dsmem_explore/bin/lars_momentum_bench",
        torch_function=lars_momentum,
        input_factory=make_lars_inputs,
        modeled_bytes_per_element=32.0,
        reread_bytes_per_element=12.0,
        staged_bytes_per_element=12,
        reductions_per_stage=(2,),
        atol=3.0e-3,
        rtol=3.0e-3,
    ),
    "rowwise_quant": WorkloadSpec(
        name="rowwise_quant",
        label="Row-wise int8 quantization",
        category="quantization",
        rows=4096,
        run_script=GOOD / "rowwise_quant_dsmem_explore/run_rowwise_quant.sh",
        binary=GOOD / "rowwise_quant_dsmem_explore/bin/rowwise_quant_bench",
        torch_function=rowwise_quant,
        input_factory=make_quant_inputs,
        modeled_bytes_per_element=9.0,
        reread_bytes_per_element=4.0,
        staged_bytes_per_element=4,
        reductions_per_stage=(1,),
        atol=1.0e-6,
        rtol=1.0e-6,
    ),
}
