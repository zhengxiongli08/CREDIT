from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .workload_metadata import model_fields


TensorTuple = tuple[torch.Tensor, ...]
TorchFunction = Callable[..., torch.Tensor | TensorTuple]
InputFactory = Callable[[int, int, torch.device], TensorTuple]


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    label: str
    category: str
    rows: int
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


WORKLOADS = {
    "layernorm_backward": WorkloadSpec(
        name="layernorm_backward",
        rows=2048,
        torch_function=layernorm_backward,
        input_factory=make_layernorm_inputs,
        **model_fields("layernorm_backward"),
        atol=2.0e-2,
        rtol=2.0e-2,
    ),
    "weighted_var_backward": WorkloadSpec(
        name="weighted_var_backward",
        rows=4096,
        torch_function=weighted_var_backward,
        input_factory=make_weighted_var_inputs,
        **model_fields("weighted_var_backward"),
        atol=3.0e-2,
        rtol=3.0e-2,
    ),
    "pearson_backward": WorkloadSpec(
        name="pearson_backward",
        rows=4096,
        torch_function=pearson_backward,
        input_factory=make_pearson_inputs,
        **model_fields("pearson_backward"),
        atol=3.0e-3,
        rtol=3.0e-3,
    ),
    "softmax_logits_backward": WorkloadSpec(
        name="softmax_logits_backward",
        rows=4096,
        torch_function=softmax_logits_backward,
        input_factory=make_softmax_inputs,
        **model_fields("softmax_logits_backward"),
        atol=3.0e-3,
        rtol=3.0e-3,
    ),
    "lars_momentum": WorkloadSpec(
        name="lars_momentum",
        rows=4096,
        torch_function=lars_momentum,
        input_factory=make_lars_inputs,
        **model_fields("lars_momentum"),
        atol=3.0e-3,
        rtol=3.0e-3,
    ),
    "rowwise_quant": WorkloadSpec(
        name="rowwise_quant",
        rows=4096,
        torch_function=rowwise_quant,
        input_factory=make_quant_inputs,
        **model_fields("rowwise_quant"),
        atol=1.0e-6,
        rtol=1.0e-6,
    ),
}
