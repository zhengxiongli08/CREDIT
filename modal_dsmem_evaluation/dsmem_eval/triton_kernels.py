from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _layernorm_backward_kernel(
    x_ptr,
    dy_ptr,
    gamma_ptr,
    out_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    sum_x = tl.zeros([BLOCK_SIZE], tl.float32)
    sum_x2 = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        sum_x += x
        sum_x2 += x * x
    mean = tl.sum(sum_x, axis=0) / n_cols
    variance = tl.maximum(tl.sum(sum_x2, axis=0) / n_cols - mean * mean, 0.0)
    inv_std = tl.rsqrt(variance + eps)

    sum_dyg = tl.zeros([BLOCK_SIZE], tl.float32)
    sum_dyg_xhat = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        dy = tl.load(dy_ptr + row_offset + cols, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0)
        dyg = dy * gamma
        xhat = (x - mean) * inv_std
        sum_dyg += dyg
        sum_dyg_xhat += dyg * xhat
    mean_dyg = tl.sum(sum_dyg, axis=0) / n_cols
    mean_dyg_xhat = tl.sum(sum_dyg_xhat, axis=0) / n_cols

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        dy = tl.load(dy_ptr + row_offset + cols, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0)
        xhat = (x - mean) * inv_std
        dyg = dy * gamma
        out = (dyg - mean_dyg - xhat * mean_dyg_xhat) * inv_std
        tl.store(out_ptr + row_offset + cols, out, mask=mask)


@triton.jit
def _weighted_var_backward_kernel(
    x_ptr,
    weight_ptr,
    dy_ptr,
    out_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    sum_w = tl.zeros([BLOCK_SIZE], tl.float32)
    sum_wx = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + row_offset + cols, mask=mask, other=0.0)
        sum_w += weight
        sum_wx += weight * x
    reduced_w = tl.sum(sum_w, axis=0)
    inv_sum_w = 1.0 / (reduced_w + eps)
    mean = tl.sum(sum_wx, axis=0) * inv_sum_w

    sum_var = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + row_offset + cols, mask=mask, other=0.0)
        diff = x - mean
        sum_var += weight * diff * diff
    inv_std = tl.rsqrt(tl.sum(sum_var, axis=0) * inv_sum_w + eps)

    sum_dy = tl.zeros([BLOCK_SIZE], tl.float32)
    sum_dy_centered = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        dy = tl.load(dy_ptr + row_offset + cols, mask=mask, other=0.0)
        sum_dy += dy
        sum_dy_centered += dy * (x - mean)
    reduced_dy = tl.sum(sum_dy, axis=0)
    reduced_dy_centered = tl.sum(sum_dy_centered, axis=0)

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + row_offset + cols, mask=mask, other=0.0)
        dy = tl.load(dy_ptr + row_offset + cols, mask=mask, other=0.0)
        diff = x - mean
        out = inv_std * (
            dy
            - weight * inv_sum_w * reduced_dy
            - weight
            * diff
            * inv_std
            * inv_std
            * inv_sum_w
            * reduced_dy_centered
        )
        tl.store(out_ptr + row_offset + cols, out, mask=mask)


@triton.jit
def _pearson_backward_kernel(
    x_ptr,
    y_ptr,
    grad_ptr,
    dx_ptr,
    dy_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    sum_x = tl.zeros([BLOCK_SIZE], tl.float32)
    sum_y = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        sum_x += tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        sum_y += tl.load(y_ptr + row_offset + cols, mask=mask, other=0.0)
    mean_x = tl.sum(sum_x, axis=0) / n_cols
    mean_y = tl.sum(sum_y, axis=0) / n_cols

    sum_x2 = tl.zeros([BLOCK_SIZE], tl.float32)
    sum_y2 = tl.zeros([BLOCK_SIZE], tl.float32)
    sum_xy = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0) - mean_x
        y = tl.load(y_ptr + row_offset + cols, mask=mask, other=0.0) - mean_y
        sum_x2 += x * x
        sum_y2 += y * y
        sum_xy += x * y
    var_x = tl.sum(sum_x2, axis=0)
    var_y = tl.sum(sum_y2, axis=0)
    covariance = tl.sum(sum_xy, axis=0)
    inv_x = tl.rsqrt(var_x + eps)
    inv_y = tl.rsqrt(var_y + eps)
    inv_xy = inv_x * inv_y
    coeff_x = covariance * inv_x * inv_x * inv_x * inv_y
    coeff_y = covariance * inv_x * inv_y * inv_y * inv_y
    grad = tl.load(grad_ptr + row)

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0) - mean_x
        y = tl.load(y_ptr + row_offset + cols, mask=mask, other=0.0) - mean_y
        dx = grad * (y * inv_xy - x * coeff_x)
        dy = grad * (x * inv_xy - y * coeff_y)
        tl.store(dx_ptr + row_offset + cols, dx, mask=mask)
        tl.store(dy_ptr + row_offset + cols, dy, mask=mask)


@triton.jit
def _softmax_logits_backward_kernel(
    logits_ptr,
    dy_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    partial_max = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        logits = tl.load(logits_ptr + row_offset + cols, mask=mask, other=-float("inf"))
        partial_max = tl.maximum(partial_max, logits)
    row_max = tl.max(partial_max, axis=0)

    partial_exp = tl.zeros([BLOCK_SIZE], tl.float32)
    partial_exp_dy = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        logits = tl.load(logits_ptr + row_offset + cols, mask=mask, other=-float("inf"))
        dy = tl.load(dy_ptr + row_offset + cols, mask=mask, other=0.0)
        exp_value = tl.exp(logits - row_max)
        partial_exp += exp_value
        partial_exp_dy += exp_value * dy
    denominator = tl.sum(partial_exp, axis=0)
    dot = tl.sum(partial_exp_dy, axis=0) / denominator

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        logits = tl.load(logits_ptr + row_offset + cols, mask=mask, other=-float("inf"))
        dy = tl.load(dy_ptr + row_offset + cols, mask=mask, other=0.0)
        probability = tl.exp(logits - row_max) / denominator
        tl.store(out_ptr + row_offset + cols, probability * (dy - dot), mask=mask)


@triton.jit
def _lars_momentum_kernel(
    weight_ptr,
    grad_ptr,
    momentum_ptr,
    weight_out_ptr,
    momentum_out_ptr,
    n_cols,
    eps: tl.constexpr,
    momentum_coeff: tl.constexpr,
    weight_decay: tl.constexpr,
    learning_rate: tl.constexpr,
    trust_coeff: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    sum_weight2 = tl.zeros([BLOCK_SIZE], tl.float32)
    sum_update2 = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        weight = tl.load(weight_ptr + row_offset + cols, mask=mask, other=0.0)
        grad = tl.load(grad_ptr + row_offset + cols, mask=mask, other=0.0)
        momentum = tl.load(momentum_ptr + row_offset + cols, mask=mask, other=0.0)
        update = momentum_coeff * momentum + grad + weight_decay * weight
        sum_weight2 += weight * weight
        sum_update2 += update * update
    weight_norm = tl.sqrt(tl.sum(sum_weight2, axis=0))
    update_norm = tl.sqrt(tl.sum(sum_update2, axis=0))
    trust = trust_coeff * weight_norm / (update_norm + eps)

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        weight = tl.load(weight_ptr + row_offset + cols, mask=mask, other=0.0)
        grad = tl.load(grad_ptr + row_offset + cols, mask=mask, other=0.0)
        momentum = tl.load(momentum_ptr + row_offset + cols, mask=mask, other=0.0)
        update = momentum_coeff * momentum + grad + weight_decay * weight
        tl.store(weight_out_ptr + row_offset + cols, weight - learning_rate * trust * update, mask=mask)
        tl.store(momentum_out_ptr + row_offset + cols, update, mask=mask)


@triton.jit
def _rowwise_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    n_cols,
    qmax: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    partial_max = tl.zeros([BLOCK_SIZE], tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        partial_max = tl.maximum(partial_max, tl.abs(x))
    max_abs = tl.max(partial_max, axis=0)
    inv_scale = tl.where(max_abs > 0.0, qmax / max_abs, 0.0)
    tl.store(scale_ptr + row, max_abs / qmax)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        rounded = libdevice.rint(x * inv_scale)
        quantized = tl.maximum(tl.minimum(rounded, qmax), -qmax).to(tl.int8)
        tl.store(q_ptr + row_offset + cols, quantized, mask=mask)


def launch_layernorm_backward(inputs, outputs, block_size: int, num_warps: int) -> None:
    x, dy, gamma = inputs
    (out,) = outputs
    _layernorm_backward_kernel[(x.shape[0],)](
        x,
        dy,
        gamma,
        out,
        x.shape[1],
        eps=1.0e-5,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def launch_weighted_var_backward(inputs, outputs, block_size: int, num_warps: int) -> None:
    x, weight, dy = inputs
    (out,) = outputs
    _weighted_var_backward_kernel[(x.shape[0],)](
        x,
        weight,
        dy,
        out,
        x.shape[1],
        eps=1.0e-5,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def launch_pearson_backward(inputs, outputs, block_size: int, num_warps: int) -> None:
    x, y, grad = inputs
    dx, dy = outputs
    _pearson_backward_kernel[(x.shape[0],)](
        x,
        y,
        grad,
        dx,
        dy,
        x.shape[1],
        eps=1.0e-6,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def launch_softmax_logits_backward(inputs, outputs, block_size: int, num_warps: int) -> None:
    logits, dy = inputs
    (out,) = outputs
    _softmax_logits_backward_kernel[(logits.shape[0],)](
        logits,
        dy,
        out,
        logits.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def launch_lars_momentum(inputs, outputs, block_size: int, num_warps: int) -> None:
    weight, grad, momentum = inputs
    weight_out, momentum_out = outputs
    _lars_momentum_kernel[(weight.shape[0],)](
        weight,
        grad,
        momentum,
        weight_out,
        momentum_out,
        weight.shape[1],
        eps=1.0e-6,
        momentum_coeff=0.9,
        weight_decay=0.01,
        learning_rate=1.0e-3,
        trust_coeff=0.02,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def launch_rowwise_quant(inputs, outputs, block_size: int, num_warps: int) -> None:
    (x,) = inputs
    q, scale = outputs
    _rowwise_quant_kernel[(x.shape[0],)](
        x,
        q,
        scale,
        x.shape[1],
        qmax=127.0,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


TRITON_LAUNCHERS = {
    "layernorm_backward": launch_layernorm_backward,
    "weighted_var_backward": launch_weighted_var_backward,
    "pearson_backward": launch_pearson_backward,
    "softmax_logits_backward": launch_softmax_logits_backward,
    "lars_momentum": launch_lars_momentum,
    "rowwise_quant": launch_rowwise_quant,
}


def allocate_outputs(name: str, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    x = inputs[0]
    if name == "pearson_backward":
        return torch.empty_like(x), torch.empty_like(x)
    if name == "lars_momentum":
        return torch.empty_like(x), torch.empty_like(x)
    if name == "rowwise_quant":
        return (
            torch.empty_like(x, dtype=torch.int8),
            torch.empty((x.shape[0],), device=x.device, dtype=torch.float32),
        )
    return (torch.empty_like(x),)
