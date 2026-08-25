from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class WorkloadModelSpec(Protocol):
    rows: int
    modeled_bytes_per_element: float
    reread_bytes_per_element: float
    staged_bytes_per_element: int
    partial_count: int


MODEL_VERSION = "effective-bandwidth-overlap-v2"


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    sm_count: int
    clock_ghz: float
    hbm_bandwidth_gbps: float
    l2_capacity_bytes: int
    l2_bandwidth_gbps: float
    local_smem_bytes_per_cycle_cta: float
    local_smem_store_bytes_per_cycle_cta: float
    dsmem_store_bytes_per_cycle_cta: float
    max_shared_bytes_per_cta: int


@dataclass(frozen=True)
class ModelEstimate:
    effective_bandwidth_gbps: float
    saved_reread_ms: float
    control_overhead_ms: float
    local_replay_ms: float
    local_store_residual_ms: float
    dsmem_store_ms: float
    predicted_delta_ms: float
    predicted_profitable: bool

    @property
    def saved_hbm_ms(self) -> float:
        """Compatibility alias for result bundles produced by the first model."""
        return self.saved_reread_ms

    @property
    def local_staging_ms(self) -> float:
        """Incremental local-SMEM cost after overlap with the input stream."""
        return self.local_replay_ms + self.local_store_residual_ms


def shared_bytes_per_cta(
    spec: WorkloadModelSpec, cols: int, cluster_size: int
) -> int:
    slice_elements = math.ceil(cols / cluster_size)
    return slice_elements * spec.staged_bytes_per_element


def estimate(
    spec: WorkloadModelSpec,
    cols: int,
    cluster_size: int,
    control_overhead_ms: float,
    profile: DeviceProfile,
    baseline_ms: float,
    rows: int | None = None,
) -> ModelEstimate:
    row_count = spec.rows if rows is None else rows
    if not math.isfinite(baseline_ms) or baseline_ms <= 0.0:
        raise ValueError("baseline_ms must be finite and positive")
    if spec.modeled_bytes_per_element <= 0.0:
        raise ValueError("modeled_bytes_per_element must be positive")

    # A non-DSMEM timing captures cache residency and achieved, rather than peak,
    # source bandwidth without consuming a DSMEM workload measurement.
    modeled_baseline_bytes = (
        row_count * cols * spec.modeled_bytes_per_element
    )
    effective_bandwidth_gbps = modeled_baseline_bytes / (baseline_ms * 1.0e6)
    saved_reread_ms = (
        baseline_ms
        * spec.reread_bytes_per_element
        / spec.modeled_bytes_per_element
    )

    resident_clusters = max(profile.sm_count // cluster_size, 1)
    waves = math.ceil(row_count / resident_clusters)
    bytes_per_source_cta = (
        4.0 * (cluster_size - 1) * spec.partial_count
    )
    store_cycles_per_wave = (
        bytes_per_source_cta / profile.dsmem_store_bytes_per_cycle_cta
    )
    dsmem_store_ms = (
        waves * store_cycles_per_wave / (profile.clock_ghz * 1.0e6)
    )
    staged_bytes_per_cta = (
        math.ceil(cols / cluster_size) * spec.staged_bytes_per_element
    )
    local_replay_ms = (
        waves
        * staged_bytes_per_cta
        / profile.local_smem_bytes_per_cycle_cta
        / (profile.clock_ghz * 1.0e6)
    )

    # The global-to-SMEM deposit is issued alongside the compulsory input pass.
    # Charge only the portion that cannot be hidden by that source stream.
    source_bytes_per_cycle_cta = effective_bandwidth_gbps / (
        profile.clock_ghz * profile.sm_count
    )
    store_residual_cycles_per_wave = staged_bytes_per_cta * max(
        1.0 / profile.local_smem_store_bytes_per_cycle_cta
        - 1.0 / source_bytes_per_cycle_cta,
        0.0,
    )
    local_store_residual_ms = (
        waves
        * store_residual_cycles_per_wave
        / (profile.clock_ghz * 1.0e6)
    )
    predicted_delta_ms = (
        saved_reread_ms
        - max(control_overhead_ms, 0.0)
        - local_replay_ms
        - local_store_residual_ms
        - dsmem_store_ms
    )
    return ModelEstimate(
        effective_bandwidth_gbps=effective_bandwidth_gbps,
        saved_reread_ms=saved_reread_ms,
        control_overhead_ms=max(control_overhead_ms, 0.0),
        local_replay_ms=local_replay_ms,
        local_store_residual_ms=local_store_residual_ms,
        dsmem_store_ms=dsmem_store_ms,
        predicted_delta_ms=predicted_delta_ms,
        predicted_profitable=predicted_delta_ms > 0.0,
    )
