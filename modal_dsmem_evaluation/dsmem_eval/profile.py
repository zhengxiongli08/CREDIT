from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from .cost_model import DeviceProfile


FLOAT = r"([0-9]+(?:\.[0-9]+)?)"


def _match_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"primitive benchmark did not report {label}")
    return float(match.group(1))


def _match_int(pattern: str, text: str, label: str) -> int:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"primitive benchmark did not report {label}")
    return int(match.group(1))


def parse_primitive_output(output: str) -> dict[str, Any]:
    name_match = re.search(r"^GPU Model:\s*(.+)$", output, flags=re.MULTILINE)
    if name_match is None:
        raise RuntimeError("primitive benchmark did not report the GPU model")

    parsed: dict[str, Any] = {
        "name": name_match.group(1).strip(),
        "clock_ghz": _match_float(
            rf"^Clock frequency:\s*{FLOAT}\s+GHz", output, "clock frequency"
        ),
        "sm_count": _match_int(r"^SM count:\s*(\d+)", output, "SM count"),
        "l1_latency_cycles": _match_float(
            rf"^L1\s+read latency\s*:\s*{FLOAT}\s+cycles", output, "L1 latency"
        ),
        "l2_latency_cycles": _match_float(
            rf"^L2\s+read latency\s*:\s*{FLOAT}\s+cycles", output, "L2 latency"
        ),
        "dram_latency_cycles": _match_float(
            rf"^DRAM read latency\s*:\s*{FLOAT}\s+cycles", output, "DRAM latency"
        ),
        "l1_bytes_per_cycle_sm": _match_float(
            rf"^L1 cache bandwidth\s*:\s*{FLOAT}\s+B/cycle/SM",
            output,
            "L1 bandwidth",
        ),
        "l2_bytes_per_cycle": _match_float(
            rf"^L2 cache bandwidth\s*:\s*{FLOAT}\s+B/cycle",
            output,
            "L2 bandwidth",
        ),
        "dram_bandwidth_gbps": _match_float(
            rf"^Global memory bandwidth\s*:\s*{FLOAT}\s+GB/s",
            output,
            "DRAM bandwidth",
        ),
        "cluster_sync_cycles": _match_float(
            rf"^Cluster sync latency\s*:\s*{FLOAT}\s+cycles/sync",
            output,
            "cluster synchronization latency",
        ),
        "local_smem_read_latency_cycles": _match_float(
            rf"^Local SMEM read latency\s*:\s*{FLOAT}\s+cycles/load",
            output,
            "local SMEM latency",
        ),
        "dsmem_read_latency_cycles": _match_float(
            rf"^DSMEM read latency\s*:\s*{FLOAT}\s+cycles/load",
            output,
            "DSMEM read latency",
        ),
        "local_smem_read_bytes_per_cycle_cta": _match_float(
            rf"^Local SMEM read throughput\s*:\s*{FLOAT}\s+B/cycle/CTA",
            output,
            "local SMEM throughput",
        ),
        "dsmem_read_bytes_per_cycle_cta": _match_float(
            rf"^DSMEM read throughput\s*:\s*{FLOAT}\s+B/cycle/CTA",
            output,
            "DSMEM read throughput",
        ),
        "local_smem_store_bytes_per_cycle_cta": _match_float(
            rf"^Local SMEM store throughput\s*:\s*{FLOAT}\s+B/cycle/CTA",
            output,
            "local SMEM store throughput",
        ),
        "dsmem_store_bytes_per_cycle_cta": _match_float(
            rf"^DSMEM store issue throughput\s*:\s*{FLOAT}\s+B/cycle/CTA",
            output,
            "DSMEM store throughput",
        ),
        "store_visibility_roundtrip_cycles": _match_float(
            rf"^Store visibility round trip\s*:\s*{FLOAT}\s+cycles/roundtrip",
            output,
            "store visibility round trip",
        ),
    }
    return parsed


def make_device_profile(parsed: dict[str, Any], properties: Any) -> DeviceProfile:
    clock_ghz = float(parsed["clock_ghz"])
    profile = DeviceProfile(
        name=str(parsed["name"]),
        sm_count=int(properties.multi_processor_count),
        clock_ghz=clock_ghz,
        hbm_bandwidth_gbps=float(parsed["dram_bandwidth_gbps"]),
        l2_capacity_bytes=int(properties.L2_cache_size),
        l2_bandwidth_gbps=float(parsed["l2_bytes_per_cycle"]) * clock_ghz,
        local_smem_bytes_per_cycle_cta=float(
            parsed["local_smem_read_bytes_per_cycle_cta"]
        ),
        local_smem_store_bytes_per_cycle_cta=float(
            parsed["local_smem_store_bytes_per_cycle_cta"]
        ),
        dsmem_store_bytes_per_cycle_cta=float(
            parsed["dsmem_store_bytes_per_cycle_cta"]
        ),
        max_shared_bytes_per_cta=int(properties.shared_memory_per_block_optin),
    )
    return profile


def profile_as_dict(profile: DeviceProfile) -> dict[str, Any]:
    return asdict(profile)
