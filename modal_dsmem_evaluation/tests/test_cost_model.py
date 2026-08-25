from __future__ import annotations

import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from dsmem_eval.cost_model import DeviceProfile, estimate


class CostModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = SimpleNamespace(
            rows=100,
            modeled_bytes_per_element=20.0,
            reread_bytes_per_element=10.0,
            staged_bytes_per_element=4,
            partial_count=2,
        )
        self.profile = DeviceProfile(
            name="test",
            sm_count=10,
            clock_ghz=1.0,
            hbm_bandwidth_gbps=1.0,
            l2_capacity_bytes=1,
            l2_bandwidth_gbps=1.0,
            local_smem_bytes_per_cycle_cta=10.0,
            local_smem_store_bytes_per_cycle_cta=100.0,
            dsmem_store_bytes_per_cycle_cta=10.0,
            max_shared_bytes_per_cta=100_000,
        )

    def test_uses_achieved_baseline_bandwidth(self) -> None:
        model = estimate(
            self.spec,
            cols=1000,
            cluster_size=2,
            control_overhead_ms=0.1,
            profile=self.profile,
            baseline_ms=2.0,
        )
        self.assertAlmostEqual(model.effective_bandwidth_gbps, 1.0)
        self.assertAlmostEqual(model.saved_reread_ms, 1.0)
        self.assertAlmostEqual(model.local_replay_ms, 0.004)
        self.assertAlmostEqual(model.local_store_residual_ms, 0.0)
        self.assertAlmostEqual(model.dsmem_store_ms, 0.000016)
        self.assertAlmostEqual(model.predicted_delta_ms, 0.895984)

        different_peaks = replace(
            self.profile,
            hbm_bandwidth_gbps=10_000.0,
            l2_bandwidth_gbps=20_000.0,
        )
        changed = estimate(
            self.spec,
            cols=1000,
            cluster_size=2,
            control_overhead_ms=0.1,
            profile=different_peaks,
            baseline_ms=2.0,
        )
        self.assertEqual(model, changed)

    def test_charges_unhidden_local_store_time(self) -> None:
        slow_store = replace(
            self.profile, local_smem_store_bytes_per_cycle_cta=0.05
        )
        model = estimate(
            self.spec,
            cols=1000,
            cluster_size=2,
            control_overhead_ms=0.1,
            profile=slow_store,
            baseline_ms=2.0,
        )
        self.assertAlmostEqual(model.local_store_residual_ms, 0.4)

    def test_rejects_invalid_baseline_time(self) -> None:
        for baseline_ms in (0.0, -1.0, math.nan):
            with self.subTest(baseline_ms=baseline_ms):
                with self.assertRaises(ValueError):
                    estimate(
                        self.spec,
                        cols=1000,
                        cluster_size=2,
                        control_overhead_ms=0.1,
                        profile=self.profile,
                        baseline_ms=baseline_ms,
                    )


if __name__ == "__main__":
    unittest.main()
