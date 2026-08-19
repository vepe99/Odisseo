from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "notebooks/scalability/galaxy_disk_fmm_large_n.py"
_spec = importlib.util.spec_from_file_location("galaxy_disk_fmm_large_n", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_mod)


_check_timing_gates = _mod._check_timing_gates


def _args(**overrides):
    base = dict(
        max_runtime_seconds=None,
        require_static_shape=False,
        max_compiled_profile_transitions=None,
        max_overflow_reprofiles=None,
        max_neighbor_edge_reprofiles=None,
        min_refresh_prepare_successes=None,
        min_adaptive_cadence_skips_rhs_calls=None,
        min_adaptive_cadence_skips_displacement=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_adaptive_cadence_rhs_calls_gate_trips_when_below_threshold():
    args = _args(min_adaptive_cadence_skips_rhs_calls=2)
    timing_stats = {"adaptive_core_refresh_cadence_skips_rhs_calls": 1}
    with pytest.raises(RuntimeError, match="rhs-calls gate"):
        _check_timing_gates(args, timing_stats)


def test_adaptive_cadence_displacement_gate_trips_when_below_threshold():
    args = _args(min_adaptive_cadence_skips_displacement=3)
    timing_stats = {"adaptive_core_refresh_cadence_skips_displacement": 2}
    with pytest.raises(RuntimeError, match="displacement gate"):
        _check_timing_gates(args, timing_stats)


def test_adaptive_cadence_gates_pass_when_thresholds_met():
    args = _args(
        min_adaptive_cadence_skips_rhs_calls=2,
        min_adaptive_cadence_skips_displacement=1,
    )
    timing_stats = {
        "script_runtime_seconds": 1.0,
        "adaptive_core_refresh_cadence_skips_rhs_calls": 2,
        "adaptive_core_refresh_cadence_skips_displacement": 1,
    }
    _check_timing_gates(args, timing_stats)
