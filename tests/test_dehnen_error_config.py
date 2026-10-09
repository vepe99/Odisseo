"""``fmm_mac_type="dehnen_error"`` reaches jaccpot with its accuracy target.

Dehnen's criterion needs ``adaptive_eps`` (eq 16a's relative force-accuracy target)
and the eq (16b) force scale; ODISSEO carries them as ``fmm_adaptive_eps`` and
``fmm_mac_force_scale_mode`` and the solver builder passes them on. Without an eps
the builder refuses rather than build a criterion with no target.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from odisseo.jaccpot_coupling import _build_fmm_solver
from odisseo.option_classes import SimulationConfig, SimulationParams


def _build(config):
    c = config
    return _build_fmm_solver(
        working_dtype=jnp.float32,
        config=c,
        params=SimulationParams(G=1.0, t_end=1.0),
        fmm_preset=c.fmm_preset,
        fmm_basis=c.fmm_basis,
        fmm_theta=c.fmm_theta,
        fmm_runtime_path=c.fmm_runtime_path,
        fmm_mac_type=c.fmm_mac_type,
        fmm_farfield_mode=c.fmm_farfield_mode,
        fmm_m2l_chunk_size=c.fmm_m2l_chunk_size,
        fmm_nearfield_mode=c.fmm_nearfield_mode,
        fmm_nearfield_edge_chunk_size=c.fmm_nearfield_edge_chunk_size,
        fmm_tree_build_mode=c.fmm_tree_build_mode,
        fmm_tree_leaf_target=c.fmm_tree_leaf_target,
        fmm_fixed_order=c.fmm_fixed_order,
        leaf_size=c.fmm_leaf_size,
        fmm_jit_tree=c.fmm_jit_tree,
        fmm_jit_traversal=c.fmm_jit_traversal,
        fmm_max_pair_queue=c.fmm_max_pair_queue,
        fmm_pair_process_block=c.fmm_pair_process_block,
        fmm_max_interactions_per_node=c.fmm_max_interactions_per_node,
        fmm_max_neighbors_per_leaf=c.fmm_max_neighbors_per_leaf,
        fmm_prepare_stage_memory_split_enabled=c.fmm_prepare_stage_memory_split_enabled,
        fmm_upward_leaf_batch_size=c.fmm_upward_leaf_batch_size,
    )


def test_dehnen_error_passes_its_accuracy_target():
    cfg = SimulationConfig(
        N_particles=64, fmm_mac_type="dehnen_error", fmm_adaptive_eps=3e-5
    )
    solver = _build(cfg)
    impl = solver._impl
    assert impl.adaptive_eps == pytest.approx(3e-5)
    assert impl.mac_force_scale_mode == "paper_fb"
    assert impl._flat_walk_criterion_active()


def test_dehnen_error_without_an_eps_is_refused():
    cfg = SimulationConfig(N_particles=64, fmm_mac_type="dehnen_error")
    with pytest.raises(ValueError, match="fmm_adaptive_eps"):
        _build(cfg)


def test_the_geometric_mac_ignores_the_criterion_fields():
    cfg = SimulationConfig(N_particles=64, fmm_adaptive_eps=3e-5)
    impl = _build(cfg)._impl
    assert impl.adaptive_eps is None
    assert not impl._flat_walk_criterion_active()
