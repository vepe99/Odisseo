"""The multi-GPU mesh lane, exercised on two FORCED CPU devices.

Everything that makes this lane hard to review -- the row-permutation trap, the
partition contract, the three-dispatch structure -- is reproducible without a GPU,
which is the point of testing it this way. Run with::

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
        python -m pytest tests/test_mesh_lane.py

Device count is fixed at first backend init, so this file needs its own pytest
invocation.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from odisseo.integration_api import INTEGRATION_LANES, resolve_lane  # noqa: E402
from odisseo.option_classes import SimulationConfig  # noqa: E402

mesh_coupling = pytest.importorskip("odisseo.mesh_coupling")
MeshOptions = mesh_coupling.MeshOptions

_TWO_DEVICES = jax.local_device_count() >= 2
needs_two = pytest.mark.skipif(
    not _TWO_DEVICES, reason="needs 2 devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)"
)


def _cfg(**kw):
    base = dict(acceleration_scheme=6, fmm_mesh=True, mesh_options=MeshOptions(dt=1e-3))
    base.update(kw)
    return SimulationConfig(**base)


# --------------------------------------------------------------------------- #
# the seam -- no devices needed
# --------------------------------------------------------------------------- #


def test_mesh_is_a_known_lane_and_wins_precedence():
    """`fmm_mesh` must be selectable, listed, and beat the other FMM flags."""
    assert "fmm_mesh" in INTEGRATION_LANES
    assert resolve_lane(_cfg()) == "fmm_mesh"
    assert resolve_lane(_cfg(fmm_blockstep=True)) == "fmm_mesh"
    assert resolve_lane(_cfg(fmm_differentiable=True)) == "fmm_mesh"


def test_every_resolved_lane_name_is_declared():
    """resolve_lane must never return a name absent from INTEGRATION_LANES."""
    for cfg in (_cfg(), _cfg(fmm_mesh=False), SimulationConfig(acceleration_scheme=0)):
        assert resolve_lane(cfg) in INTEGRATION_LANES


def test_mesh_lane_requires_options():
    """A missing MeshOptions must be refused, not guessed at."""
    from odisseo.integration_api import _integrate_fmm_mesh

    with pytest.raises(ValueError, match="mesh_options"):
        _integrate_fmm_mesh(None, None, _cfg(mesh_options=None), None)


def test_mesh_lane_rejects_conflicting_flags():
    """Co-set gradient or block-step flags must fail loudly, naming what is missing."""
    from odisseo.integration_api import _integrate_fmm_mesh

    with pytest.raises(NotImplementedError, match="gradient"):
        _integrate_fmm_mesh(None, None, _cfg(fmm_differentiable=True), None)
    with pytest.raises(NotImplementedError, match="rung"):
        _integrate_fmm_mesh(None, None, _cfg(fmm_blockstep=True), None)


@pytest.mark.parametrize(
    "kw", [dict(dt=0.0), dict(dt=1e-3, ndev=0), dict(dt=1e-3, theta=0.0),
           dict(dt=1e-3, working_dtype="float16"), dict(dt=1e-3, partitioner="nope")]
)
def test_mesh_options_validates(kw):
    """Bad option combinations must raise at construction, not at step 1."""
    with pytest.raises(ValueError):
        MeshOptions(**kw)


# --------------------------------------------------------------------------- #
# the partition contract
# --------------------------------------------------------------------------- #


def _disc(n, seed=3):
    rng = np.random.default_rng(seed)
    r = 10.0 * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    pos = np.stack([r * np.cos(th), r * np.sin(th), rng.normal(scale=0.2, size=n)], 1)
    return pos.astype(np.float32), rng.uniform(0.8, 1.2, n).astype(np.float32)


@needs_two
def test_partition_layout_is_geometry_independent():
    """`cap` and `counts` depend on (n, ndev, leaf) only -- never on positions.

    This is what makes a re-partition a host permutation plus a device_put rather
    than a recompile: the compiled evaluator is keyed on (config, ndev, cap, mesh),
    and none of those move when particles do.
    """
    n = 2048
    a, ma = _disc(n, seed=1)
    b, mb = _disc(n, seed=2)
    b = b * np.array([5.0, 0.2, 3.0], np.float32)  # a wildly different geometry
    pa = mesh_coupling.build_mesh_partition(a, ma, ndev=2, leaf_size=64)
    pb = mesh_coupling.build_mesh_partition(b, mb, ndev=2, leaf_size=64)
    assert pa.cap == pb.cap
    assert np.array_equal(np.asarray(pa.counts), np.asarray(pb.counts))


@needs_two
def test_padding_is_refused_by_default():
    """A partition that would pad must fail loudly, naming the trim that fixes it."""
    a, ma = _disc(1000)
    with pytest.raises(ValueError, match="pads"):
        mesh_coupling.build_mesh_partition(a, ma, ndev=3, leaf_size=64)


@needs_two
def test_rank_in_inverts_the_device_local_gid_sort():
    """`rank_in` must undo the per-device sort exactly, on every row.

    This is the arithmetic behind trap 1. If it is wrong the forces are still
    smooth and plausible, just attached to the wrong particles.
    """
    n = 1024
    a, ma = _disc(n)
    p = mesh_coupling.build_mesh_partition(a, ma, ndev=2, leaf_size=64)
    for d in range(p.ndev):
        sl = slice(d * p.cap, (d + 1) * p.cap)
        gid = np.asarray(p.gid_flat)[sl]
        rank = np.asarray(p.rank_in)[sl]
        assert np.array_equal(np.sort(gid)[rank], gid)


# --------------------------------------------------------------------------- #
# the traps, end to end on two CPU devices
# --------------------------------------------------------------------------- #


@needs_two
def test_a_naive_gid_read_is_wrong_even_with_zero_padding():
    """THE trap. Reading the force by the INPUT gid map corrupts nearly every row.

    `scatter_to_input_order`'s docstring says the maps "agree whenever no device is
    padded" -- true of them as maps, false of ROW ORDER. This test exists so nobody
    "simplifies" the aligner away: the failure is silent, the forces stay smooth and
    plausible, and they are wrong by tens of percent.
    """
    pytest.importorskip("jaccpot.distributed.fmm")
    import jax.numpy as jnp
    from jaccpot.distributed.fmm import DistributedFMMConfig, make_force_evaluator
    from yggdrax.distributed import make_mesh

    n = 1024
    pos, mass = _disc(n)
    p = mesh_coupling.build_mesh_partition(pos, mass, ndev=2, leaf_size=64)
    assert p.cap * p.ndev == p.n, "this test is about the ZERO-padding case"

    cfg = DistributedFMMConfig(leaf_size=64, theta=0.7, order=4, softening=0.05).resolved_for(
        p.cap, 2
    )
    mesh = make_mesh(2)
    ev = make_force_evaluator(cfg, 2, p.cap, mesh, jit=True)
    a_raw, gid_out, _ = ev(
        jnp.asarray(p.pos_flat), jnp.asarray(p.mass_flat),
        jnp.asarray(p.gid_flat), jnp.asarray(p.counts),
    )
    # The returned gid order is NOT the input gid order.
    disagree = int(np.sum(np.asarray(gid_out).reshape(-1) != np.asarray(p.gid_flat)))
    assert disagree > n // 2, (
        f"only {disagree}/{n} rows permuted -- if this ever reaches 0 the trap is "
        "gone and the aligner could be simplified, but verify that before doing it"
    )

    align = mesh_coupling.make_aligner(mesh)
    aligned = align(a_raw, gid_out, jnp.asarray(p.rank_in))
    mesh_coupling.verify_alignment(
        aligned, a_raw, gid_out, jnp.asarray(p.gid_flat), p.n
    )  # must not raise


@needs_two
def test_a_short_mesh_rollout_conserves_angular_momentum():
    """A real KDK rollout on two devices must conserve L to leapfrog accuracy."""
    pytest.importorskip("jaccpot.distributed.fmm")
    from odisseo.option_classes import SimulationParams

    n = 1024
    pos, mass = _disc(n)
    rng = np.random.default_rng(11)
    vel = rng.normal(scale=0.05, size=(n, 3)).astype(np.float32)
    state = np.stack([pos, vel], axis=1)

    opts = MeshOptions(dt=1e-3, ndev=2, leaf_size=64, theta=0.7, order=4,
                       softening=0.05, check_overflow_every=2)
    res = mesh_coupling.integrate_mesh_jaccpot(
        state, mass, SimulationConfig(acceleration_scheme=6),
        SimulationParams(G=1.0), options=opts, n_steps=6,
    )
    assert res.state.shape == (n, 2, 3)
    assert np.all(np.isfinite(res.state))
    m = mass.astype(np.float64)
    def L(s):
        return (m[:, None] * np.cross(s[:, 0, :].astype(np.float64),
                                      s[:, 1, :].astype(np.float64))).sum(0)
    rel = np.linalg.norm(L(res.state) - L(state)) / (
        (m * np.linalg.norm(np.cross(pos, vel), axis=1)).sum() + 1e-300)
    assert rel < 1e-4, f"dL/L = {rel:.3e}"
    assert res.first_diag, "the first force's diagnostics must be reported"


@needs_two
def test_the_step_is_three_dispatches_not_one_fused_program():
    """Trap 2, pinned structurally rather than by reproducing the deadlock.

    Fusing the force into one jit with the integrator puts the traversal buffers and
    the integrator temporaries in a single live range; at 21M on five cards one
    device fails an allocation, never joins the AllGather, and the rest hang at the
    rendezvous forever. Asserting on the SOURCE is the cheap proxy: drift and kick
    must each be their own jitted function.
    """
    import inspect

    src = inspect.getsource(mesh_coupling.integrate_mesh_jaccpot)
    assert "donate_argnums" in src
    assert src.count("@partial(jax.jit") >= 2, "drift and kick must be separate jits"
    assert "def kdk" not in src, "a single fused kdk is the deadlock this avoids"
