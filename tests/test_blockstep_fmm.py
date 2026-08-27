"""The momentum-conserving block-step FMM lane (Nornax + Jaccpot).

The point of :mod:`odisseo.blockstep_coupling` is a *property*, so these tests
assert the property rather than that the code runs:

* **Momentum** is conserved to round-off (~1e-13 relative, and in practice
  ~1e-17), *structurally* -- the residual does not move when ``theta`` and the
  expansion order move the force error by orders of magnitude. That sweep is the
  sharp form of the claim and is what a kernel recomputing ``dr`` instead of
  negating it would fail.
* **The per-level split** conserves momentum level by level, not merely in
  total. This is the property the production coupler cannot supply *even in
  principle*, and measurement here says it is the only one: both lanes' total
  momentum residuals land at ~1e-17, so a test asserting an order-of-magnitude
  gap on the total would fail for reasons unrelated to the code.
* **Energy** is *bounded*, not conserved. Leapfrog is symplectic, so the drift
  oscillates; the test rejects secular growth specifically, which a bound alone
  would not.
* The block-step force agrees with the existing ``OdisseoFMMCoupler`` force on
  the **total** acceleration to FMM tolerance. Not per level -- the mutual far
  field splits at cell granularity and a direct-sum oracle splits per pair.
* The cheap oracle is nornax's ``MutualDirectSumGravity``, momentum-exact by
  construction.
* Gradients are exact at fixed topology, checked against finite differences of
  the *same* frozen plan.

Two traps these tests are built around
--------------------------------------
**A test that passes for the wrong reason.** A configuration with no far pairs
makes the FMM a direct sum, so an accuracy assertion passes at 1e-16 while
testing nothing. Every test that touches a far-field number goes through
``_assert_far_field_is_exercised``, and :func:`test_the_far_field_is_not_vacuous`
pins that the chosen ``theta``/``leaf_size`` really do produce far pairs. A
single N = 256 Plummer sphere produces **none** at any ``theta`` up to 1.1 -- the
tree is too shallow for the mutual MAC to fire -- so the IC here is two
well-separated clumps instead, which puts a genuinely far node pair in the tree
at ODISSEO's production ``theta = 0.6``.

**Fusion falling back silently.** The per-level path computes the same
trajectory at one traversal per active level instead of one per boundary, so no
correctness test can see it. :func:`test_nornax_selects_the_fused_boundary_path`
asserts the selection directly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from odisseo import construct_initial_state  # noqa: E402
from odisseo.blockstep_coupling import (  # noqa: E402
    BlockStepOptions,
    _assert_far_field_is_exercised,
    _check_rung_range,
    assert_fused_boundary_selected,
    blockstep_initial_state,
    blockstep_total_acceleration,
    build_blockstep_force,
    chunked_potential_energy,
    integrate_blockstep_jaccpot,
    integrate_blockstep_jitted,
    total_linear_momentum,
)
from odisseo.option_classes import (  # noqa: E402
    NFW_POTENTIAL,
    SimulationConfig,
    SimulationParams,
)

# The mutual MAC accepts when theta * |c_B - c_A| > R_A + R_B. It is symmetric
# and hence stricter than the target-centric criterion at the same numeric
# value, and it is the *tree depth* that decides whether it can ever fire. On a
# single N = 256 Plummer sphere it fires nowhere: zero far pairs at every theta
# up to 1.1 and every leaf size tried, which would make every far-field number
# below vacuous. The two-clump IC below fixes that structurally instead of by
# loosening theta -- measured 18 far and 96 near pairs at this configuration,
# with a 2e-4 force error against an exact direct sum.
N_PARTICLES = 256
K_MAX = 2
THETA = 0.6
LEAF_SIZE = 16
MAX_ORDER = 4
SOFTENING = 1.0e-2
# Large enough that the rung criterion dt_i = eta * sqrt(softening / |a_i|)
# spreads particles over several rungs. Measured occupancies on this IC:
# 1e-2 -> [254, 2, 0] (rung 2 empty), 2e-2 -> [229, 25, 2], 4e-2 -> [36, 193, 27].
# Below ~1e-2 everything lands on rung 0 and the block scheme silently collapses
# to a shared timestep -- a rollout that passes the momentum assertion without
# ever exercising a cross-level kick. Above ~4e-2 this IC has an under-resolved
# close encounter, which is a fine thing to *measure* (see the rung-ladder test)
# and a bad baseline for everything else.
DT_MAX = 2.0e-2


def _nornax_has_fused_boundary() -> bool:
    """Whether the installed nornax carries the fused-boundary primitive (#8).

    Catches ``Exception`` rather than ``ImportError`` so that a future
    import-time incompatibility reports as *skipped* rather than *errored*.
    """
    try:
        from nornax.solvers import fused_boundary_model  # noqa: F401
    except Exception:
        return False
    return True


def _jaccpot_has_blockstep_fmm() -> bool:
    try:
        import jaccpot
    except Exception:
        return False
    return hasattr(jaccpot, "BlockStepFMM")


pytestmark = pytest.mark.skipif(
    not (_nornax_has_fused_boundary() and _jaccpot_has_blockstep_fmm()),
    reason=(
        "needs jaccpot.BlockStepFMM and a nornax carrying the fused-boundary "
        "primitive (nornax main >= 8fe9dbd, 'Differentiable individual-timestep "
        "KDK leapfrog integrator (#7)')"
    ),
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _two_plummer_clumps(n=N_PARTICLES, seed=0, separation=8.0):
    """Two well-separated Plummer clumps -- the far field's own test bench.

    A single sphere at this N is a bad test system, and it took a measurement to
    see why: the tree is shallow enough that the mutual MAC accepts almost
    nothing, so ``theta`` had to be pushed to 0.9 before any far pair existed, at
    which point the FMM's force error is ~4e-3 and no meaningful tolerance can be
    asserted. Two clumps put a genuinely well-separated node pair in the tree at
    ODISSEO's *production* ``theta = 0.6``: 18 far pairs, 96 near, and a force
    error of 2e-4 against an exact direct sum.

    Masses are deliberately **unequal**. With equal masses and a near-field-only
    force, a target-centric gather is accidentally antisymmetric to the last bit
    (the prefactor ``G m_j / r^3`` is the same number for both endpoints), which
    would let a momentum comparison pass for a reason unrelated to the mutual
    restructure.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    blocks = []
    for centre in (
        np.array([-separation / 2.0, 0.0, 0.0]),
        np.array([separation / 2.0, 0.0, 0.0]),
    ):
        u = rng.uniform(1e-3, 0.9, size=half)
        r = 1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
        cos_t = rng.uniform(-1.0, 1.0, size=half)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=half)
        sin_t = np.sqrt(1.0 - cos_t**2)
        blocks.append(
            np.stack(
                [r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t], -1
            )
            + centre
        )
    pos = np.concatenate(blocks)
    # A cool, non-zero-momentum velocity field: the momentum assertion must not
    # be able to pass by everything sitting still.
    vel = rng.normal(scale=0.15, size=(n, 3)) + np.array([0.03, -0.02, 0.01])
    mass = rng.uniform(0.5, 1.5, size=n) / n
    return (
        jnp.asarray(pos, dtype=jnp.float64),
        jnp.asarray(vel, dtype=jnp.float64),
        jnp.asarray(mass, dtype=jnp.float64),
    )


@pytest.fixture(scope="module")
def system():
    pos, vel, mass = _two_plummer_clumps()
    return construct_initial_state(pos, vel), mass


@pytest.fixture(scope="module")
def config():
    return SimulationConfig(N_particles=N_PARTICLES, softening=SOFTENING)


@pytest.fixture(scope="module")
def params():
    return SimulationParams(G=1.0)


def _options(**kw):
    """Options for the HOST lane, which is what most of this module tests.

    `topology_backend` is pinned rather than left to the default, and that is
    deliberate: the default is `"auto"`, which resolves to `"device"` wherever the
    installed jaccpot can provide it, and `integrate_blockstep_jaccpot` then routes
    to the jitted lane. Most tests below are *about* the host lane -- the nornax
    fusion switches, `rebuild_every`, `scan_base_steps`, the per-step `progress`
    callback -- so letting the default carry them onto the other lane would leave
    those behaviours untested while still passing. The device lane has its own
    tests further down; pass `topology_backend="device"` (or "auto") to reach it.
    """
    base = dict(
        dt_max=DT_MAX,
        k_max=K_MAX,
        theta=THETA,
        max_order=MAX_ORDER,
        leaf_size=LEAF_SIZE,
        topology_backend="host",
    )
    base.update(kw)
    return BlockStepOptions(**base)


@pytest.fixture(scope="module")
def prepared(system, config, params):
    """A force model with its topology built on the initial positions."""
    state, mass = system
    force = build_blockstep_force(config, params, _options())
    force.prepare(state[:, 0, :], mass)
    return force


# --------------------------------------------------------------------------
# the two traps
# --------------------------------------------------------------------------


def test_the_far_field_is_not_vacuous(prepared):
    """Assert the far list is non-empty before trusting any far-field number.

    With no far pairs the FMM degenerates to a direct sum and every accuracy
    assertion in this file would pass at 1e-16 while testing nothing. This has
    produced a flattering ``0.0e+00`` in this code's history, so it is checked
    explicitly rather than assumed.
    """
    num_far = _assert_far_field_is_exercised(prepared)
    assert num_far > 0
    # The near list must also be non-empty, or the *near* kernel is the vacuous
    # one -- both halves of the force have to be exercised for the accuracy
    # comparisons below to mean anything.
    assert int(getattr(prepared.state, "num_near_pairs", 0)) > 0


def test_a_no_far_pair_configuration_is_rejected(system, config, params):
    """The vacuity guard fires, and reads **occupancy** rather than capacity.

    The empty far list is constructed directly instead of chased with a strict
    ``theta``: which theta empties it depends on the IC and the tree, so a test
    written that way skips itself the moment either changes -- and a guard that
    silently stops being tested is exactly the failure this guard exists to
    prevent.

    The occupancy half matters because capacity padding broke the obvious
    implementation. Once the pair lists are padded, ``far_a.shape[0]`` is the
    allocated width and stays nonzero for a topology holding no far pairs at all,
    so a guard written against the shape stops guarding the moment padding is
    switched on. This builds the state that separates the two.
    """
    import dataclasses

    state, mass = system
    force = build_blockstep_force(config, params, _options(jit_force=True))
    force.prepare(state[:, 0, :], mass)
    inner = force.state
    assert inner.far_capacity >= inner.num_far_pairs > 0
    assert inner.near_capacity >= inner.num_near_pairs > 0
    # Padding is really present here, so the distinction is not academic.
    assert inner.far_capacity > inner.num_far_pairs

    class _EmptyFarField:
        """Occupancy zero, capacity nonzero -- what the shape check misses."""

        state = dataclasses.replace(inner, num_far_pairs=0)

    with pytest.raises(RuntimeError, match="no far pairs"):
        _assert_far_field_is_exercised(_EmptyFarField())


def test_nornax_selects_the_fused_boundary_path(prepared):
    """The fused path must be *selected*, not merely available.

    A silent fallback to the per-level path produces the same trajectory while
    paying ``sum_s (active levels at s)`` traversals per base step instead of
    ``n_sub + 1`` -- 19 against 9 at ``k_max = 3``. No correctness test can
    catch that, so the selection is asserted directly.
    """
    scanned = assert_fused_boundary_selected(prepared, K_MAX)
    # ODISSEO declares traced_boundary_weights=False, so nornax keeps the
    # unrolled boundary loop -- see test_boundary_walk_default_is_unrolled.
    assert scanned is False

    from nornax.solvers import fused_boundary_model

    assert fused_boundary_model(prepared, K_MAX) is prepared

    # A k_max disagreement is a misconfiguration, not a degradation: nornax
    # raises, and the ODISSEO guard turns that into an actionable error.
    with pytest.raises(RuntimeError, match="fused boundary path"):
        assert_fused_boundary_selected(prepared, K_MAX + 1)


def test_boundary_walk_default_is_unrolled_and_is_overridable(config, params):
    """The boundary walk is the *second* fusion switch, and nornax defaults it wrong.

    ``BlockStepFMM.boundary_kick`` accepts a ``level_weights`` argument, so
    nornax's signature probe opts into scanning the boundaries. That inlines the
    whole force into one program, which -- with the tree rebuilt every base step
    -- is recompiled every base step: measured 18.6 s against 10.4 s per base
    step at N = 512 on CPU. ODISSEO therefore declares
    ``traced_boundary_weights=False``. This pins both the default and that the
    override still works, since the scan is the right trade under an outer jit.
    """
    from nornax.solvers import supports_traced_level_weights

    default = build_blockstep_force(config, params, _options())
    assert supports_traced_level_weights(default) is False

    scanned = build_blockstep_force(
        config, params, _options(traced_boundary_weights=True)
    )
    assert supports_traced_level_weights(scanned) is True

    # None hands the decision back to nornax's signature probe, which says yes.
    probed = build_blockstep_force(
        config, params, _options(traced_boundary_weights=None)
    )
    assert supports_traced_level_weights(probed) is True


def test_scanned_and_unrolled_boundary_walks_agree(system, config, params):
    """Same trajectory either way: the boundary walk is a compile-cost choice.

    Both run ``n_sub + 1`` traversals and apply the same weights; the scan
    differs only in that the weight row is indexed by a traced boundary index.
    Powers of two are exact in binary floating point, so the two weightings are
    bit-identical and the trajectories must agree to round-off.
    """
    state, mass = system
    common = dict(n_base=2, track_energy=False)
    unrolled = integrate_blockstep_jaccpot(
        state,
        mass,
        config,
        params,
        options=_options(traced_boundary_weights=False),
        **common,
    )
    scanned = integrate_blockstep_jaccpot(
        state,
        mass,
        config,
        params,
        options=_options(traced_boundary_weights=True),
        **common,
    )
    assert unrolled.scanned_boundaries is False
    assert scanned.scanned_boundaries is True
    rel = float(
        jnp.linalg.norm(unrolled.state - scanned.state)
        / jnp.linalg.norm(scanned.state)
    )
    assert rel < 1.0e-12, f"boundary walks diverged at {rel:.3e}"


def test_the_jitted_force_wrapper_is_bit_for_bit_the_same_force(
    system, config, params, prepared
):
    """``jit_force`` is a compilation choice; it must not be a different force.

    The wrapper routes every entry point -- boundary kick, total acceleration,
    single level -- through one compiled ``sum_k w_k a_k`` kernel, deriving the
    static ``active_floor``/``half`` form into a weight vector so that one
    program serves them all. Each of those rewrites is a place a weight could be
    built wrong while every other test still passed, so all three are compared
    against the unwrapped model.
    """
    from odisseo.blockstep_coupling import JittedMutualForce

    state, mass = system
    positions = state[:, 0, :]
    velocities = state[:, 1, :]
    rung = jnp.asarray(
        np.random.default_rng(17).integers(0, K_MAX + 1, N_PARTICLES), dtype=jnp.int32
    )

    jitted = build_blockstep_force(config, params, _options(jit_force=True))
    assert isinstance(jitted, JittedMutualForce)
    jitted.prepare(positions, mass)

    # nornax must still see a fusable model through the wrapper.
    assert assert_fused_boundary_selected(jitted, K_MAX) is False

    def close(a, b, tol=1.0e-12):
        return float(jnp.linalg.norm(a - b) / max(float(jnp.linalg.norm(b)), 1e-300))

    assert close(
        jitted.total_accelerations(positions, mass),
        prepared.total_accelerations(positions, mass),
    ) < 1.0e-12

    for level in range(K_MAX + 1):
        assert close(
            jitted.level_accelerations(positions, mass, rung=rung, level=level),
            prepared.level_accelerations(positions, mass, rung=rung, level=level),
        ) < 1.0e-12

    kick_kwargs = dict(rung=rung, active_floor=1, dt_max=DT_MAX, half=0.5)
    assert close(
        jitted.boundary_kick(positions, velocities, mass, **kick_kwargs),
        prepared.boundary_kick(positions, velocities, mass, **kick_kwargs),
    ) < 1.0e-12

    # One program serves every call above -- that is the point of deriving the
    # static form into weights rather than passing it through as a static arg.
    assert jitted.num_compiles == 1


def test_a_per_level_only_model_is_rejected_by_the_guard(prepared):
    """A model that cannot fuse must fail loudly at the ODISSEO seam."""

    class _PerLevelOnly:
        """Satisfies MutualForceModel but not FusedMutualForceModel."""

        def __init__(self, inner):
            self._inner = inner

        def level_accelerations(self, positions, masses, *, rung, level, args=None):
            return self._inner.level_accelerations(
                positions, masses, rung=rung, level=level, args=args
            )

    with pytest.raises(RuntimeError, match="not selected"):
        assert_fused_boundary_selected(_PerLevelOnly(prepared), K_MAX)


# --------------------------------------------------------------------------
# the property: momentum
# --------------------------------------------------------------------------


def test_momentum_is_conserved_across_a_block_step_rollout(system, config, params):
    """``sum_i m_i v_i`` constant to round-off over a multi-rung rollout.

    This is the whole reason the lane exists, so the bar is 1e-13 relative --
    round-off for a float64 reduction of this width -- and not a loose bound.
    """
    state, mass = system
    result = integrate_blockstep_jaccpot(
        state,
        mass,
        config,
        params,
        options=_options(),
        n_base=6,
        track_energy=False,
    )
    assert result.num_far_pairs > 0
    assert result.fused
    # Rungs must actually be mixed, or the block scheme collapses to a shared
    # timestep and the antisymmetry across levels is never exercised.
    assert int(jnp.count_nonzero(result.rung_histogram[0] > 0)) > 1
    assert float(jnp.max(result.momentum_drift)) < 1.0e-13
    assert bool(jnp.all(jnp.isfinite(result.state)))


@pytest.mark.parametrize("theta", [0.5, 0.6, 0.8])
@pytest.mark.parametrize("order", [2, 4])
def test_the_momentum_residual_is_structural_not_a_tolerance(
    system, config, params, theta, order
):
    """Momentum conservation must not move when the force accuracy moves.

    This is the sharpest falsifiable form of the claim. Sweeping ``theta`` and
    the expansion order changes the force error by orders of magnitude; if the
    residual tracked it, the cancellation would be numerical luck rather than
    structural. It does not, because each pair is evaluated once and applied
    ``+f``/``-f``: the residual is set by the width of the reduction alone.

    This is also the test that would catch a kernel recomputing ``dr`` for the
    second endpoint instead of negating the first -- the one change that turns
    an exact cancellation into an approximate one while leaving every accuracy
    number untouched.
    """
    state, mass = system
    force = build_blockstep_force(
        config, params, _options(theta=theta, max_order=order)
    )
    force.prepare(state[:, 0, :], mass)
    assert _assert_far_field_is_exercised(force) > 0

    acc = force.total_accelerations(state[:, 0, :], mass)
    scale = float(jnp.sum(jnp.abs(mass))) * float(
        jnp.mean(jnp.linalg.norm(acc, axis=-1))
    )
    residual = float(jnp.linalg.norm(jnp.sum(mass[:, None] * acc, axis=0))) / scale
    assert residual < 1.0e-13, f"theta={theta}, p={order}: residual {residual:.3e}"


def test_the_shared_timestep_lane_is_not_more_momentum_conserving(
    system, config, params, prepared
):
    """The shared-timestep coupler must not beat the mutual force on momentum.

    jaccpot's documentation motivates the mutual restructure by attributing a
    ~1e-3 .. 1e-5 momentum residual to the target-centric force. **That did not
    reproduce.** Measured through ``evaluate_acceleration_jaccpot`` at N = 20000,
    theta = 0.7, order 4, float64, both lanes land at ~3e-17 -- with equal
    masses (2.98e-17 vs 2.55e-17) and with unequal ones (2.62e-17 vs 3.11e-17) --
    while the two forces differ by 2.8e-3, so the far field is genuinely active
    and this is not a degenerate direct-sum configuration. The same holds at
    N = 256 here.

    Asserting an order-of-magnitude gap would therefore be a test that fails for
    a reason unrelated to this code, so only the *direction* is asserted. The
    case for this lane does not rest on this number anyway: it rests on the
    per-level split, which the production coupler cannot express at all -- see
    :func:`test_each_level_conserves_momentum_on_its_own`.
    """
    from odisseo.jaccpot_coupling import evaluate_acceleration_jaccpot

    state, mass = system

    def residual(acc):
        scale = float(jnp.sum(jnp.abs(mass))) * float(
            jnp.mean(jnp.linalg.norm(acc, axis=-1))
        )
        return float(jnp.linalg.norm(jnp.sum(mass[:, None] * acc, axis=0))) / scale

    r_mutual = residual(prepared.total_accelerations(state[:, 0, :], mass))
    r_target = residual(
        evaluate_acceleration_jaccpot(
            state,
            mass,
            config,
            params,
            leaf_size=LEAF_SIZE,
            max_order=MAX_ORDER,
            fmm_basis="real",
            fmm_theta=THETA,
            fmm_tree_leaf_target=LEAF_SIZE,
        )
    )
    assert r_mutual < 1.0e-13, f"mutual residual {r_mutual:.3e} is not round-off"
    assert r_target >= 0.1 * r_mutual, (
        f"the target-centric lane ({r_target:.3e}) came out an order of magnitude "
        f"*better* than the mutual one ({r_mutual:.3e}); that inverts the premise "
        "of this module and means one lane is not the force it claims to be"
    )


# --------------------------------------------------------------------------
# the property: energy
# --------------------------------------------------------------------------


def test_energy_drift_is_bounded_and_oscillates(system, config, params):
    """Leapfrog is symplectic: the energy error oscillates, it does not accumulate.

    The bar is a bound *and* a shape. A bound alone is satisfied by a slow
    one-way accumulation over a short run, which is exactly the failure a
    non-symplectic integrator shows; requiring sign changes rejects that while
    accepting the oscillation a symplectic map actually produces.

    An earlier version of this test compared the mean |drift| of the rollout's
    second half against its first, and that was the wrong instrument: at
    ``dt_max = 4e-2`` this IC has one close encounter around base step 9 whose
    unresolved kick moves the energy by 1.3e-3 in a single step and then holds
    it there. That is a step function, not secular growth, and no half-vs-half
    ratio distinguishes them. It is resolved by a deeper rung ladder, which is
    what :func:`test_a_deeper_rung_ladder_reduces_the_energy_error` measures.
    """
    state, mass = system
    result = integrate_blockstep_jaccpot(
        state,
        mass,
        config,
        params,
        options=_options(),
        n_base=12,
        track_energy=True,
        energy_chunk=128,
    )
    drift = np.asarray(result.energy_drift)
    assert np.all(np.isfinite(drift))
    # Measured 1.3e-5 over 16 base steps at this configuration; 1e-4 leaves room
    # for platform-dependent summation without accepting a runaway.
    assert np.max(np.abs(drift)) < 1.0e-4, f"energy drift unbounded: {drift}"

    signs = np.sign(drift[1:])
    changes = int(np.count_nonzero(np.diff(signs[signs != 0]) != 0))
    assert changes >= 1, (
        "the energy error never changed sign, i.e. it accumulated one way rather "
        f"than oscillating, which a symplectic map should not do: {drift}"
    )


def test_a_deeper_rung_ladder_reduces_the_energy_error(system, config, params):
    """The individual timesteps must actually buy accuracy, not just run.

    Every other test here would pass with ``k_max = 0`` -- a block scheme that
    collapsed to a shared timestep still conserves momentum, still fuses, still
    matches the oracle. This is the one that fails if the rung machinery is
    inert: at a fixed ``dt_max`` where this IC has an under-resolved close
    encounter, adding one rung (so the finest particles step at ``dt_max / 8``
    instead of ``dt_max / 4``) must measurably reduce the energy error.

    Measured on this IC at ``dt_max = 4e-2`` over 16 base steps: max |dE/E| goes
    from 1.26e-3 at ``k_max = 2`` to 7.12e-5 at ``k_max = 3``, an 18x
    improvement, with only 27 of 256 particles on the new rung. The bar below is
    5x, which is well clear of run-to-run variation but far below the measured
    effect.
    """
    state, mass = system
    drifts = {}
    for k_max in (2, 3):
        result = integrate_blockstep_jaccpot(
            state,
            mass,
            config,
            params,
            options=_options(dt_max=4.0e-2, k_max=k_max),
            n_base=12,
            track_energy=True,
            energy_chunk=128,
        )
        # The ladder is only meaningful if the extra rung is actually populated.
        assert int(result.rung_histogram[0][-1]) > 0, (
            f"k_max={k_max}: the finest rung is empty, so this configuration "
            "does not test the ladder at all"
        )
        drifts[k_max] = float(np.max(np.abs(np.asarray(result.energy_drift))))

    assert drifts[3] * 5.0 < drifts[2], (
        f"a deeper rung ladder did not help: max |dE/E| was {drifts[2]:.3e} at "
        f"k_max=2 and {drifts[3]:.3e} at k_max=3. Either the rung assignment is "
        "not reaching the finest level or the sub-step kicks are not being applied"
    )


# --------------------------------------------------------------------------
# agreement with the other force paths
# --------------------------------------------------------------------------


def test_total_force_matches_the_shared_timestep_coupler(
    system, config, params, prepared
):
    """The two lanes are the same physics: totals agree to FMM tolerance.

    Only the *total* is compared. Per level they cannot agree: the mutual far
    field assigns each cell the rung of its finest particle and splits at cell
    granularity, while any per-pair split (a direct sum's, or the target-centric
    lane's) is finer. Both are genuine partitions of the interaction set, so the
    totals must match even though the decompositions do not.
    """
    from odisseo.jaccpot_coupling import evaluate_acceleration_jaccpot

    state, mass = system
    assert _assert_far_field_is_exercised(prepared) > 0

    a_block = prepared.total_accelerations(state[:, 0, :], mass)
    a_shared = evaluate_acceleration_jaccpot(
        state,
        mass,
        config,
        params,
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
        fmm_basis="real",
        fmm_theta=THETA,
        fmm_tree_leaf_target=LEAF_SIZE,
    )
    rel = float(jnp.linalg.norm(a_block - a_shared) / jnp.linalg.norm(a_shared))
    assert rel < 1.0e-3, f"total accelerations disagree at {rel:.3e}"


def test_total_force_matches_the_mutual_direct_sum_oracle(system, prepared):
    """The cheap oracle: nornax's momentum-exact dense direct sum.

    The FMM's own force error against an exact sum is the tolerance here, so
    this is the test that would catch a broken expansion -- the shared-timestep
    comparison above would not, since both lanes share jaccpot's far field.
    """
    from nornax.forces.mutual_direct import MutualDirectSumGravity

    state, mass = system
    positions = state[:, 0, :]
    oracle = MutualDirectSumGravity(G=1.0, softening=SOFTENING)
    a_exact = oracle.total_accelerations(positions, mass)
    a_fmm = prepared.total_accelerations(positions, mass)
    rel = float(jnp.linalg.norm(a_fmm - a_exact) / jnp.linalg.norm(a_exact))
    assert rel < 1.0e-3, f"FMM vs direct sum: {rel:.3e}"


def test_levels_partition_the_total_acceleration(system, prepared):
    """Summing the levels reproduces the fused total, to round-off.

    ``total_accelerations`` is one traversal and the per-level sum is
    ``k_max + 1`` of them; they are the same quantity, and this is what makes
    the fused path a pure optimisation rather than a different force.
    """
    state, mass = system
    positions = state[:, 0, :]
    rung = jnp.asarray(
        np.random.default_rng(7).integers(0, K_MAX + 1, N_PARTICLES), dtype=jnp.int32
    )
    per_level = sum(
        prepared.level_accelerations(positions, mass, rung=rung, level=k)
        for k in range(K_MAX + 1)
    )
    total = prepared.total_accelerations(positions, mass)
    rel = float(jnp.linalg.norm(per_level - total) / jnp.linalg.norm(total))
    assert rel < 1.0e-12, f"levels do not partition the total: {rel:.3e}"


def test_each_level_conserves_momentum_on_its_own(system, prepared):
    """Per-level antisymmetry is the scheme's defining property, not a summary one.

    A force that only conserved momentum *in total* would still be illegal here:
    the block step applies one level at a time.
    """
    state, mass = system
    positions = state[:, 0, :]
    rung = jnp.asarray(
        np.random.default_rng(11).integers(0, K_MAX + 1, N_PARTICLES), dtype=jnp.int32
    )
    total_scale = float(
        jnp.mean(jnp.linalg.norm(prepared.total_accelerations(positions, mass), -1))
    ) * float(jnp.sum(jnp.abs(mass)))
    for k in range(K_MAX + 1):
        a_k = prepared.level_accelerations(positions, mass, rung=rung, level=k)
        residual = float(jnp.linalg.norm(jnp.sum(mass[:, None] * a_k, axis=0)))
        assert residual / total_scale < 1.0e-13, f"level {k}: {residual:.3e}"


# --------------------------------------------------------------------------
# configuration agreement
# --------------------------------------------------------------------------


def test_rung_range_agrees_between_nornax_and_the_model(system, prepared):
    """``BlockStepFMM`` rejects out-of-range rungs; ``assign_rungs`` clips into range.

    They agree only because ODISSEO hands both the same ``k_max``. This pins the
    round trip: nornax's assignment from the FMM's own acceleration always lands
    inside the model's accepted range.
    """
    from nornax.blockstep.rungs import assign_rungs

    state, mass = system
    acc = prepared.total_accelerations(state[:, 0, :], mass)
    rung = assign_rungs(acc, dt_max=DT_MAX, k_max=K_MAX, eta=0.1, eps=SOFTENING)
    assert int(jnp.min(rung)) >= 0
    assert int(jnp.max(rung)) <= K_MAX
    # The FMM accepts it without complaint.
    prepared.level_accelerations(state[:, 0, :], mass, rung=rung, level=0)


def test_an_out_of_range_rung_is_rejected_not_clamped(system, config, params):
    """A rung above ``k_max`` has no kick weight, so it must not be silently used."""
    state, mass = system
    bad = jnp.full((N_PARTICLES,), K_MAX + 1, dtype=jnp.int32)
    with pytest.raises(ValueError, match=r"\[0, k_max"):
        _check_rung_range(bad, K_MAX)
    force = build_blockstep_force(config, params, _options())
    with pytest.raises(ValueError):
        blockstep_initial_state(state, mass, force, _options(), rung=bad)


def test_rung_validation_survives_a_concrete_array_closed_over_by_a_trace():
    """``isinstance(x, Tracer)`` is not the question "can I read this value".

    A *concrete* array closed over by a ``lax.cond`` branch is not a Tracer, yet
    reducing it inside the trace still yields one, so ``int(...)`` raises. The
    validation must attempt the read and catch ``JAXTypeError`` -- gating on
    ``isinstance`` lets exactly this case through into a
    ``ConcretizationTypeError`` deeper in the stack.
    """
    rung = jnp.zeros((8,), dtype=jnp.int32)  # concrete, not a tracer
    assert not isinstance(rung, jax.core.Tracer)

    def branch(v):
        _check_rung_range(rung, K_MAX)  # closed-over concrete array
        return v + 1.0

    out = jax.jit(lambda flag, v: jax.lax.cond(flag, branch, lambda x: x, v))(
        True, jnp.asarray(1.0)
    )
    assert float(out) == 2.0


def test_an_external_potential_config_is_rejected(system, params):
    """Self-gravity only: an external force has no partner to receive -f."""
    state, mass = system
    cfg = SimulationConfig(
        N_particles=N_PARTICLES,
        softening=SOFTENING,
        external_accelerations=(NFW_POTENTIAL,),
    )
    with pytest.raises(ValueError, match="self-gravity only"):
        build_blockstep_force(cfg, params, _options())
    with pytest.raises(ValueError, match="self-gravity only"):
        integrate_blockstep_jaccpot(
            state, mass, cfg, params, options=_options(), n_base=1
        )


def test_options_reject_an_impossible_rebuild_cadence():
    """The tree is built on the host, so it cannot refresh faster than a base step."""
    with pytest.raises(ValueError, match="rebuild_every"):
        _options(rebuild_every=0)


# --------------------------------------------------------------------------
# the scanned and eager drivers agree
# --------------------------------------------------------------------------


def test_scanned_and_eager_base_step_drivers_agree(system, config, params):
    """``block_kdk_rollout`` (scanned) and ``block_kdk_base_step`` (eager) agree.

    Same tree, same schedule, same arithmetic -- so the two differ only in
    whether nornax's base steps are inlined into one program. The trajectories
    must match to round-off; the two paths differ in *peak memory*, not results.
    """
    state, mass = system
    common = dict(n_base=4, track_energy=False)
    eager = integrate_blockstep_jaccpot(
        state,
        mass,
        config,
        params,
        options=_options(rebuild_every=4, scan_base_steps=False),
        **common,
    )
    scanned = integrate_blockstep_jaccpot(
        state,
        mass,
        config,
        params,
        options=_options(rebuild_every=4, scan_base_steps=True),
        **common,
    )
    rel = float(
        jnp.linalg.norm(eager.state - scanned.state) / jnp.linalg.norm(scanned.state)
    )
    assert rel < 1.0e-12, f"scanned and eager drivers diverged at {rel:.3e}"


def test_the_tree_is_rebuilt_once_per_rebuild_interval(system, config, params):
    """Topology lifetime: one host tree build per interval, not per sub-step.

    Getting this wrong is the difference between an O(N) run and one that
    rebuilds a tree per sub-step, and it is invisible in the results.
    """
    state, mass = system
    force = build_blockstep_force(config, params, _options())
    calls = {"n": 0}
    real_prepare = force.prepare

    def counting_prepare(positions, masses):
        calls["n"] += 1
        return real_prepare(positions, masses)

    force.prepare = counting_prepare  # type: ignore[method-assign]
    integrate_blockstep_jaccpot(
        state,
        mass,
        config,
        params,
        options=_options(),
        n_base=5,
        force=force,
        track_energy=False,
    )
    # One build to seed the state, then one per subsequent base step.
    assert calls["n"] == 5, f"expected 5 tree builds for 5 base steps, got {calls['n']}"


# --------------------------------------------------------------------------
# differentiability
# --------------------------------------------------------------------------


def test_rollout_gradient_matches_finite_differences(system, config, params):
    """``d(summary)/d(IC)`` through a fixed-topology, frozen-rung rollout.

    Both sides must see the *same* frozen plan: FD over a run that rebuilds the
    tree or reassigns rungs disagrees whenever a pair crosses a MAC boundary or
    a particle changes rung, and the disagreement is not a gradient error. So
    the topology is prepared once, outside the differentiated function, and
    ``reassign_rungs=False`` holds the schedule fixed -- the same treatment
    nornax gives its own schedule with ``stop_gradient``.
    """
    from nornax.solvers import advance_base_step
    from nornax.state import BlockStepState

    state, mass = system
    positions0, velocities0 = state[:, 0, :], state[:, 1, :]

    force = build_blockstep_force(config, params, _options())
    force.prepare(positions0, mass)
    assert _assert_far_field_is_exercised(force) > 0
    assert_fused_boundary_selected(force, K_MAX)

    rung = jnp.asarray(
        np.random.default_rng(3).integers(0, K_MAX + 1, N_PARTICLES), dtype=jnp.int32
    )

    def summary(velocities):
        block = BlockStepState(
            positions=positions0,
            velocities=velocities,
            masses=mass,
            acc=jnp.zeros_like(positions0),
            rung=rung,
            base_index=jnp.asarray(0, dtype=jnp.int32),
        )
        for _ in range(2):
            block = advance_base_step(block, DT_MAX, force, k_max=K_MAX)
        return jnp.sum(block.positions**2)

    grad = jax.grad(summary)(velocities0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0

    # Finite differences along one random direction: a full Jacobian would be
    # 768 evaluations of a rollout.
    direction = jnp.asarray(
        np.random.default_rng(13).normal(size=velocities0.shape), dtype=jnp.float64
    )
    direction = direction / jnp.linalg.norm(direction)
    h = 1.0e-6
    fd = float(
        (summary(velocities0 + h * direction) - summary(velocities0 - h * direction))
        / (2.0 * h)
    )
    ad = float(jnp.sum(grad * direction))
    assert abs(ad - fd) <= 1.0e-6 * max(abs(fd), 1.0), f"AD {ad:.8e} vs FD {fd:.8e}"


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def test_chunked_potential_energy_matches_the_dense_reference(system):
    """The chunked energy diagnostic is the dense one, without the (N, N) matrix."""
    from nornax.diagnostics import gravitational_potential_energy

    state, mass = system
    dense = gravitational_potential_energy(
        state[:, 0, :], mass, G=1.0, softening=SOFTENING
    )
    chunked = chunked_potential_energy(
        state[:, 0, :], mass, G=1.0, softening=SOFTENING, chunk=37
    )
    assert abs(float(chunked - dense)) <= 1.0e-12 * abs(float(dense))


def test_momentum_diagnostic_matches_nornax(system):
    from nornax.diagnostics import total_linear_momentum as nornax_momentum

    state, mass = system
    ours = total_linear_momentum(mass, state[:, 1, :])
    theirs = nornax_momentum(mass, state[:, 1, :])
    assert float(jnp.max(jnp.abs(ours - theirs))) == 0.0


def test_blockstep_total_acceleration_builds_its_own_force(system, config, params):
    """The convenience entry point needs no prebuilt model."""
    state, mass = system
    acc = blockstep_total_acceleration(state, mass, config, params, _options())
    assert acc.shape == (N_PARTICLES, 3)
    assert bool(jnp.all(jnp.isfinite(acc)))


# --------------------------------------------------------------------------
# known upstream limitation: the Pallas near field drops the dt_max gradient
# --------------------------------------------------------------------------


def _pallas_dt_max_gradient(backend, interpret, system, config, params):
    """Return (AD, FD) for ``d/d(dt_max)`` of a boundary kick on one backend."""
    state, mass = system
    positions, velocities = state[:, 0, :], state[:, 1, :]
    rung = jnp.asarray(
        np.random.default_rng(23).integers(0, K_MAX + 1, N_PARTICLES), dtype=jnp.int32
    )
    force = build_blockstep_force(
        config, params, _options(backend=backend, pallas_interpret=interpret)
    )
    force.prepare(positions, mass)
    assert _assert_far_field_is_exercised(force) > 0
    assert int(getattr(force.state, "num_near_pairs", 0)) > 0

    def loss(dt_max):
        kicked = force.boundary_kick(
            positions, velocities, mass, rung=rung, active_floor=0, dt_max=dt_max
        )
        return jnp.sum(kicked**2)

    dt0, h = DT_MAX, 1.0e-7
    return float(jax.grad(loss)(dt0)), float((loss(dt0 + h) - loss(dt0 - h)) / (2.0 * h))


def test_the_dt_max_gradient_is_exact_on_the_pure_jax_backend(system, config, params):
    """``d/d(dt_max)`` must be exact: nornax relies on it being differentiable.

    nornax deliberately keeps ``dt_max`` traced by scaling it *into* the boundary
    weight table rather than baking it in, so a loss can be differentiated with
    respect to the timestep. This pins that the default backend honours it.
    """
    ad, fd = _pallas_dt_max_gradient("jax", False, system, config, params)
    assert abs(ad - fd) <= 1.0e-6 * abs(fd), f"AD {ad:.6e} vs FD {fd:.6e}"


def test_the_pallas_backend_drops_most_of_the_dt_max_gradient(system, config, params):
    """KNOWN UPSTREAM DEFECT, pinned so it cannot be relied on or silently change.

    ``jaccpot/pallas/nearfield_mutual.py`` returns ``jnp.zeros_like(level_weights)``
    from its reverse rule, on the stated grounds that the level table is "discrete
    or frozen". It is neither: ``level_weights[k] == half * dt_max / 2**k`` is a
    smooth function of ``dt_max``, and the forward force is *linear* in it. So the
    near field's entire contribution to ``d/d(dt_max)`` is dropped and only the
    far field's survives -- measured 111x too small at this configuration (ratio
    0.0090), not merely a missing higher-order term.

    The same reverse rule zeroes ``softening_sq`` and ``g_value``, so
    ``d/d(softening)`` and ``d/d(G)`` are lost through the near field too.

    This asserts the *discrepancy*, not a tolerance, so the test fails the moment
    upstream fixes it -- at which point it should be replaced by the exactness
    assertion above, parametrized over both backends. Fixing it properly means
    reducing ``f_geometric . Fbar`` per level inside the reverse kernel, which
    has the tile in registers and could emit a ``(k_max + 1,)`` cotangent.
    """
    ad, fd = _pallas_dt_max_gradient("pallas", True, system, config, params)
    assert abs(fd) > 1.0e-6, "the finite-difference reference is degenerate here"
    ratio = ad / fd
    assert ratio < 0.5, (
        f"the Pallas dt_max gradient is no longer badly wrong (ratio {ratio:.4f}). "
        "If upstream fixed the level_weights cotangent, delete this test and "
        "parametrize test_the_dt_max_gradient_is_exact_on_the_pure_jax_backend "
        "over both backends instead."
    )


def test_one_compiled_program_survives_every_topology_rebuild(system, config, params):
    """The Phase-1 payoff, asserted end to end from ODISSEO.

    The topology reaches the compiled force as a *traced pytree argument*, so the
    program is keyed on its shapes; ``static_shapes`` pads the pair lists and the
    level schedule so those shapes hold still. With both, the ~200 s per-rebuild
    compile that made ``jit_force`` a trade rather than a win simply stops
    happening.

    The nudge is deliberately **non-rigid**. Adding a scalar to every coordinate
    is a rigid translation, which leaves the Morton order and every MAC outcome
    untouched -- a check written that way reports perfect stability for a topology
    that is in fact drifting, and that is exactly how a false claim of shape
    stability got into this repo's docs once already.
    """
    from odisseo.blockstep_coupling import JittedMutualForce

    state, mass = system
    positions = state[:, 0, :]
    rng = np.random.default_rng(31)

    force = build_blockstep_force(config, params, _options(jit_force=True))
    assert isinstance(force, JittedMutualForce)

    for scale in (0.0, 1.0e-3, 1.0e-1):
        nudged = positions + scale * jnp.asarray(
            rng.normal(size=positions.shape), dtype=positions.dtype
        )
        force.prepare(nudged, mass)
        assert _assert_far_field_is_exercised(force) > 0
        jit_acc = force.total_accelerations(nudged, mass)
        # Padding must not leak into the answer: compare against an unpadded,
        # un-jitted model built on the same positions.
        plain = build_blockstep_force(config, params, _options())
        plain.prepare(nudged, mass)
        ref = plain.total_accelerations(nudged, mass)
        rel = float(jnp.linalg.norm(jit_acc - ref) / jnp.linalg.norm(ref))
        assert rel < 1.0e-13, f"padded/jitted vs plain disagree at {rel:.3e}"

    assert force.num_compiles == 1, (
        f"the force compiled {force.num_compiles} times across three rebuilds; "
        "either the topology is not arriving as a traced argument or the "
        "capacity padding is not holding the shapes"
    )


# --------------------------------------------------------------------------
# the fully jitted lane
# --------------------------------------------------------------------------


def _device_backend_available() -> bool:
    try:
        import inspect

        from jaccpot import BlockStepFMM as _Model
    except Exception:
        return False
    return "topology_backend" in inspect.signature(_Model).parameters


device_lane = pytest.mark.skipif(
    not _device_backend_available(),
    reason="installed jaccpot has no BlockStepFMM(topology_backend=...)",
)


@device_lane
def test_the_jitted_lane_matches_the_host_loop_lane(system, config, params):
    """The two lanes are the same scheme, driven differently.

    ``integrate_blockstep_jitted`` walks the boundaries itself rather than going
    through nornax's ``advance_base_step``, because a per-base-step topology has
    to live in the scan *carry* and nornax has no hook for that. So the thing to
    pin is that it is still the same schedule: the **rung histograms must match
    exactly**, since those come from the same ``assign_rungs`` on comparable
    accelerations, while the trajectories may differ at FMM tolerance because the
    two lanes build different trees (LBVH against static-radix).
    """
    state, mass = system
    common = dict(
        dt_max=DT_MAX, k_max=K_MAX, theta=THETA, max_order=MAX_ORDER,
        leaf_size=LEAF_SIZE,
    )
    host = integrate_blockstep_jaccpot(
        state, mass, config, params,
        options=BlockStepOptions(**common), n_base=3, track_energy=False,
    )
    jitted = integrate_blockstep_jitted(
        state, mass, config, params,
        options=BlockStepOptions(**common, topology_backend="device"),
        n_base=3, track_energy=False,
    )
    assert jitted.num_far_pairs > 0
    assert np.array_equal(
        np.asarray(host.rung_histogram[-1]), np.asarray(jitted.rung_histogram[-1])
    ), (
        f"schedules diverged: host {np.asarray(host.rung_histogram[-1]).tolist()} "
        f"vs jitted {np.asarray(jitted.rung_histogram[-1]).tolist()}"
    )
    rel = float(
        jnp.linalg.norm(jitted.state - host.state) / jnp.linalg.norm(host.state)
    )
    assert rel < 1.0e-3, f"lanes disagree beyond FMM tolerance at {rel:.3e}"


@device_lane
def test_the_jitted_lane_conserves_momentum_with_the_tree_rebuilt_in_scan(
    system, config, params
):
    """The property has to survive the topology moving underneath the integrator.

    Each base step gets a freshly-built tree, so the interaction *partition*
    changes from step to step. Momentum conservation is indifferent to that --
    it comes from evaluating each pair once and applying both signs -- and this
    asserts it does in fact stay at round-off across the rebuilds.
    """
    state, mass = system
    result = integrate_blockstep_jitted(
        state, mass, config, params,
        options=BlockStepOptions(
            dt_max=DT_MAX, k_max=K_MAX, theta=THETA, max_order=MAX_ORDER,
            leaf_size=LEAF_SIZE, topology_backend="device",
        ),
        n_base=4, track_energy=True, energy_chunk=128,
    )
    assert float(jnp.max(result.momentum_drift)) < 1.0e-13
    assert bool(jnp.all(jnp.isfinite(result.state)))
    assert result.energy_drift is not None
    assert abs(float(result.energy_drift[-1])) < 1.0e-4
    # The rungs must actually be mixed, or the block scheme collapsed.
    assert int(jnp.count_nonzero(result.rung_histogram[-1] > 0)) > 1


@device_lane
def test_the_jitted_lane_refuses_the_host_topology_backend(system, config, params):
    """A host traversal cannot be traced, so asking for it here is a hard error."""
    state, mass = system
    with pytest.raises(ValueError, match="topology_backend='device'"):
        integrate_blockstep_jitted(
            state, mass, config, params,
            # Asked for explicitly. This used to rely on `"host"` being the
            # default; it is `"auto"` now, which resolves to `"device"` here and
            # would make this raise nothing at all.
            options=BlockStepOptions(
                dt_max=DT_MAX, k_max=K_MAX, theta=THETA, max_order=MAX_ORDER,
                leaf_size=LEAF_SIZE, topology_backend="host",
            ),
            n_base=1,
        )


@device_lane
def test_the_device_topology_backend_actually_reaches_the_model(config, params):
    """``topology_backend`` must not be silently dropped on the way to jaccpot.

    It was, for a while: the kwarg was threaded into ``build_blockstep_force``
    with a ``**`` helper whose call site had moved, so the replace that added it
    matched nothing and ``BlockStepOptions(topology_backend="device")`` quietly
    built a *host* model. Nothing failed -- the host path is correct, just 3750x
    slower on the tree build -- and it was only visible because
    ``force.capacities`` came back ``None``.

    ``integrate_blockstep_jitted`` was unaffected (it calls ``freeze_template``
    directly), which is exactly why this needs its own test: the fast lane hid
    the bug in the slow one.
    """
    options = BlockStepOptions(
        dt_max=DT_MAX, k_max=K_MAX, theta=THETA, max_order=MAX_ORDER,
        leaf_size=LEAF_SIZE, topology_backend="device",
    )
    force = build_blockstep_force(config, params, options)
    assert getattr(force, "topology_backend", "host") == "device", (
        "options.topology_backend did not reach BlockStepFMM"
    )
    # The device backend has no unpadded form, so it must imply static shapes.
    assert getattr(force, "static_shapes", False) is True

    host = build_blockstep_force(
        config, params,
        BlockStepOptions(
            dt_max=DT_MAX, k_max=K_MAX, theta=THETA, max_order=MAX_ORDER,
            leaf_size=LEAF_SIZE, topology_backend="host",
        ),
    )
    assert getattr(host, "topology_backend", "host") == "host"


# --------------------------------------------------------------------------
# one front door: the default resolves to the fast lane, and the host entry
# point dispatches to it
# --------------------------------------------------------------------------


def test_auto_resolves_to_device_when_jaccpot_can_and_host_when_it_cannot(
    monkeypatch,
):
    """``"auto"`` is the whole reason the default can be fast AND safe."""
    import odisseo.blockstep_coupling as bc

    options = _options(topology_backend="auto")

    monkeypatch.setattr(bc, "device_topology_available", lambda: True)
    assert bc.resolve_topology_backend(options) == "device"

    monkeypatch.setattr(bc, "device_topology_available", lambda: False)
    assert bc.resolve_topology_backend(options) == "host"


def test_an_explicit_device_request_raises_rather_than_degrading(monkeypatch):
    """The point of three values: `"auto"` may fall back, `"device"` may not.

    A default that silently degraded would present as a slow machine rather than
    as a missing dependency -- the host lane is ~17x slower per base step at
    N = 20 000, which is easy to misread as anything else.
    """
    import odisseo.blockstep_coupling as bc

    monkeypatch.setattr(bc, "device_topology_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs a jaccpot"):
        bc.resolve_topology_backend(_options(topology_backend="device"))
    # ...and the fallback is still available by name.
    assert bc.resolve_topology_backend(_options(topology_backend="auto")) == "host"


def test_resolve_lane_names_every_lane_and_rejects_the_unknown(config):
    """The selector's decision, pinned as a value rather than inferred from output.

    Every lane returns a plain state, so before this existed the only way to
    check dispatch was to compare float arrays against a reference call -- which
    a same-input control shows cannot distinguish two lanes on GPU at any
    tolerance. A lane NAME is exact on every platform.
    """
    from odisseo.integration_api import INTEGRATION_LANES, resolve_lane
    from odisseo.option_classes import DIRECT_ACC, FMM_ACC

    base = config

    assert resolve_lane(base._replace(acceleration_scheme=DIRECT_ACC)) == "direct"

    fmm = base._replace(acceleration_scheme=FMM_ACC)
    assert resolve_lane(fmm) == "fmm_forward"
    assert resolve_lane(fmm._replace(fmm_differentiable=True)) == "fmm_differentiable"
    assert resolve_lane(fmm._replace(fmm_blockstep=True)) == "fmm_blockstep"
    # Precedence is documented on integrate(); pin it rather than trust the order.
    assert (
        resolve_lane(fmm._replace(fmm_blockstep=True, fmm_differentiable=True))
        == "fmm_blockstep"
    )

    # Every name it can return is declared, so a new lane cannot be added
    # without appearing in the published tuple.
    for cfg in (
        base._replace(acceleration_scheme=DIRECT_ACC),
        fmm,
        fmm._replace(fmm_differentiable=True),
        fmm._replace(fmm_blockstep=True),
    ):
        assert resolve_lane(cfg) in INTEGRATION_LANES

    with pytest.raises(ValueError, match="unknown acceleration_scheme"):
        resolve_lane(base._replace(acceleration_scheme=9999))


def test_the_availability_probe_checks_yggdrax_and_not_only_jaccpot(monkeypatch):
    """The device lane's two halves live in two repositories, so probe both.

    Every other test here monkeypatches ``device_topology_available`` itself,
    which is exactly how this went unnoticed: the resolver was covered and the
    probe never was. It inspected only ``BlockStepFMM``'s signature, so against
    a yggdrax predating ``dual_tree_walk_mutual`` it still answered ``True``,
    ``"auto"`` resolved to ``"device"``, and the run died with an ``ImportError``
    raised by the function-local import in ``jaccpot.mutual.device_topology`` --
    the late failure ``"auto"`` exists to prevent.
    """
    import yggdrax.interactions as yi

    import odisseo.blockstep_coupling as bc

    assert bc.device_topology_available() is True, (
        "guard: this environment must HAVE the device lane, or the test below "
        "passes vacuously"
    )

    monkeypatch.delattr(yi, "dual_tree_walk_mutual")
    assert bc.device_topology_available() is False
    assert bc.resolve_topology_backend(_options(topology_backend="auto")) == "host"
    with pytest.raises(RuntimeError, match="yggdrax"):
        bc.resolve_topology_backend(_options(topology_backend="device"))


def test_an_unknown_topology_backend_is_rejected():
    import odisseo.blockstep_coupling as bc

    with pytest.raises(ValueError, match="must be 'auto', 'host' or 'device'"):
        bc.resolve_topology_backend(_options(topology_backend="gpu"))


@device_lane
def test_the_host_entry_point_dispatches_to_the_jitted_lane(system, config, params):
    """``integrate_blockstep_jaccpot`` is the front door for BOTH lanes.

    Routing rather than asking the caller to pick a function: with the device
    backend it must produce what calling the jitted lane directly does, not
    merely something similar. "What it produces" is carried by exact equality on
    the structural diagnostics; the state gets only a coarse bound, because a
    same-input control shows a bitwise state comparison cannot distinguish the
    two lanes on GPU at any tolerance. See the measured note at the assertion.
    """
    state, mass = system
    common = dict(
        dt_max=DT_MAX, k_max=K_MAX, theta=THETA, max_order=MAX_ORDER,
        leaf_size=LEAF_SIZE, topology_backend="device",
    )
    through_front_door = integrate_blockstep_jaccpot(
        state, mass, config, params,
        options=BlockStepOptions(**common), n_base=2, track_energy=False,
    )
    direct = integrate_blockstep_jitted(
        state, mass, config, params,
        options=BlockStepOptions(**common), n_base=2, track_energy=False,
    )
    # The "same lane" claim is carried EXACTLY by the structural diagnostics --
    # a front door that dropped an option (topology_backend has regressed that
    # way once) builds a different topology and moves these, and they are
    # integers, so no floating-point caveat applies to them.
    assert through_front_door.num_far_pairs == direct.num_far_pairs
    assert through_front_door.num_near_pairs == direct.num_near_pairs
    assert through_front_door.fused == direct.fused
    assert through_front_door.scanned_boundaries == direct.scanned_boundaries
    assert jnp.array_equal(
        through_front_door.rung_histogram, direct.rung_histogram
    )
    # The state comparison below is a COARSE not-a-different-lane check. It is
    # deliberately NOT the evidence for dispatch, and it must not be tightened
    # into looking like it is: it was rtol=0 and was red on an A100 while green
    # on CPU, and a same-input control says no threshold can rescue that. Run on
    # one A100, 2 base steps, comparing this test's own quantities:
    #
    #   A-vs-A (front door called TWICE)   abs 8.882e-16  rel 2.599e-15  28/1536
    #   A-vs-B (front door vs direct)      abs 8.882e-16  rel 2.599e-15  26/1536
    #
    # Read the ABSOLUTE column; the relative one is only trustworthy here by
    # luck. 8.882e-16 is exactly 2 ULP at the largest state element, one rung of
    # an exact 2/4/6/8 ULP ladder these deviations land on. A per-element
    # relative statistic is the wrong tool on a state spanning four orders of
    # magnitude -- it reports which near-zero element won the argmax, not any
    # deviation (jaccpot's equivalent test has its max-rel at |x| = 9.5e-05 with
    # a one-ULP deviation, giving a 79x spread that is pure denominator). Ours
    # happens to sit at |x| = 0.342, an ordinary magnitude, so it is not
    # contaminated -- but do not reuse the relative figure without checking that.
    #
    # Identical to four significant figures. Calling the same entry point twice
    # differs as much as the two entry points differ, so the state comparison
    # carries no information about which lane ran, at any tolerance. (An earlier
    # single measurement of A-vs-B gave 2.8e-17 -- 30x smaller. The quantity is
    # not stable run to run, which is the same finding again: one draw is not a
    # bound. Do not re-derive a tolerance from a single run.)
    #
    # What DOES carry the claim is the integer block above. The bound here is
    # ~1e7 below a re-implementation, which differs at O(1e-2) or worse, so it
    # still catches a genuinely different lane -- and nothing finer.
    assert jnp.allclose(
        through_front_door.state, direct.state, rtol=1e-9, atol=1e-12
    ), "the front door must dispatch to the jitted lane, not re-implement it"


@device_lane
def test_host_only_kwargs_are_rejected_on_the_device_lane(system, config, params):
    """Dropping them silently would kill a caller's diagnostics without a word.

    `progress` and `record_every` exist because the host lane is a Python loop
    with a host round trip per base step -- exactly what the jitted lane removes.
    There is no per-step re-entry to call back into, so they cannot be honoured.
    """
    state, mass = system
    options = BlockStepOptions(
        dt_max=DT_MAX, k_max=K_MAX, theta=THETA, max_order=MAX_ORDER,
        leaf_size=LEAF_SIZE, topology_backend="device",
    )
    for kwargs in ({"record_every": 2}, {"progress": lambda i, rec: None}):
        with pytest.raises(ValueError, match="only applies to the host lane"):
            integrate_blockstep_jaccpot(
                state, mass, config, params, options=options, n_base=1, **kwargs
            )


# --------------------------------------------------------------------------
# the unified integrate() API reaches this lane
# --------------------------------------------------------------------------


def test_integrate_reaches_the_blockstep_lane(system, config, params):
    """``integrate()`` must route FMM_ACC + fmm_blockstep here, not elsewhere.

    Pinned against calling the lane directly rather than merely checking the
    shape: a selector that fell through to the forward coupler would still return
    a plausible ``(N, 2, 3)`` state, and that is precisely the failure this has to
    catch.
    """
    from odisseo.integration_api import integrate
    from odisseo.option_classes import FMM_ACC

    state, mass = system
    options = _options()
    cfg = config._replace(
        acceleration_scheme=FMM_ACC,
        fmm_blockstep=True,
        blockstep_options=options,
        num_timesteps=2,
    )
    # Ask integrate() which lane it took. This is the EXACT half of the claim --
    # a string, so it is deterministic on every platform -- and it is what makes
    # this test as strong as its front-door sibling. Without it the only evidence
    # was a float comparison that a same-input control shows cannot distinguish
    # two lanes at any tolerance.
    from odisseo.integration_api import resolve_lane

    assert resolve_lane(cfg) == "fmm_blockstep"
    through_integrate, lane = integrate(state, mass, cfg, params, return_lane=True)
    assert lane == "fmm_blockstep", (
        "integrate() ran a different lane than resolve_lane() reports -- the "
        "dispatch and the report have drifted apart"
    )

    direct = integrate_blockstep_jaccpot(
        state, mass, config, params, options=options, n_base=2, track_energy=False
    )
    assert through_integrate.shape == state.shape
    # Coarse check, not rtol=0 -- see the measured control in the front-door
    # test above. This is now a corroborating check rather than the whole one:
    # the lane assertion above carries the claim exactly. It still adds
    # something -- it catches a lane that reports "fmm_blockstep" but computes
    # something else -- and a selector falling through to the forward coupler
    # differs at O(1), not at 1e-15.
    assert jnp.allclose(
        through_integrate, direct.state, rtol=1e-9, atol=1e-12
    ), "integrate() did not dispatch to the block-step lane"


def test_integrate_requires_blockstep_options(system, config, params):
    """``blockstep_options=None`` must raise, not be defaulted.

    Written the other way round first, asserting that None meant "the defaults".
    It does not and must not: ``BlockStepOptions`` has no default ``dt_max``, and a
    ``dt_max`` below every particle's own criterion puts the whole system on rung 0,
    collapsing the block scheme to a shared timestep with no symptom -- a run that
    passes while exercising none of this lane. So the absence of a default is the
    feature, and this pins the error message that explains it.
    """
    from odisseo.integration_api import integrate
    from odisseo.option_classes import FMM_ACC

    state, mass = system
    cfg = config._replace(
        acceleration_scheme=FMM_ACC,
        fmm_blockstep=True,
        blockstep_options=None,
        num_timesteps=1,
    )
    with pytest.raises(ValueError, match="needs config.blockstep_options"):
        integrate(state, mass, cfg, params)


def test_integrate_rejects_a_bad_blockstep_config(system, config, params):
    from odisseo.integration_api import integrate
    from odisseo.option_classes import FMM_ACC

    state, mass = system
    base = config._replace(acceleration_scheme=FMM_ACC, fmm_blockstep=True)

    with pytest.raises(TypeError, match="must be a BlockStepOptions"):
        integrate(
            state, mass,
            base._replace(blockstep_options={"k_max": 2}, num_timesteps=1),
            params,
        )

    with pytest.raises(ValueError, match="number of base steps"):
        integrate(
            state, mass,
            base._replace(blockstep_options=_options(), num_timesteps=0),
            params,
        )
