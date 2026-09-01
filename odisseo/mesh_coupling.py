"""Multi-GPU ("mesh") FMM lane: a distributed rollout over jaccpot's mesh force.

WHY THIS MODULE EXISTS
----------------------
``jaccpot.distributed.fmm`` exposes a *force* on a device mesh and nothing else --
no leapfrog, no Verlet, no scan over steps. ODISSEO's other four lanes are all
single-device. So a multi-GPU force at 10^7 particles had, until this lane, nowhere
to go. ``tools/mesh_galaxy_run.py`` is the script this was factored out of, and it
remains the reference for the numbers quoted below.

WHAT IS VALIDATED
-----------------
262 144 particles / 2 cards / 40 steps: dL/L 2.4e-06, COM drift 3.3e-04, zero
overflow flags, 1.06 s/step. And 20 971 520 / 5 cards / 127 steps with rendering:
dL/L 2.63e-06, median 69.47 s/step.

TWO TRAPS THIS MODULE EXISTS TO CONTAIN
---------------------------------------
**1. The evaluator's output rows are permuted, even with ZERO padding.**
``make_force_evaluator`` returns rows in the per-device Morton order the tree build
produced. ``scatter_to_input_order``'s docstring says the maps "agree whenever no
device is padded" -- true of them as *maps*, false of *row order*, and an easy
sentence to misread. Reading the force on that assumption gives every particle a
Morton neighbour's acceleration: smooth, plausible, and wrong by tens of percent.
Measured on two CPU devices at cap == count == 512, a naive read is wrong on **1022
of 1024 rows**.

A rollout cannot afford a host-side scatter per step, so it realigns ON DEVICE: the
partition is frozen, so each device owns a fixed set of global ids and the tree only
permutes *within* a device. That makes the realignment an argsort plus two gathers
inside one ``shard_map``, with no collective. :func:`verify_alignment` checks it
against ``scatter_to_input_order`` on the first force after every (re)partition, and
that check should never become optional.

**2. The force must NOT be fused into one jit with the integrator.**
At 21M on five cards that puts the traversal buffers and the integrator temporaries
in a single live range; one device fails an allocation, never joins the
``AllGather``, and the other four hang at the rendezvous FOREVER -- a deadlock at
0 % GPU utilisation, not an OOM message. Drift, force and kick are three separate
dispatches with ``donate_argnums``, so peak is the max of the two rather than the
sum. Do not "simplify" that.

WHAT THIS LANE DOES NOT REPORT
------------------------------
Total energy. The distributed evaluator returns accelerations only
(``compute_potential=False`` is hardwired in ``jaccpot/distributed/fmm.py``), so a
self-gravity potential would have to be estimated by a subsampled direct sum -- and
an estimator is not a conservation check. Momentum, angular momentum, centre-of-mass
drift and kinetic energy are exact, and are what :class:`MeshInvariants` carries;
``self_potential_energy`` is a documented ``None``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "MeshOptions",
    "MeshPartition",
    "MeshInvariants",
    "MeshResult",
    "mesh_available",
    "build_mesh_partition",
    "make_aligner",
    "verify_alignment",
    "assert_no_capacity_overflow",
    "mesh_invariants",
    "integrate_mesh_jaccpot",
]


@dataclass(frozen=True)
class MeshOptions:
    """Configuration for the multi-GPU mesh lane.

    Frozen so it is hashable and therefore safe to hang off ``SimulationConfig``,
    which is a ``NamedTuple`` used as a static jit argument.

    Parameters
    ----------
    dt : float
        Fixed timestep. No default, deliberately: this lane runs a fixed-step KDK
        and a silently-guessed dt is the worst kind of wrong.
    ndev : int
        Devices to run on. The mesh takes the first ``ndev`` local devices.
    leaf_size : int
        Particles per tree leaf.
    theta : float
        Opening angle.
    order : int
        Multipole expansion order.
    softening : float or None
        Plummer softening. ``None`` derives ``0.5 * rdisk / sqrt(N / 1e5)``.
    partitioner : str
        ``"rcb"`` or ``"morton"``.
    m2l_chunk : int
        Far-field M2L chunk; bounds peak memory in the far field.
    nearfield_chunk : int
        Near-field target-leaf block; bounds peak memory in the near field.
    working_dtype : str
        ``"float32"`` or ``"float64"``. The lane inherits its working precision
        from the input arrays, so this -- not ``jax_enable_x64`` -- selects it.
    nearfield_accum : str
        Near-field accumulator width passed to jaccpot. ``"wide"`` buys ~439x in
        force accuracy for ~2 % in time and is usually the right choice here.
    repartition_every : int
        Re-partition every N steps; 0 disables. The RCB domains are built once and
        frozen, and the cross-domain near list grows about 0.3 % per step as
        particles move away from the decomposition they were given.
    verify_alignment : bool
        Check the row realignment against ``scatter_to_input_order`` on the first
        force after every partition. Leave this on.
    check_overflow_every : int
        Re-check capacity overflow every N steps; 0 checks only the first force.
        Caps are static but pair counts grow as the system clusters, so an overflow
        can switch on mid-run and silently truncate the force.
    mac_type : str
        Multipole acceptance criterion: ``bh``, ``engblom``, ``dehnen`` (geometric,
        the default) or ``dehnen_error`` (the mass-dependent criterion, which needs
        ``adaptive_eps``).
    adaptive_eps : float or None
        Relative force-accuracy target for ``mac_type="dehnen_error"``. Mandatory
        under that criterion and ignored otherwise.
    mac_cross_criterion : bool
        Whether the criterion also decides cross-domain pairs. Default True; False
        is the self-only ablation.
    external_acceleration : callable or None
        ``(positions) -> (N, 3)`` added to the self-gravity force each step. Used
        for an analytic halo the IC did not sample.
    axis_name : str
        Mesh axis name; must match jaccpot's.
    """

    dt: float
    ndev: int = 2
    leaf_size: int = 512
    theta: float = 0.7
    order: int = 6
    softening: Optional[float] = None
    partitioner: str = "rcb"
    m2l_chunk: int = 65_536
    nearfield_chunk: int = 512
    working_dtype: str = "float32"
    nearfield_accum: str = "wide"
    repartition_every: int = 0
    # Dehnen (2014) section 5 mass-dependent MAC. "dehnen" is the geometric sphere
    # MAC (theta gates both walks). Under "dehnen_error" theta stops gating the SELF
    # walk entirely and `adaptive_eps` replaces it as the accuracy knob, while theta
    # still gates the CROSS walk's geometry. Measured on the mesh at N=1048576:
    # 4.4-7.7x better p99.99 force error at 0.77-0.88x the near-field work, growing
    # to 8.8-15.1x at four devices -- most of it in the CROSS walk, which is why
    # `mac_cross_criterion` defaults on (the self-only ablation is 1.87x at MORE work).
    mac_type: str = "dehnen"
    adaptive_eps: Optional[float] = None
    mac_cross_criterion: bool = True
    verify_alignment: bool = True
    check_overflow_every: int = 10
    external_acceleration: Optional[Callable[[Any], Any]] = None
    axis_name: str = "gpus"

    def __post_init__(self) -> None:
        """Validate the option combination.

        Raises
        ------
        ValueError
            If any option is outside its supported range.
        """
        if not (self.dt > 0):
            raise ValueError(f"dt must be > 0, got {self.dt!r}")
        if self.ndev < 1:
            raise ValueError(f"ndev must be >= 1, got {self.ndev!r}")
        if self.leaf_size < 1:
            raise ValueError(f"leaf_size must be >= 1, got {self.leaf_size!r}")
        if not (self.theta > 0):
            raise ValueError(f"theta must be > 0, got {self.theta!r}")
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order!r}")
        if self.working_dtype not in ("float32", "float64"):
            raise ValueError(
                f"working_dtype must be float32 or float64, got {self.working_dtype!r}"
            )
        if self.partitioner not in ("rcb", "morton"):
            raise ValueError(
                f"partitioner must be rcb or morton, got {self.partitioner!r}"
            )
        if self.mac_type not in ("bh", "engblom", "dehnen", "dehnen_error"):
            raise ValueError(
                f"mac_type must be one of bh/engblom/dehnen/dehnen_error, got "
                f"{self.mac_type!r}. 'dehnen_theta' is REFUTED (12-9300x worse error "
                f"at 1.35-15x the work) and the distributed lane rejects it outright."
            )
        if self.mac_type == "dehnen_error" and self.adaptive_eps is None:
            # Not a defaultable knob: under the criterion theta gates nothing on the
            # self walk, so an unset eps would hand accuracy to a knob that decides
            # nothing -- and the run would look fine while being far less accurate.
            raise ValueError(
                "mac_type='dehnen_error' requires adaptive_eps, which replaces theta "
                "as the self walk's accuracy knob."
            )
        if self.adaptive_eps is not None and not (self.adaptive_eps > 0):
            raise ValueError(
                f"adaptive_eps must be > 0, got {self.adaptive_eps!r}"
            )


@dataclass(frozen=True)
class MeshPartition:
    """A frozen device decomposition plus the maps a rollout needs to read it back.

    Parameters
    ----------
    pos_flat : Any
        ``(ndev * cap, 3)`` positions in device-major order.
    mass_flat : Any
        ``(ndev * cap,)`` masses.
    gid_flat : Any
        ``(ndev * cap,)`` global ids; ``-1`` marks a padding row.
    counts : Any
        ``(ndev,)`` real particle count per device.
    rank_in : Any
        ``(ndev * cap,)`` rank of each row's global id among the ids its own device
        holds. Precomputed on the host because the input order is frozen.
    order_ix : Any
        ``(n,)`` row -> original particle index.
    cap : int
        Rows per device.
    ndev : int
        Device count.
    n : int
        Real particle count.
    """

    pos_flat: Any
    mass_flat: Any
    gid_flat: Any
    counts: Any
    rank_in: Any
    order_ix: Any
    cap: int
    ndev: int
    n: int


@dataclass(frozen=True)
class MeshInvariants:
    """Conserved quantities for a mesh rollout.

    Parameters
    ----------
    momentum : Any
        Total linear momentum, ``(3,)``.
    angular_momentum : Any
        Total angular momentum, ``(3,)``.
    com : Any
        Centre of mass, ``(3,)``.
    kinetic_energy : float
        Total kinetic energy.
    external_potential_energy : float or None
        Potential energy in the external field, when one is supplied.
    self_potential_energy : None
        Always ``None``. The distributed evaluator returns accelerations only, so
        self-gravity potential is unavailable without a jaccpot change; a
        subsampled estimator would not be a conservation check.
    """

    momentum: Any
    angular_momentum: Any
    com: Any
    kinetic_energy: float
    external_potential_energy: Optional[float] = None
    self_potential_energy: None = None


@dataclass
class MeshResult:
    """What a mesh rollout returns.

    Parameters
    ----------
    state : Any
        Final ``(N, 2, 3)`` state in the caller's input order.
    invariants : list
        One :class:`MeshInvariants` per recorded step.
    step_times : list
        Wall-clock seconds per step.
    first_diag : dict
        Per-device diagnostics from the first force.
    n : int
        Particle count actually integrated.
    cap : int
        Rows per device.
    softening : float
        Softening actually used.
    num_repartitions : int
        How many times the decomposition was rebuilt.
    """

    state: Any
    invariants: list = field(default_factory=list)
    step_times: list = field(default_factory=list)
    first_diag: dict = field(default_factory=dict)
    n: int = 0
    cap: int = 0
    softening: float = 0.0
    num_repartitions: int = 0


def mesh_available() -> tuple[bool, str]:
    """Report whether this machine can run the mesh lane, and why not if it cannot.

    Probes all three requirements rather than one. A probe that checks a single
    repository mispredicts and then dies late, in the middle of a run -- the lesson
    of the block-step lane's own availability check.

    Returns
    -------
    tuple[bool, str]
        ``(available, reason)``; ``reason`` is empty when available.
    """
    try:
        from jaccpot.distributed.fmm import make_force_evaluator  # noqa: F401
    except Exception as exc:
        return False, f"jaccpot.distributed.fmm is unavailable: {exc}"
    try:
        from yggdrax.distributed import make_mesh  # noqa: F401
    except Exception as exc:
        return False, f"yggdrax.distributed.make_mesh is unavailable: {exc}"
    try:
        import jax

        n = jax.local_device_count()
    except Exception as exc:  # pragma: no cover - backend discovery is environmental
        return False, f"could not count devices: {exc}"
    if n < 1:
        return False, "no local devices"
    return True, ""


def build_mesh_partition(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    ndev: int,
    leaf_size: int,
    partitioner: str = "rcb",
    allow_padding: bool = False,
) -> MeshPartition:
    """Partition particles across devices and precompute the read-back maps.

    Parameters
    ----------
    positions : np.ndarray
        ``(N, 3)`` positions in the caller's order.
    masses : np.ndarray
        ``(N,)`` masses.
    ndev : int
        Device count.
    leaf_size : int
        Tree leaf size; sets the padding quantum.
    partitioner : str
        ``"rcb"`` or ``"morton"``.
    allow_padding : bool
        Permit a partition where some device is padded. Off by default because
        :func:`make_aligner`'s ``rank_in`` assumes every row is real.

    Returns
    -------
    MeshPartition
        The frozen decomposition.

    Raises
    ------
    ValueError
        If padding would be required and ``allow_padding`` is False.
    """
    from jaccpot.distributed.fmm import partition_for_devices

    part = partition_for_devices(
        positions, masses, ndev, leaf_size=leaf_size, partitioner=partitioner
    )
    cap = int(part["cap"])
    n = int(part["n"])
    if cap * ndev != n and not allow_padding:
        raise ValueError(
            f"this partition pads: cap={cap} x ndev={ndev} != n={n}. The mesh lane "
            f"wants N = ndev * k * leaf_size so every row is real; trim to "
            f"{(n // (ndev * leaf_size)) * ndev * leaf_size} particles, or pass "
            "allow_padding=True and accept that the realignment check is then only "
            "meaningful on the rows with gid >= 0."
        )
    gid_flat = np.asarray(part["gid_flat"])
    rank_in = np.empty(ndev * cap, np.int32)
    for d in range(ndev):
        sl = slice(d * cap, (d + 1) * cap)
        # Stable, so the -1 padding ids tie in a defined order on the padded path.
        rank_in[sl] = np.argsort(
            np.argsort(gid_flat[sl], kind="stable"), kind="stable"
        ).astype(np.int32)
    return MeshPartition(
        pos_flat=part["pos_flat"],
        mass_flat=part["mass_flat"],
        gid_flat=gid_flat,
        counts=part["counts"],
        rank_in=rank_in,
        order_ix=gid_flat.astype(np.int64),
        cap=cap,
        ndev=ndev,
        n=n,
    )


def make_aligner(mesh: Any, *, axis_name: str = "gpus") -> Callable:
    """Build a device-local realignment of the evaluator's output to input order.

    ``rank_in`` is a *traced argument* rather than a closure, so a re-partition does
    not force a recompile of the aligner.

    Parameters
    ----------
    mesh : Any
        The device mesh.
    axis_name : str
        Mesh axis name.

    Returns
    -------
    Callable
        ``(values, gid_out, rank_in) -> values`` in input row order.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    def local(v, gid_out, rk):
        ascending = v[jnp.argsort(gid_out.reshape(-1))]
        return ascending[rk]

    fn = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(axis_name, None), P(axis_name, None), P(axis_name)),
        out_specs=P(axis_name, None),
        check_vma=False,
    )
    return jax.jit(fn)


def verify_alignment(
    aligned: Any, accel_raw: Any, gid_out: Any, gid_in: Any, n: int
) -> None:
    """Check the on-device realignment against the repo's own host-side scatter.

    Verify the PERMUTATION, before any arithmetic touches it. A permutation is
    exact, so an equality test is the right one -- but only on a value that has not
    been through an add. Subtracting an external-field term back off reintroduces
    float32 rounding and turns an exact check into a ~1e-6 mismatch on 80 % of rows,
    which looks like a broken map and is not one.

    Parameters
    ----------
    aligned : Any
        The realigned accelerations.
    accel_raw : Any
        The evaluator's raw output, before realignment.
    gid_out : Any
        Global ids as the evaluator returned them.
    gid_in : Any
        The ``gid_flat`` that went IN. Load-bearing:
        ``scatter_to_input_order`` returns rows in GLOBAL particle order, while the
        aligner returns them in the partition's ROW order. Comparing the two
        directly reports every row as wrong -- which is what happens if this
        argument is dropped.
    n : int
        Real particle count.

    Raises
    ------
    AssertionError
        If the on-device realignment disagrees with ``scatter_to_input_order``.
    """
    from jaccpot.distributed.fmm import scatter_to_input_order

    ref = np.asarray(scatter_to_input_order(accel_raw, gid_out, n))
    gi = np.asarray(gid_in).reshape(-1).astype(np.int64)
    real = gi >= 0
    ref_rows = ref[gi[real]]
    got = np.asarray(aligned).reshape(-1, ref.shape[-1])[real]
    if not np.array_equal(got.astype(ref.dtype), ref_rows):
        bad = int(np.sum(np.any(got.astype(ref.dtype) != ref_rows, axis=-1)))
        raise AssertionError(
            f"on-device realignment disagrees with scatter_to_input_order on {bad} "
            f"of {n} rows. Do not 'fix' this by relaxing the tolerance: a "
            "permutation is exact, and a near-miss here means every particle is "
            "reading a Morton neighbour's acceleration."
        )


def assert_no_capacity_overflow(diag: Any, *, where: str) -> None:
    """Raise if any device reported a capacity overflow.

    Reads the field names from jaccpot rather than re-listing them, so a new
    diagnostic cannot go unchecked here.

    Parameters
    ----------
    diag : Any
        ``(ndev, len(DIAG_FIELDS))`` diagnostics from the evaluator.
    where : str
        Context for the error message, e.g. ``"step 40"``.

    Raises
    ------
    RuntimeError
        If any overflow flag is set on any device.
    """
    from jaccpot.distributed.fmm import DIAG_FIELDS

    d = np.asarray(diag)
    fired = {
        name: float(d[:, i].sum())
        for i, name in enumerate(DIAG_FIELDS)
        if name.endswith("overflow") and i < d.shape[1] and d[:, i].sum() > 0
    }
    if fired:
        raise RuntimeError(
            f"capacity overflow at {where}: {fired}. The force is truncated and the "
            "run would read FASTER while being wrong, so it is stopped here. Caps "
            "are static but pair counts grow as the system clusters; raise the "
            "relevant cap or loosen theta."
        )


def mesh_invariants(pos: Any, vel: Any, mass: Any) -> MeshInvariants:
    """Compute the exactly-conserved quantities for a sharded state.

    Padding rows carry mass 0, so every moment is already correct with no mask.

    Parameters
    ----------
    pos : Any
        ``(rows, 3)`` positions.
    vel : Any
        ``(rows, 3)`` velocities.
    mass : Any
        ``(rows,)`` masses.

    Returns
    -------
    MeshInvariants
        Momentum, angular momentum, centre of mass and kinetic energy.
    """
    import jax.numpy as jnp

    p = jnp.sum(mass[:, None] * vel, axis=0)
    ell = jnp.sum(mass[:, None] * jnp.cross(pos, vel), axis=0)
    com = jnp.sum(mass[:, None] * pos, axis=0) / jnp.sum(mass)
    ke = 0.5 * jnp.sum(mass * jnp.sum(vel * vel, axis=1))
    return MeshInvariants(
        momentum=np.asarray(p),
        angular_momentum=np.asarray(ell),
        com=np.asarray(com),
        kinetic_energy=float(ke),
    )


def integrate_mesh_jaccpot(
    primitive_state: Any,
    mass: Any,
    config: Any,
    params: Any,
    *,
    options: MeshOptions,
    n_steps: int,
    progress: bool = False,
) -> MeshResult:
    """Kick-drift-kick a system across a device mesh, over jaccpot's distributed FMM.

    One force evaluation per step, over an evaluator compiled once. Positions,
    velocities and accelerations stay sharded for the whole rollout; only a handful
    of scalars ever crosses to the host.

    Parameters
    ----------
    primitive_state : Any
        ``(N, 2, 3)`` positions and velocities.
    mass : Any
        ``(N,)`` masses.
    config : Any
        ``SimulationConfig``; ``config.softening`` is used when
        ``options.softening`` is None.
    params : Any
        ``SimulationParams``; ``params.G`` is used.
    options : MeshOptions
        Lane configuration.
    n_steps : int
        Number of fixed steps.
    progress : bool
        Print a line per recorded step.

    Returns
    -------
    MeshResult
        Final state in the caller's input order, plus diagnostics.

    Raises
    ------
    RuntimeError
        If the mesh lane is unavailable, or a capacity overflow fires.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    ok, why = mesh_available()
    if not ok:
        raise RuntimeError(f"the mesh lane is unavailable: {why}")

    from jaccpot.distributed.fmm import DistributedFMMConfig, make_force_evaluator
    from yggdrax.distributed import make_mesh

    wdt = np.float32 if options.working_dtype == "float32" else np.float64
    state = np.asarray(primitive_state)
    pos = np.ascontiguousarray(state[:, 0, :], dtype=wdt)
    vel = np.ascontiguousarray(state[:, 1, :], dtype=wdt)
    m = np.asarray(mass, dtype=wdt)

    soft = options.softening
    if soft is None:
        soft = float(getattr(config, "softening", 0.0)) or 1e-3

    part = build_mesh_partition(
        pos,
        m,
        ndev=options.ndev,
        leaf_size=options.leaf_size,
        partitioner=options.partitioner,
    )
    cfg = DistributedFMMConfig(
        leaf_size=options.leaf_size,
        theta=options.theta,
        order=options.order,
        softening=soft,
        G=float(getattr(params, "G", 1.0)),
        m2l_chunk=options.m2l_chunk,
        nearfield_chunk=options.nearfield_chunk,
        nearfield_accum=options.nearfield_accum,
        mac_type=options.mac_type,
        adaptive_eps=options.adaptive_eps,
        mac_cross_criterion=bool(options.mac_cross_criterion),
    ).resolved_for(part.cap, options.ndev)

    mesh = make_mesh(options.ndev)
    evaluate = make_force_evaluator(cfg, options.ndev, part.cap, mesh, jit=True)
    align = make_aligner(mesh, axis_name=options.axis_name)

    shard2 = NamedSharding(mesh, P(options.axis_name, None))
    shard1 = NamedSharding(mesh, P(options.axis_name))
    X = jax.device_put(jnp.asarray(part.pos_flat), shard2)
    M = jax.device_put(jnp.asarray(part.mass_flat), shard1)
    V = jax.device_put(jnp.asarray(vel[part.order_ix]), shard2)
    RANK = jax.device_put(jnp.asarray(part.rank_in), shard1)
    GID = jnp.asarray(part.gid_flat)
    COUNTS = jnp.asarray(part.counts)

    ext = options.external_acceleration

    def force(x):
        a_raw, gid_o, diag = evaluate(x, M, GID, COUNTS)
        a = align(a_raw, gid_o, RANK)
        if ext is not None:
            a = a + ext(x)
        return a, gid_o, diag, a_raw

    # The KDK arithmetic is deliberately NOT fused into one jit with the force.
    # Fusing them makes XLA hold the evaluator's traversal buffers and the
    # integrator's temporaries in a single live range; at 21M on five cards that
    # overflows, one device fails an allocation, never joins the AllGather, and the
    # other four hang at the rendezvous forever -- a deadlock at 0 % utilisation,
    # not an OOM message. Three dispatches keep peak at the max, not the sum.
    from functools import partial

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def drift(x, v, a, dt):
        vh = v + 0.5 * dt * a
        return x + dt * vh, vh

    @partial(jax.jit, donate_argnums=(0, 1))
    def kick(vh, a, dt):
        return vh + 0.5 * dt * a

    A, gid_o, diag, A_raw = force(X)
    jax.block_until_ready(A)
    assert_no_capacity_overflow(diag, where="the first force")
    if options.verify_alignment:
        verify_alignment(
            align(A_raw, gid_o, RANK), A_raw, gid_o, GID, part.n
        )

    from jaccpot.distributed.fmm import DIAG_FIELDS

    d0 = np.asarray(diag)
    first_diag = {
        name: [float(v) for v in d0[:, i]]
        for i, name in enumerate(DIAG_FIELDS)
        if i < d0.shape[1]
    }

    result = MeshResult(
        state=None,
        first_diag=first_diag,
        n=part.n,
        cap=part.cap,
        softening=float(soft),
    )
    dt = float(options.dt)
    for it in range(1, int(n_steps) + 1):
        t0 = time.perf_counter()
        X, VH = drift(X, V, A, dt)
        A, gid_o, diag, _ = force(X)
        V = kick(VH, A, dt)
        jax.block_until_ready(X)
        result.step_times.append(time.perf_counter() - t0)
        if options.check_overflow_every and it % options.check_overflow_every == 0:
            assert_no_capacity_overflow(diag, where=f"step {it}")
        if progress:
            inv = mesh_invariants(X, V, M)
            result.invariants.append(inv)
            print(f"  step {it:>6d}  {result.step_times[-1]:7.2f}s  KE={inv.kinetic_energy:.6e}")

    # Back to the caller's row order. `order_ix` maps row -> original index, so the
    # inverse permutation puts particle i back at row i.
    inv_ix = np.empty(part.n, np.int64)
    inv_ix[part.order_ix[: part.n]] = np.arange(part.n)
    out = np.asarray(state, dtype=wdt).copy()
    out[:, 0, :] = np.asarray(X)[: part.n][inv_ix]
    out[:, 1, :] = np.asarray(V)[: part.n][inv_ix]
    result.state = out
    return result
