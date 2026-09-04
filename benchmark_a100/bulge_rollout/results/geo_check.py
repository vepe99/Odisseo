"""Is the geometric arm OVERFLOWING? Flags + accuracy vs an fp64 direct sum, one shot.

The geo determinism arm showed a median run-to-run deviation of 2.4e-02 against 1.7e-07 for
the criterion arms. Before calling that "the geometric MAC is non-deterministic", check the
confound the arm script never checked: geo's derived cross caps are HALF the criterion's
(cross_q 2,097,152 vs 4,194,304; cross_int 8,192 vs 16,384) while its cross_near is HIGHER
(17.08M vs 14.02M). A truncated walk is wrong, and can be wrong differently each time.
"""
import sys, numpy as np
sys.path.insert(0, "/export/home/tbuck/Odisseo")
IC = "/export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz"
NDEV, LEAF, THETA, ORDER = 4, 1024, 0.7, 6
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from odisseo import mesh_coupling
from jaccpot.distributed.fmm import DistributedFMMConfig, make_force_evaluator, DIAG_FIELDS
from yggdrax.distributed import make_mesh
from tools.mesh_galaxy_run import direct_sum_probe

ic = np.load(IC); mass = np.asarray(ic["mass"])
pos = np.ascontiguousarray(np.asarray(ic["state0"])[:, 0, :], dtype=np.float32)
n = len(mass); soft = float(0.5*float(ic["rdisk_code"])/np.sqrt(n/1e5))
part = mesh_coupling.build_mesh_partition(pos, mass, ndev=NDEV, leaf_size=LEAF, partitioner="rcb")
mesh = make_mesh(NDEV)
for mac, extra in (("dehnen", {}), ("dehnen_error", dict(adaptive_eps=1e-5, mac_cross_criterion=True))):
    cfg = DistributedFMMConfig(leaf_size=LEAF, theta=THETA, order=ORDER, softening=soft, G=1.0,
        nearfield_accum="wide", mac_type=mac, m2l_chunk=65536, nearfield_chunk=512,
        **extra).resolved_for(part.cap, NDEV)
    ev = make_force_evaluator(cfg, NDEV, part.cap, mesh, jit=True)
    al = mesh_coupling.make_aligner(mesh)
    s2 = NamedSharding(mesh, P("gpus", None)); s1 = NamedSharding(mesh, P("gpus"))
    X = jax.device_put(jnp.asarray(part.pos_flat), s2)
    M = jax.device_put(jnp.asarray(part.mass_flat), s1)
    R = jax.device_put(jnp.asarray(part.rank_in), s1)
    a_raw, gid_o, diag = ev(X, M, jnp.asarray(part.gid_flat), jnp.asarray(part.counts))
    A = np.asarray(al(a_raw, gid_o, R))[:n]
    D = np.asarray(diag)
    dec = {k: [float(v) for v in D[:, i]] for i, k in enumerate(DIAG_FIELDS) if i < D.shape[1]}
    ovf = {k: sum(v) for k, v in dec.items() if k.endswith("overflow")}
    hot = {k: v for k, v in ovf.items() if v > 0}
    rng = np.random.default_rng(20260902)
    tg = np.sort(rng.choice(n, size=256, replace=False))
    ref = direct_sum_probe(np.asarray(jax.device_get(X))[:n], np.asarray(jax.device_get(M))[:n],
                           tg, soft, 1.0)
    got = A[tg].astype(np.float64)
    num = np.linalg.norm(got-ref, axis=1); den = np.linalg.norm(ref, axis=1)
    rl2 = float(np.linalg.norm(num)/np.linalg.norm(den))
    print(f"\n=== {mac} ===", flush=True)
    print(f"  caps: cross_q={cfg.cross_max_pair_queue:,} cross_nbr={cfg.cross_max_neighbors_per_leaf:,} "
          f"cross_int={cfg.cross_max_interactions_per_node:,}")
    print(f"  pairs: self_near={sum(dec['self_near_pairs']):,.0f} cross_near={sum(dec['cross_near_pairs']):,.0f} "
          f"cross_far={sum(dec['cross_far_pairs']):,.0f}")
    print(f"  OVERFLOW: {hot if hot else 'all flags clear'}")
    print(f"  rel_l2 vs fp64 direct sum (256 targets) = {rl2:.4e}   non-finite = {int(np.count_nonzero(~np.isfinite(A)))}")
