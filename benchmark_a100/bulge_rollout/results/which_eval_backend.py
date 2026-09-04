"""For each MAC: score EVERY evaluation against the fp64 truth. Which call is right?

The Sep-2 evidence left a contradiction for the geometric MAC: its first call is 5.28e-03
from the fp64 direct sum, yet it disagrees with every LATER call by 2.7-4.2 % median (max
2.7e+28), while those later calls agree with each other to 1e-7..1e-5. Either the first call
is right and the later ones are wrong, or the reverse. Nothing so far scored the later calls
against truth. This does, for both MACs, in one process, against ONE shared fp64 reference.

    eval k -> rel_l2(A_k, truth) = ||A_k - ref||_2 / ||ref||_2 over the same 512 targets
    plus the full pairwise rel_l2 matrix (proper norms, no near-zero-denominator pathology)
    plus pair counts per eval.

Same configuration as every Sep-2 arm (17,825,792 / 4 cards / leaf 1024) so the numbers sit
next to that evidence rather than beside it.
"""
import sys, time, json
import numpy as np
sys.path.insert(0, "/export/home/tbuck/Odisseo")
IC = "/export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz"
NDEV, LEAF, THETA, ORDER, EPS = 4, 1024, 0.7, 6, 1e-5
M2L_CHUNK, NEARFIELD_CHUNK = 65536, 512     # the run script's values; the config's own
                                            # defaults are None = full batch = 292 GiB OOM
NPROBE, NEVAL = 512, 4
import os; BACKEND = os.environ.get("NF_BACKEND", "auto"); OUT = f"/export/scratch/tbuck/odisseo_runs/which_eval_geo_{BACKEND}.json"

import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from odisseo import mesh_coupling
from jaccpot.distributed.fmm import DistributedFMMConfig, make_force_evaluator, DIAG_FIELDS
from yggdrax.distributed import make_mesh
from tools.mesh_galaxy_run import direct_sum_probe
import yggdrax
print(f"devices={[d.id for d in jax.devices()]}  yggdrax={yggdrax.__file__}", flush=True)

ic = np.load(IC); st = np.asarray(ic["state0"]); mass = np.asarray(ic["mass"])
pos = np.ascontiguousarray(st[:, 0, :], dtype=np.float32); n = len(mass)
soft = float(0.5 * float(ic["rdisk_code"]) / np.sqrt(n / 1e5))
part = mesh_coupling.build_mesh_partition(pos, mass, ndev=NDEV, leaf_size=LEAF, partitioner="rcb")
mesh = make_mesh(NDEV)
s2 = NamedSharding(mesh, P("gpus", None)); s1 = NamedSharding(mesh, P("gpus"))
X = jax.device_put(jnp.asarray(part.pos_flat), s2); M = jax.device_put(jnp.asarray(part.mass_flat), s1)
R = jax.device_put(jnp.asarray(part.rank_in), s1); GID = jnp.asarray(part.gid_flat); CNT = jnp.asarray(part.counts)
al = mesh_coupling.make_aligner(mesh)

# ONE fp64 reference, shared by every evaluation of every MAC.
pos_rows = np.asarray(jax.device_get(X))[:n]; mass_rows = np.asarray(jax.device_get(M))[:n]
rng = np.random.default_rng(20260902); targets = np.sort(rng.choice(n, size=NPROBE, replace=False))
t0 = time.perf_counter(); ref = direct_sum_probe(pos_rows, mass_rows, targets, soft, 1.0)
ref_norm = float(np.linalg.norm(ref)); print(f"# fp64 reference for {NPROBE} targets: {time.perf_counter()-t0:.0f} s", flush=True)

def rel_l2_vs_truth(A):  return float(np.linalg.norm(A[:n][targets].astype(np.float64) - ref) / ref_norm)
def rel_l2_pair(A, B):   return float(np.linalg.norm(A - B) / np.linalg.norm(A))

results = {}
for mac, extra in (("dehnen", dict(nearfield_backend=BACKEND)),):
    print(f"\n================ {mac}  nearfield_backend={BACKEND} ================", flush=True)
    cfg = DistributedFMMConfig(leaf_size=LEAF, theta=THETA, order=ORDER, softening=soft, G=1.0,
                               nearfield_accum="wide", mac_type=mac, m2l_chunk=M2L_CHUNK,
                               nearfield_chunk=NEARFIELD_CHUNK, **extra).resolved_for(part.cap, NDEV)
    ev = make_force_evaluator(cfg, NDEV, part.cap, mesh, jit=True)
    As, counts, truth, nonfin = [], [], [], []
    for k in range(NEVAL):
        t0 = time.perf_counter()
        a_raw, gid_o, diag = ev(X, M, GID, CNT); a = al(a_raw, gid_o, R); jax.block_until_ready(a)
        A = np.asarray(a); As.append(A)
        D = np.asarray(diag)
        c = {f: int(sum(D[:, i])) for i, f in enumerate(DIAG_FIELDS)
             if f.endswith("_pairs") and i < D.shape[1]}
        ovf = {f: float(sum(D[:, i])) for i, f in enumerate(DIAG_FIELDS) if f.endswith("overflow") and i < D.shape[1]}
        counts.append(c); truth.append(rel_l2_vs_truth(A)); nonfin.append(int(np.count_nonzero(~np.isfinite(A))))
        print(f"  eval {k}: {time.perf_counter()-t0:6.1f} s  rel_l2 vs TRUTH = {truth[-1]:.6e}   non-finite={nonfin[-1]}"
              f"   self_near={c.get('self_near_pairs',0):,} cross_near={c.get('cross_near_pairs',0):,}"
              f" cross_far={c.get('cross_far_pairs',0):,}   overflow={'clear' if not any(ovf.values()) else ovf}", flush=True)
    mat = {}
    print("  pairwise rel_l2 (proper norms):")
    for i in range(NEVAL):
        row = "   " + f"{i}: " + " ".join(f"{rel_l2_pair(As[i], As[j]):.3e}" if j > i else "         " for j in range(NEVAL))
        print(row)
        for j in range(i + 1, NEVAL): mat[f"{i}-{j}"] = rel_l2_pair(As[i], As[j])
    best = int(np.argmin(truth)); worst = int(np.argmax(truth))
    spread = (max(truth) - min(truth)) / min(truth)
    print(f"  => most accurate eval: {best} ({truth[best]:.6e}); least: {worst} ({truth[worst]:.6e}); spread {spread:.2e}")
    print(f"  => {'ALL evals equally accurate: non-determinism is round-off only' if spread < 0.05 else 'EVALS DIFFER IN ACCURACY: a real defect, not round-off'}")
    results[mac] = dict(rel_l2_vs_truth=truth, pair_counts=counts, pairwise=mat, nonfinite=nonfin,
                        accuracy_spread=spread, caps=dict(self_q=cfg.max_pair_queue, self_int=cfg.max_interactions_per_node,
                        cross_q=cfg.cross_max_pair_queue, cross_int=cfg.cross_max_interactions_per_node))
json.dump(dict(N=n, ndev=NDEV, leaf=LEAF, probe=NPROBE, neval=NEVAL, results=results), open(OUT, "w"), indent=1)
print(f"\n# wrote {OUT}")
