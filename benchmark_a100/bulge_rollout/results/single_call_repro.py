"""Re-evaluate a saved step's positions ONCE, fresh, and score it. Position- or order-dependent?

The rollout probe is accurate at steps 0 and 3 and ~45 % wrong at steps 1 and 2, for BOTH MACs,
which see identical positions. If a single first-call evaluation on the saved X1 reproduces
the 45 %, the fault is a property of those POSITIONS (a tree-shape trigger) and this is a
minimal repro. If it comes back at ~3e-03, the fault depends on call order within a process.

Usage: single_call_repro.py <snapshot.npz> [mac] [snapshot|ic]
"""
import sys, time, json
import numpy as np
sys.path.insert(0, "/export/home/tbuck/Odisseo")
SNAP = sys.argv[1]; MAC = sys.argv[2] if len(sys.argv) > 2 else "dehnen_error"
IC = "/export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz"
NDEV, LEAF, THETA, ORDER, EPS = 4, 1024, 0.7, 6, 1e-5
M2L_CHUNK, NEARFIELD_CHUNK, NPROBE = 65536, 512, 256

import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from odisseo import mesh_coupling
from jaccpot.distributed.fmm import DistributedFMMConfig, make_force_evaluator, DIAG_FIELDS
from yggdrax.distributed import make_mesh
from tools.mesh_galaxy_run import direct_sum_probe

snap = np.load(SNAP); st = np.asarray(snap["state0"]); mass = np.asarray(snap["mass"])
step = int(snap["step"]) if "step" in snap.files else -1
pos = np.ascontiguousarray(st[:, 0, :], dtype=np.float32); n = len(mass)
soft = float(snap["softening"]) if "softening" in snap.files else float(0.5 * 0.24 / np.sqrt(n / 1e5))
print(f"snapshot step {step}: N={n:,} softening={soft:.6f} mac={MAC}", flush=True)

# TWO partitions, because the rollout's is FROZEN at the IC's positions while the particles
# drift. (i) a fresh RCB on the snapshot's own positions -- clean domains, no strays;
# (ii) the ROLLOUT's partition: RCB on the IC positions, with the snapshot's positions
# loaded into that row order -- so particles that drifted out of their device's box are
# still owned by it, exactly as in the rollout. If (ii) reproduces the error and (i) does
# not, the fault is the frozen partition meeting drift, not the positions per se.
PART_FROM = sys.argv[3] if len(sys.argv) > 3 else "snapshot"
if PART_FROM == "ic":
    ic = np.load(IC); pos_ic = np.ascontiguousarray(np.asarray(ic["state0"])[:, 0, :], dtype=np.float32)
    part0 = mesh_coupling.build_mesh_partition(pos_ic, mass, ndev=NDEV, leaf_size=LEAF, partitioner="rcb")
    # same row order as the rollout: row -> original index is part0.order_ix; load X_snap into it
    order = np.asarray(part0.order_ix)[: len(mass)]
    pos_flat = np.zeros_like(np.asarray(part0.pos_flat)); pos_flat[: len(order)] = pos[order]
    class _P: pass
    part = _P(); part.pos_flat = pos_flat; part.mass_flat = part0.mass_flat; part.rank_in = part0.rank_in
    part.gid_flat = part0.gid_flat; part.counts = part0.counts; part.cap = part0.cap; part.order_ix = part0.order_ix
    print(f"partition: FROZEN at the IC (the rollout's), snapshot positions loaded into it", flush=True)
else:
    part = mesh_coupling.build_mesh_partition(pos, mass, ndev=NDEV, leaf_size=LEAF, partitioner="rcb")
    print(f"partition: fresh RCB on the snapshot's own positions", flush=True)
mesh = make_mesh(NDEV)
s2 = NamedSharding(mesh, P("gpus", None)); s1 = NamedSharding(mesh, P("gpus"))
X = jax.device_put(jnp.asarray(part.pos_flat), s2); M = jax.device_put(jnp.asarray(part.mass_flat), s1)
R = jax.device_put(jnp.asarray(part.rank_in), s1); GID = jnp.asarray(part.gid_flat); CNT = jnp.asarray(part.counts)
extra = dict(adaptive_eps=EPS, mac_cross_criterion=True) if MAC == "dehnen_error" else {}
cfg = DistributedFMMConfig(leaf_size=LEAF, theta=THETA, order=ORDER, softening=soft, G=1.0, nearfield_accum="wide",
                           mac_type=MAC, m2l_chunk=M2L_CHUNK, nearfield_chunk=NEARFIELD_CHUNK, **extra).resolved_for(part.cap, NDEV)
ev = make_force_evaluator(cfg, NDEV, part.cap, mesh, jit=True); al = mesh_coupling.make_aligner(mesh)

out = {"snapshot": SNAP, "step": step, "mac": MAC, "partition_from": PART_FROM, "calls": []}
pos_rows = np.asarray(jax.device_get(X))[:n]; mass_rows = np.asarray(jax.device_get(M))[:n]
rng = np.random.default_rng(20260901); targets = np.sort(rng.choice(n, size=NPROBE, replace=False))
ref = direct_sum_probe(pos_rows, mass_rows, targets, soft, 1.0); ref_n = np.linalg.norm(ref)
for k in range(2):   # the FIRST call is the one that matters; the second says whether it changes
    t0 = time.perf_counter()
    a_raw, gid_o, diag = ev(X, M, GID, CNT); a = al(a_raw, gid_o, R); jax.block_until_ready(a)
    A = np.asarray(a)[:n]; D = np.asarray(diag)
    rl2 = float(np.linalg.norm(A[targets].astype(np.float64) - ref) / ref_n)
    c = {f: int(sum(D[:, i])) for i, f in enumerate(DIAG_FIELDS) if f.endswith("_pairs") and i < D.shape[1]}
    ovf = sum(float(sum(D[:, i])) for i, f in enumerate(DIAG_FIELDS) if f.endswith("overflow") and i < D.shape[1])
    print(f"  call {k}: {time.perf_counter()-t0:.0f} s  rel_l2 vs truth = {rl2:.4e}   pairs={c}   overflow_any={ovf}", flush=True)
    per = np.linalg.norm(A[targets].astype(np.float64) - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-300)
    out["calls"].append(dict(rel_l2=rl2, pairs=c, overflow_any=ovf,
                             per_target_rel=per.tolist(), target_rows=targets.tolist(),
                             target_device=(targets // part.cap).tolist()))
np.save(SNAP.replace(".npz", f"_A_{MAC}_{PART_FROM}.npy"), A)
verdict = ("POSITION-DEPENDENT: a fresh first call on these positions reproduces the error"
           if out["calls"][0]["rel_l2"] > 0.05 else
           "NOT reproduced by a fresh call: the fault depends on call order / process state")
print(f"  => {verdict}")
out["verdict"] = verdict
json.dump(out, open(SNAP.replace(".npz", f"_repro_{MAC}_{PART_FROM}.json"), "w"), indent=1)
