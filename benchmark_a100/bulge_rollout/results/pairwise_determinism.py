"""Compare EVERY pair of evaluations, not just each against the first.

The previous arm compared eval k against eval 0 only. If the FIRST call is anomalous -- and
there is direct evidence it is (cross_near differed on eval 0, then was stable for nine
repeats) -- then "every repeat differs from A0" is equally consistent with the repeats
agreeing perfectly WITH EACH OTHER. Those are opposite conclusions and the old script cannot
tell them apart.

Reports a full matrix of pairwise median deviations, and pair counts per eval.
"""
import sys, itertools, json
import numpy as np
sys.path.insert(0, "/export/home/tbuck/Odisseo")
MAC = sys.argv[1]; CROSS = bool(int(sys.argv[2])); NEVAL = int(sys.argv[3]) if len(sys.argv) > 3 else 4
IC = "/export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz"
NDEV, LEAF, THETA, ORDER = 4, 1024, 0.7, 6
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from odisseo import mesh_coupling
from jaccpot.distributed.fmm import DistributedFMMConfig, make_force_evaluator, DIAG_FIELDS
from yggdrax.distributed import make_mesh
ic = np.load(IC); mass = np.asarray(ic["mass"])
pos = np.ascontiguousarray(np.asarray(ic["state0"])[:, 0, :], dtype=np.float32)
n = len(mass); soft = float(0.5*float(ic["rdisk_code"])/np.sqrt(n/1e5))
part = mesh_coupling.build_mesh_partition(pos, mass, ndev=NDEV, leaf_size=LEAF, partitioner="rcb")
mesh = make_mesh(NDEV)
kw = dict(leaf_size=LEAF, theta=THETA, order=ORDER, softening=soft, G=1.0,
          nearfield_accum="wide", mac_type=MAC, m2l_chunk=65536, nearfield_chunk=512)
if MAC == "dehnen_error": kw.update(adaptive_eps=1e-5, mac_cross_criterion=CROSS)
cfg = DistributedFMMConfig(**kw).resolved_for(part.cap, NDEV)
ev = make_force_evaluator(cfg, NDEV, part.cap, mesh, jit=True)
al = mesh_coupling.make_aligner(mesh)
s2 = NamedSharding(mesh, P("gpus", None)); s1 = NamedSharding(mesh, P("gpus"))
X = jax.device_put(jnp.asarray(part.pos_flat), s2)
M = jax.device_put(jnp.asarray(part.mass_flat), s1)
R = jax.device_put(jnp.asarray(part.rank_in), s1)
GID = jnp.asarray(part.gid_flat); CNT = jnp.asarray(part.counts)
PAIR = ("self_near_pairs","self_far_pairs","cross_near_pairs","cross_far_pairs")
As, Ds = [], []
for k in range(NEVAL):
    a_raw, gid_o, diag = ev(X, M, GID, CNT); a = al(a_raw, gid_o, R); jax.block_until_ready(a)
    D = np.asarray(diag)
    As.append(np.asarray(a)[:n])
    Ds.append({f: sum(float(v) for v in D[:, i]) for i, f in enumerate(DIAG_FIELDS)
               if i < D.shape[1] and f in PAIR})
    print(f"  eval {k}: " + "  ".join(f"{f.replace('_pairs','')}={Ds[-1][f]:,.0f}" for f in PAIR), flush=True)
print(f"\n  pairwise median |dA|/|A| (rows/cols = eval index), {MAC} cross={int(CROSS)}:")
hdr = "        " + "".join(f"{j:>12}" for j in range(NEVAL)); print(hdr)
mat = {}
for i in range(NEVAL):
    row = f"  {i:>4}  "
    for j in range(NEVAL):
        if j <= i: row += f"{'':>12}"; continue
        d = np.abs(As[j]-As[i]) / np.maximum(np.abs(As[i]), 1e-30)
        nd = int(np.count_nonzero(As[j].view(np.uint32) != As[i].view(np.uint32)))
        med = float(np.median(d)); mat[f"{i}-{j}"] = {"median": med, "differing_words": nd}
        row += f"{med:>12.3e}"
    print(row, flush=True)
print("\n  bitwise-identical pairs: " +
      (", ".join(k for k, v in mat.items() if v["differing_words"] == 0) or "NONE"))
json.dump({"mac": MAC, "cross": CROSS, "pairs": Ds, "matrix": mat},
          open(f"/export/scratch/tbuck/odisseo_runs/pairwise_{MAC}_{int(CROSS)}.json", "w"), indent=1)
