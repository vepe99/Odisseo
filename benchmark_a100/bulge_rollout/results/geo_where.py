"""WHERE are the geometric MAC's garbage particles? Two calls, save both, localise the difference.

geo is 4.97e-03 from truth on call 0 and 0.31-0.54 on calls 1-3, with an identical accept mask
and 1e-7 per-particle MEDIAN deviation: a small varying set of particles carries garbage. This
saves A0 and A1 (row order) and asks, of the particles whose |A1-A0| is large:
  - which DEVICE they are on          (row // cap)
  - where in the device's row range    (rank/cap: a cluster at the END = an unwritten tail)
  - their |A0| and radius              (bulge centre? halo edge? or uncorrelated?)
  - whether the SAME rows are bad on a third call (fixed rows = buffer; moving = race)
"""
import sys, time, json
import numpy as np
sys.path.insert(0, "/export/home/tbuck/Odisseo")
IC = "/export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz"
NDEV, LEAF, THETA, ORDER = 4, 1024, 0.7, 6
M2L_CHUNK, NEARFIELD_CHUNK = 65536, 512
OUT = "/export/scratch/tbuck/odisseo_runs/geo_where"
import os; os.makedirs(OUT, exist_ok=True)

import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from odisseo import mesh_coupling
from jaccpot.distributed.fmm import DistributedFMMConfig, make_force_evaluator, DIAG_FIELDS
from yggdrax.distributed import make_mesh

ic = np.load(IC); st = np.asarray(ic["state0"]); mass = np.asarray(ic["mass"])
pos = np.ascontiguousarray(st[:, 0, :], dtype=np.float32); n = len(mass)
soft = float(0.5 * float(ic["rdisk_code"]) / np.sqrt(n / 1e5))
part = mesh_coupling.build_mesh_partition(pos, mass, ndev=NDEV, leaf_size=LEAF, partitioner="rcb")
cap = part.cap
mesh = make_mesh(NDEV)
s2 = NamedSharding(mesh, P("gpus", None)); s1 = NamedSharding(mesh, P("gpus"))
X = jax.device_put(jnp.asarray(part.pos_flat), s2); M = jax.device_put(jnp.asarray(part.mass_flat), s1)
R = jax.device_put(jnp.asarray(part.rank_in), s1); GID = jnp.asarray(part.gid_flat); CNT = jnp.asarray(part.counts)
al = mesh_coupling.make_aligner(mesh)
cfg = DistributedFMMConfig(leaf_size=LEAF, theta=THETA, order=ORDER, softening=soft, G=1.0,
                           nearfield_accum="wide", mac_type="dehnen",
                           m2l_chunk=M2L_CHUNK, nearfield_chunk=NEARFIELD_CHUNK).resolved_for(cap, NDEV)
ev = make_force_evaluator(cfg, NDEV, cap, mesh, jit=True)
pos_rows = np.asarray(jax.device_get(X))[:n].astype(np.float64)
r_rows = np.linalg.norm(pos_rows, axis=1)

def one():
    a_raw, gid_o, diag = ev(X, M, GID, CNT); a = al(a_raw, gid_o, R); jax.block_until_ready(a)
    return np.asarray(a)[:n], np.asarray(a_raw)[:n], np.asarray(gid_o), np.asarray(diag)

As, Araws, Gs = [], [], []
for k in range(3):
    t0 = time.perf_counter(); a, a_raw, g, d = one(); As.append(a); Araws.append(a_raw); Gs.append(g)
    print(f"eval {k}: {time.perf_counter()-t0:.0f} s  non-finite={int(np.count_nonzero(~np.isfinite(a)))}  "
          f"gid_out bitwise-equal-to-eval0={bool(np.array_equal(g, Gs[0]))}", flush=True)
np.save(f"{OUT}/A0.npy", As[0]); np.save(f"{OUT}/A1.npy", As[1]); np.save(f"{OUT}/A2.npy", As[2])
np.save(f"{OUT}/Araw0.npy", Araws[0]); np.save(f"{OUT}/Araw1.npy", Araws[1])

a_rms = float(np.sqrt(np.mean(np.sum(As[0].astype(np.float64)**2, axis=1))))
def bad_rows(A, B, thr=1e-3):
    d = np.linalg.norm(A.astype(np.float64) - B.astype(np.float64), axis=1) / a_rms
    return np.flatnonzero(d > thr), d
bad01, d01 = bad_rows(As[0], As[1]); bad02, d02 = bad_rows(As[0], As[2]); bad12, d12 = bad_rows(As[1], As[2])
print(f"\nper-particle |dA|/a_rms > 1e-3:  eval0-1: {bad01.size:,}   eval0-2: {bad02.size:,}   eval1-2: {bad12.size:,}   of {n:,}")
print(f"  |dA|/a_rms percentiles (0-1): p50={np.percentile(d01,50):.2e} p99={np.percentile(d01,99):.2e} p99.99={np.percentile(d01,99.99):.2e} max={d01.max():.2e}")
same = np.intersect1d(bad01, bad02)
print(f"  rows bad in BOTH 0-1 and 0-2: {same.size:,}  ({100*same.size/max(bad01.size,1):.1f} % of 0-1's bad rows)  "
      f"-> {'FIXED rows: a buffer' if same.size > 0.8*bad01.size else 'MOVING rows: a race'}")

# is the RAW (pre-alignment) output already bad, or does the aligner introduce it?
draw = np.linalg.norm(Araws[0].astype(np.float64) - Araws[1].astype(np.float64), axis=1) / a_rms
print(f"  RAW (pre-aligner) rows with |dA|/a_rms > 1e-3: {int(np.count_nonzero(draw > 1e-3)):,}  "
      f"-> {'defect is UPSTREAM of the aligner' if np.count_nonzero(draw>1e-3) > 0 else 'raw is clean: the ALIGNER is the defect'}")

print("\nWHERE (eval 0 vs 1):")
dev = bad01 // cap; rank = (bad01 % cap) / cap
print(f"  by device: " + "  ".join(f"dev{d}: {int(np.sum(dev==d)):,}" for d in range(NDEV)))
h, _ = np.histogram(rank, bins=10, range=(0, 1))
print(f"  position within device's row range (10 bins, 0=front .. 1=tail): {h.tolist()}")
print(f"    -> {'clustered at the TAIL: unwritten-tail mechanism' if h[-1] > 3*np.median(h[:-1]) else ('clustered at the FRONT' if h[0] > 3*np.median(h[1:]) else 'spread out: not a tail')}")
a0n = np.linalg.norm(As[0].astype(np.float64), axis=1)
print(f"  |A0| of bad rows: median {np.median(a0n[bad01]):.3e} vs all rows {np.median(a0n):.3e}")
print(f"  radius of bad rows: median {np.median(r_rows[bad01]):.3f} (p10 {np.percentile(r_rows[bad01],10):.3f}, p90 {np.percentile(r_rows[bad01],90):.3f}) vs all {np.median(r_rows):.3f}")
big = bad01[np.argsort(-d01[bad01])[:8]]
print(f"  worst 8 rows: " + ", ".join(f"row {i} dev{i//cap} rank {(i%cap)/cap:.3f} r={r_rows[i]:.3f} |A0|={a0n[i]:.2e} |A1|={np.linalg.norm(As[1][i]):.2e}" for i in big))
json.dump(dict(n=n, cap=cap, ndev=NDEV, a_rms=a_rms, nbad01=int(bad01.size), nbad02=int(bad02.size), nbad12=int(bad12.size),
               nsame=int(same.size), raw_bad=int(np.count_nonzero(draw>1e-3)),
               by_device=[int(np.sum(dev==d)) for d in range(NDEV)], rank_hist=h.tolist(),
               bad_rows_01=bad01[:5000].tolist()), open(f"{OUT}/where.json", "w"), indent=1)
print(f"\n# wrote {OUT}/where.json and A0/A1/A2.npy")
