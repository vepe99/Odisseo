"""Correctness battery for the distributed FMM at the production configuration.

Three questions, in one process so there is only one compile:

1. **Is the force DETERMINISTIC?** Two evaluations of the same input, on the same devices,
   with the same compiled program, must be bit-identical. XLA is deterministic for a fixed
   program and shapes, so a bitwise difference means something non-deterministic is in the
   loop -- atomics, or a read of memory that was never written. This is the sharpest available
   test of the uninitialised-padding hypothesis for the intermittent NaN.

2. **Is it STABLE over repeats?** Ten evaluations, reporting the max pairwise deviation, so a
   rare divergence has a chance to show up rather than relying on a single pair.

3. **How accurate is it, on a tighter sample?** fp64 direct sum over ALL 17.8 M sources for
   512 targets (double the production probe), with percentiles, so the tail is measured rather
   than inferred from 256 draws.

Nothing here integrates. It is a force-level audit.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, "/export/home/tbuck/Odisseo")
IC = "/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz"
NDEV, LEAF, THETA, ORDER, EPS = 8, 1024, 0.7, 6, 1e-5
# These MUST match tools/mesh_galaxy_run.py's own defaults. jaccpot's defaults for both are
# None, i.e. UNCHUNKED, and the rollout script overrides them -- omitting them here asked for
# a single 292.86 GiB allocation on a 40 GB card, because the far-field M2L materialises in
# full. An audit of a configuration has to be the SAME configuration.
M2L_CHUNK, NEARFIELD_CHUNK = 65536, 512
NPROBE = 512
NREPEAT = 10

import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
print(f"devices: {[d.id for d in jax.devices()]}", flush=True)

from odisseo import mesh_coupling
from jaccpot.distributed.fmm import (
    DistributedFMMConfig, make_force_evaluator, DIAG_FIELDS,
)
from yggdrax.distributed import make_mesh

ic = np.load(IC)
state0 = np.asarray(ic["state0"]); mass = np.asarray(ic["mass"])
pos = np.ascontiguousarray(state0[:, 0, :], dtype=np.float32)
n = len(mass)
soft = float(0.5 * float(ic["rdisk_code"]) / np.sqrt(n / 1e5))
print(f"N={n:,}  softening={soft:.7f}", flush=True)

part = mesh_coupling.build_mesh_partition(pos, mass, ndev=NDEV, leaf_size=LEAF,
                                          partitioner="rcb")
mesh = make_mesh(NDEV)
cfg = DistributedFMMConfig(
    leaf_size=LEAF, theta=THETA, order=ORDER, softening=soft, G=1.0,
    nearfield_accum="wide", mac_type="dehnen_error", adaptive_eps=EPS,
    m2l_chunk=M2L_CHUNK, nearfield_chunk=NEARFIELD_CHUNK,
).resolved_for(part.cap, NDEV)
print(f"caps: self_q={cfg.max_pair_queue:,} self_int={cfg.max_interactions_per_node:,} "
      f"cross_q={cfg.cross_max_pair_queue:,} cross_int={cfg.cross_max_interactions_per_node:,}",
      flush=True)

# Cross-check against the arguments the production rollout recorded, so a divergence
# between "the config we audit" and "the config we ran" fails here instead of producing a
# reassuring number about something else.
_ref = "/export/scratch/tbuck/odisseo_runs/nandiag/nd_diag.json"
try:
    import json as _json
    _a = _json.load(open(_ref))["args"]
except Exception as _e:
    print(f"# (no reference args to cross-check against: {_e})", flush=True)
else:
    _want = {"leaf": LEAF, "theta": THETA, "order": ORDER, "ndev": NDEV,
             "adaptive_eps": EPS, "mac_type": "dehnen_error",
             "nearfield_accum": "wide", "m2l_chunk": M2L_CHUNK,
             "nearfield_chunk": NEARFIELD_CHUNK, "dtype": "float32"}
    _bad = {k: (_a.get(k), v) for k, v in _want.items() if _a.get(k) != v}
    if _bad:
        raise SystemExit(f"audit config differs from the run's: {_bad}")
    print(f"# config cross-checked against {_ref}: identical on "
          f"{', '.join(sorted(_want))}", flush=True)

ev = make_force_evaluator(cfg, NDEV, part.cap, mesh, jit=True)
al = mesh_coupling.make_aligner(mesh)
s2 = NamedSharding(mesh, P("gpus", None)); s1 = NamedSharding(mesh, P("gpus"))
X = jax.device_put(jnp.asarray(part.pos_flat), s2)
M = jax.device_put(jnp.asarray(part.mass_flat), s1)
R = jax.device_put(jnp.asarray(part.rank_in), s1)
GID = jnp.asarray(part.gid_flat); CNT = jnp.asarray(part.counts)

def one():
    a_raw, gid_o, diag = ev(X, M, GID, CNT)
    a = al(a_raw, gid_o, R)
    jax.block_until_ready(a)
    return np.asarray(a), np.asarray(diag)

t0 = time.perf_counter()
A0, D0 = one()
print(f"# first force (compile incl.) {time.perf_counter()-t0:.1f} s", flush=True)
dec0 = {k: [float(v) for v in D0[:, i]] for i, k in enumerate(DIAG_FIELDS) if i < D0.shape[1]}
print("# overflow:", {k: sum(v) for k, v in dec0.items() if k.endswith("overflow")}, flush=True)
print(f"# self_near={sum(dec0.get('self_near_pairs',[0])):,.0f} "
      f"cross_near={sum(dec0.get('cross_near_pairs',[0])):,.0f} "
      f"fs=[{min(dec0.get('force_scale_min',[0])):.6g},{max(dec0.get('force_scale_max',[0])):.6g}]",
      flush=True)

print(f"\n=== 1+2. DETERMINISM over {NREPEAT} evaluations of identical input ===", flush=True)
worst_bits = 0; worst_rel = 0.0; first_diff_at = None
pair_l2 = []; pair_p999 = []; pair_pmax = []; A_last = None
pair_counts_stable = True
for k in range(1, NREPEAT):
    Ak, Dk = one()
    nbits = int(np.count_nonzero(Ak.view(np.uint32) != A0.view(np.uint32)))
    # THE statistic. `max|dA|/max(|A0|,tiny)` -- which this script used to report -- is
    # meaningless here: a particle in the bulge centre has a NET acceleration near zero
    # because contributions cancel, so any absolute difference divides by ~0 and the max
    # blows up to O(1) regardless of how small the perturbation is. It reported 8.5 and
    # that is not an 850 % error.
    #
    # rel_l2 = ||Ak - A0||_2 / ||A0||_2 is the aggregate deviation and has no such
    # denominator pathology. Alongside it: the per-particle deviation normalised by the
    # RMS acceleration (a fixed, well-conditioned scale), reported as percentiles.
    dA = Ak - A0
    rel_l2_pair = float(np.linalg.norm(dA) / np.linalg.norm(A0))
    a_rms = float(np.sqrt(np.mean(np.sum(A0 * A0, axis=1))))
    per_particle = np.linalg.norm(dA, axis=1) / a_rms
    rel = rel_l2_pair
    if nbits and first_diff_at is None:
        first_diff_at = k
    worst_bits = max(worst_bits, nbits); worst_rel = max(worst_rel, rel)
    pair_l2.append(rel_l2_pair)
    pair_p999.append(float(np.quantile(per_particle, 0.999)))
    pair_pmax.append(float(per_particle.max()))
    deck = {kk: [float(v) for v in Dk[:, i]] for i, kk in enumerate(DIAG_FIELDS) if i < Dk.shape[1]}
    for f in ("self_near_pairs", "cross_near_pairs", "self_far_pairs", "cross_far_pairs"):
        if deck.get(f) != dec0.get(f):
            pair_counts_stable = False
            print(f"   eval {k}: {f} CHANGED {dec0.get(f)} -> {deck.get(f)}", flush=True)
    nonfinite = int(np.count_nonzero(~np.isfinite(Ak)))
    A_last = Ak
    print(f"   eval {k}: differing words = {nbits:,}/{A0.size:,}"
          f"   rel_l2 vs eval0 = {rel_l2_pair:.3e}"
          f"   per-particle/a_rms p99.9 = {pair_p999[-1]:.3e} max = {pair_pmax[-1]:.3e}"
          f"   non-finite = {nonfinite}", flush=True)

print(f"\n   VERDICT: {'BIT-IDENTICAL across all repeats' if worst_bits==0 else 'NON-DETERMINISTIC'}")
if worst_bits:
    print(f"   worst pairwise rel_l2 vs eval 0 : {max(pair_l2):.3e}   (median {np.median(pair_l2):.3e})")
    print(f"   worst per-particle/a_rms p99.9  : {max(pair_p999):.3e}")
    print(f"   worst per-particle/a_rms max    : {max(pair_pmax):.3e}")
    print(f"   -> compare against the lane's OWN error vs fp64 truth, reported below.")
    print(f"      Non-determinism matters only if it is COMPARABLE to that.")
print(f"   pair counts {'stable' if pair_counts_stable else 'UNSTABLE'} across repeats")

print(f"\n=== 3. ACCURACY vs an fp64 direct sum over all {n:,} sources, {NPROBE} targets ===",
      flush=True)
from tools.mesh_galaxy_run import direct_sum_probe  # noqa: E402
pos_rows = np.asarray(jax.device_get(X))[:n]
mass_rows = np.asarray(jax.device_get(M))[:n]
rng = np.random.default_rng(20260902)
targets = np.sort(rng.choice(n, size=NPROBE, replace=False))
t0 = time.perf_counter()
a_ref = direct_sum_probe(pos_rows, mass_rows, targets, soft, 1.0)
a_got = A0[:n][targets].astype(np.float64)
num = np.linalg.norm(a_got - a_ref, axis=1); den = np.linalg.norm(a_ref, axis=1)
rel = num / np.maximum(den, 1e-300)
# The decisive question: is a DIFFERENT (non-deterministic) evaluation just as accurate?
# If eval 0 and eval 9 both sit at the same rel_l2 against the same fp64 reference, the
# non-determinism does not degrade the physics -- it just reshuffles round-off.
rel_l2_last = None
if A_last is not None:
    a_got2 = A_last[:n][targets].astype(np.float64)
    rel_l2_last = float(np.linalg.norm(np.linalg.norm(a_got2 - a_ref, axis=1))
                        / np.linalg.norm(den))
out = {
    "rel_l2": float(np.linalg.norm(num) / np.linalg.norm(den)),
    "median": float(np.median(rel)), "p90": float(np.quantile(rel, .90)),
    "p99": float(np.quantile(rel, .99)), "max": float(rel.max()),
    "probe": NPROBE, "seconds": round(time.perf_counter()-t0, 1),
    "determinism_worst_differing_words": worst_bits,
    "determinism_worst_pairwise_rel_l2": (max(pair_l2) if pair_l2 else 0.0),
    "determinism_median_pairwise_rel_l2": (float(np.median(pair_l2)) if pair_l2 else 0.0),
    "determinism_worst_perparticle_p999": (max(pair_p999) if pair_p999 else 0.0),
    "pair_counts_stable": pair_counts_stable,
    "rel_l2_last_eval": rel_l2_last,
    "ndev": NDEV, "leaf": LEAF, "N": n,
}
if rel_l2_last is not None:
    print(f"   accuracy of eval 0    vs fp64: rel_l2 = {out['rel_l2']:.6e}")
    print(f"   accuracy of eval {NREPEAT-1} vs fp64: rel_l2 = {rel_l2_last:.6e}")
    print(f"   -> the two differ by {abs(rel_l2_last-out['rel_l2'])/out['rel_l2']:.2e} relative;"
          f" the non-determinism {'does NOT' if abs(rel_l2_last-out['rel_l2'])/out['rel_l2'] < 0.05 else 'DOES'}"
          f" change the achieved accuracy")
print("   " + "  ".join(f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in out.items()), flush=True)
json.dump(out, open("/export/scratch/tbuck/odisseo_runs/trust_battery.json", "w"), indent=1)
print("\n# wrote /export/scratch/tbuck/odisseo_runs/trust_battery.json")
