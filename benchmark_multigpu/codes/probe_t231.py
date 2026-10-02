#!/usr/bin/env python
"""T2.3.1 -- decouple MAC granularity from the P2P tile: how much direct work would
a leaf-64 MAC with a leaf-256 kernel tile do?

Uses the exact neighbour lists of the leaf-64 and leaf-256 static-radix trees at
the same theta.  The leaf-64 leaves are contiguous Morton ranges of the sorted
particles, so four consecutive ones form a 256-particle tile; a tile's direct
sources are the UNION of its four children's neighbour sets (each source is a
64-leaf).  Reports direct share of N for: leaf 256 (today), leaf 64 (MAC and
tile at 64), and the hybrid (MAC at 64, tile 256).  No timing.
"""
import os, sys, json
from pathlib import Path
HERE = Path(os.path.dirname(os.path.abspath(__file__))); ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np
from common.gpu_guard import pick_idle_gpus, set_cuda_visible
from compare_force import apply_fast_lane_env
from common.budget import jaccpot_direct_budget
from common.ic import IC_GENERATORS
N = int(os.environ.get("PROBE_N", "200000")); ORDER = 4
THETAS = [float(t) for t in os.environ.get("PROBE_THETAS", "0.6 0.8").split()]
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
devices = [int(os.environ["BENCH_FORCE_GPU"])] if os.environ.get("BENCH_FORCE_GPU") else pick_idle_gpus(1)
set_cuda_visible(devices); apply_fast_lane_env(N)
import jax, jax.numpy as jnp
from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig, TreeConfig
pos, mass = IC_GENERATORS["plummer"](N, seed=0)
M = jnp.asarray(mass, jnp.float32); P = jnp.asarray(pos, jnp.float32)

def prepared(leaf, theta):
    s = FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=theta, G=1.0,
        softening=1e-7, working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=leaf),
            farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
        fixed_order=ORDER)
    p, ev = s.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=leaf, max_order=ORDER, theta=theta)
    return p

def rows_and_occ(p):
    nl = p.neighbor_list
    counts = np.asarray(nl.counts).astype(np.int64); offsets = np.asarray(nl.offsets).astype(np.int64)
    nbrs = np.asarray(nl.neighbors).astype(np.int64); leaf_nodes = np.asarray(nl.leaf_indices).astype(np.int64)
    node_to_leaf = np.full(int(leaf_nodes.max()) + 1, -1, np.int64); node_to_leaf[leaf_nodes] = np.arange(len(leaf_nodes))
    rows = [node_to_leaf[nbrs[offsets[l]:offsets[l] + counts[l]]] for l in range(len(counts))]
    mask = np.asarray(p.nearfield_leaf_particle_mask); occ = mask.reshape(mask.shape[0], -1).sum(1).astype(np.int64)
    # leaf order along the sorted particle axis: node_ranges start
    starts = np.asarray(p.tree.node_ranges)[leaf_nodes, 0]
    return rows, occ, np.argsort(starts)

out = {}
for theta in THETAS:
    p256 = prepared(256, theta); b256 = jaccpot_direct_budget(p256, N)
    p64 = prepared(64, theta); b64 = jaccpot_direct_budget(p64, N)
    rows64, occ64, order64 = rows_and_occ(p64)
    # tiles: 4 consecutive leaf-64 leaves along the Morton axis
    tiles = [order64[i:i + 4] for i in range(0, len(order64), 4)]
    direct_hybrid = []; wts = []
    for t in tiles:
        union = set()
        for l in t: union.update(rows64[l].tolist())
        union.difference_update(t.tolist())        # tile-mates are the self term
        src = int(occ64[list(union)].sum()) + int(occ64[t].sum()) - 1 if union else int(occ64[t].sum()) - 1
        direct_hybrid.append(src); wts.append(int(occ64[t].sum()))
    direct_hybrid = np.asarray(direct_hybrid, float); wts = np.asarray(wts, float)
    hyb_mean = float((direct_hybrid * wts).sum() / wts.sum())
    print(f"theta {theta}: direct share of N per target -- leaf 256: {b256['direct_share_of_N']:.3f}   "
          f"leaf 64: {b64['direct_share_of_N']:.3f}   hybrid (MAC@64, tile 256): {hyb_mean/N:.3f}   "
          f"[p2p pair-evals: 256 {b256['p2p_pair_evaluations']:.2e}, 64 {b64['p2p_pair_evaluations']:.2e}, "
          f"hybrid {float((direct_hybrid*wts).sum()):.2e}]", flush=True)
    out[theta] = dict(leaf256=b256, leaf64=b64, hybrid_share=hyb_mean / N, hybrid_pairs=float((direct_hybrid * wts).sum()))
Path(ROOT / "artifacts" / f"probe_t231_plummer{N}.json").write_text(json.dumps(out, indent=2))
