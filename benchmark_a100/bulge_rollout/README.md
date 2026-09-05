# A realistic disc + bulge at 21 million particles

Sessions of 2026-09-01 → 09-05 on `horeka/disc-bulge-21m-quarter-orbit`.

**Done: 25,165,824 particles (20,971,520 disc + 4,194,304 bulge) for a quarter orbit on 8×A100,
489 steps at 82 s/step (11.3 h), with an fp64 direct-sum probe every 25 steps — all twenty in the
t=0 accuracy class, dL/L 4.6e-7.** `findings.md` §15 has the result; `results/qorbit25m/` the logs,
diagnostics, analysis and movies.

**And the reason it took until 09-05:** every multi-GPU rollout on this box before 2026-09-04 computed
~45 % wrong forces from its second step while every invariant looked healthy — XLA's
`ragged_all_to_all` on jax 0.9.0 silently returning its fill value under buffer donation. Root-caused
in pure JAX, fixed upstream (jaccpot #322, yggdrax #69), documented in `findings.md` §14. The
production environment is a standalone jax 0.10.2 venv; `SETUP.md` §0 says how and why.

| file | what it is |
|---|---|
| `SETUP.md` | **Start here to run it.** Exact commands, IC provenance, versions, commits, output formats, and the memory table that decides `ndev`/`leaf` on a new machine. |
| `findings.md` | Why every choice is what it is. The audit of what landed upstream, five defects found by measuring, and the four pretest rounds with their numbers. |
| `HOREKA_PROMPT.md` | A self-contained prompt to paste into a fresh Claude Code session on HoreKa. |
| `horeka_quarter_orbit.sbatch` | SLURM script, one node, site-specific fields marked `CONFIRM`. |
| `results/` | The measurement evidence: per-arm diagnostics JSON and run logs, including both OOMs. |

## The short version

**The galaxy.** 21,012,480 particles: 17,510,400 disc + 3,502,080 *self-gravitating* bulge
(Hernquist, a = 800 pc), equal particle masses, analytic NFW halo. Built with AGAMA through a
self-consistent model. IC at `/export/scratch/tbuck/odisseo_ic/disk_bulge_21m_v2.npz`
(478 MB; **on scratch, so subject to purge** -- copy it before relying on it).

**The chosen configuration**, measured on 6 x A100-40GB:

    leaf 1024, theta 0.7, order 6, fp32, nearfield_accum=wide
    mac_type=dehnen_error, adaptive_eps=1e-5, cross criterion ON
    dt 5e-4, 489 steps = a quarter orbit at the baryonic half-mass radius

    136.3 s/step  ->  18.5 h      force rel_l2 4.03e-03 (fp64 direct sum, probe 256)
    dL/L 2.3e-09 after 6 steps    every overflow flag clear
    peak 40,105 MiB coordinator / ~31,450 MiB per worker

**Why the Dehnen criterion, and what it cost.** Against the geometric MAC at leaf 512 it is
**3.77x more accurate** (rel_l2 1.52e-02 -> 4.03e-03), and the tail moves more than the
median (p99 3.39x against median 1.80x) because eq (16a) equalises *absolute* force error and
so refines hardest in the bulge cusp -- exactly where the geometric MAC is worst. It costs
2.17x per step, because it only fits at leaf 1024 and doubling the leaf doubles the
particle-pair work that the near-field kernel actually pays for. At dt 5e-4 that still
completes a full quarter orbit in 18.5 h.

**The scaling limit worth reporting upstream.** `mac_type="dehnen_error"` cannot run at
leaf 512 at this scale: its derived caps ask for a single 46-64 GiB buffer on a 40 GB card.
Its cap coefficients were calibrated on 2 and 4 devices at N = 1,048,576, and six devices at
twenty times the particles is outside that fit. `--m2l-chunk`/`--nearfield-chunk` do not
reach the offending buffer -- measured byte-identical OOM with chunks 8x and 4x smaller.
