# The fused multi-GPU lane on HoreKa

Why: the development box (8x A100-PCIe-40GB, no NVLink, shared, no scheduler) cannot give
reliable numbers beyond two cards. HoreKa's accelerated nodes give exclusive 4-GPU nodes.
First runs use ONE process per node driving its 4 GPUs, which today's code supports. One
process per GPU (`jax.distributed.initialize`) and multi-node come later.

Nothing has run on HoreKa yet. Every line marked CONFIRM in the sbatch script is a guess.

## 1. Environment (once)

The recipe that produced the record on the development box (Odisseo
`benchmark_a100/bulge_rollout/SETUP.md` section 0), adapted:

```bash
python3.12 -m venv $HOME/venvs/jaccpot-fused        # standalone: NEVER --system-site-packages
source $HOME/venvs/jaccpot-fused/bin/activate
pip install "jax[cuda12]==0.10.2" numpy scipy jaxtyping beartype matplotlib
git clone git@github.com:TobiBu/yggdrax.git $HOME/src/yggdrax   # then: git checkout <PR head>
git clone git@github.com:TobiBu/jaccpot.git $HOME/src/jaccpot   # then: git checkout <PR head>
git clone git@github.com:vepe99/Odisseo.git $HOME/src/Odisseo   # bench/multi-gpu-harness
pip install --no-deps -e $HOME/src/yggdrax -e $HOME/src/jaccpot
python -c "import jax_plugins.xla_cuda12 as p; print(p.__file__)"   # must be inside the venv
```

* jax must be >= 0.10.2: below 0.9.1 the native `ragged_all_to_all` returns its fill value
  once buffers are donated, which silently drops the cross field (memory
  `ragged-all-to-all-forward-corruption`).
* A venv built with `--system-site-packages` can load an old CUDA plugin while `pip list`
  shows the new one. The `__file__` check above is the test.
* Pin the clones to fixed commits. An editable install follows whatever branch its checkout
  is on.

## 2. Submit

```bash
sbatch --export=ALL,JACCPOT_DIR=$HOME/src/jaccpot,BENCH_DIR=$HOME/src/Odisseo/benchmark_multigpu \
    $HOME/src/Odisseo/benchmark_multigpu/slurm/horeka_fused_sweep.sbatch
```

The job validates first (`fused_sweep.sh`): the 1-card force at N = 2e5, then the 2- and
4-card accuracy gates at p4/5/6, checked by `check_validation.py` (no capacity flag,
error falling with order, p6 within 1.25x of the 1-card error). It stops before the sweep
if any check fails. Four cards have never been measured anywhere, so that is the first
real result.

## 3. Rules that carry over

* Compare rows only within one machine. HoreKa's interconnect differs from the
  development box, so ndev 1 and 2 are re-measured there as its own baseline.
* Every number is a full force per call (tree, walk, exchange, evaluation), min over
  repeats after warm-up. The JSON rows carry the capacities and flags. A row whose flag
  fired is not a result.
* Keep the persistent compile cache (`JAX_COMPILATION_CACHE_DIR`). The first force
  compiles for 30-90 s.
