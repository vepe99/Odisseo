Benchmarks
==========

# GD1 stream: Odisseo vs gala and galpy

To compare `Odisseo` with established CPU codes, we evolve the same direct N-body problem with [`gala`](https://gala.adrian.pw) and [`galpy`](https://docs.galpy.org) for increasing particle number $N$, and record both the time to solution and the conservation of energy and angular momentum.

## ⚙️ Setup

The problem is the GD1 stream of `notebooks/GD1.ipynb`, identical for every code:

- **Progenitor:** Plummer sphere ($M = 10^{4.05}\,M_\odot$, $a = 100$ pc) on the GD1 orbit, integrated back 3 Gyr and then evolved forward 3 Gyr.
- **External potential:** NFW + Miyamoto–Nagai + power law with cutoff (MWPotential2014). The three codes' accelerations agree to $< 10^{-13}$.
- **Self-gravity:** direct summation, Plummer softening $\epsilon = 0.1$ pc.
- **Integration:** fp64, second-order leapfrog, 1000 fixed steps ($dt = 3$ Myr), same initial conditions for every code. `Odisseo` on the GPU is also run in JAX's native fp32 (`jax_enable_x64` off).

| | Odisseo | gala | galpy |
|---|---|---|---|
| N-body engine | `time_integration`, `DIRECT_ACC` (JAX) | `DirectNBody` (Cython) | `numpy` direct sum |
| Hardware | 1 A100 GPU (fp64 and fp32), or 1 CPU core | 1 CPU core | 1 CPU core |

## ⏱️ Timing

<img src="./notebooks/benchmarks/stream/results/stream_timing.png" alt="Wall-clock time vs N" width="100%">

On one A100, `Odisseo` in fp64 is 35× faster than `gala` at $N = 500$ and about 800× faster at $N = 8000$. In fp32 it is another 1.9× (at $N = 500$) to 3× (from $N = 8000$) faster, about 2500× faster than `gala` at $N = 8000$. 

## ⚖️ Energy and angular momentum conservation

The external potential is static and axisymmetric, so the total energy $E$ and the $z$-component of the angular momentum $L_z$ are conserved:

$$
\delta E = \frac{|E(t) - E(0)|}{|E(0)|}, \quad
\delta L_z = \frac{|L_z(t) - L_z(0)|}{|\vec{L}(0)|}
$$

Both are computed from snapshots every 10 steps, using the same estimator for all codes.

<img src="./notebooks/benchmarks/stream/results/conservation/conservation.png" alt="Energy and angular momentum conservation" width="100%">

`Odisseo` (GPU and CPU) and `galpy` integrate the same leapfrog map: their curves agree to $\lesssim 3\times10^{-8}$. The energy error does not drift; it peaks at $\sim 2\times10^{-4}$ at each pericentre. $L_z$ is conserved to machine precision.

## 📂 Code

Everything is in [`notebooks/benchmarks/stream`](https://github.com/vepe99/Odisseo/tree/main/notebooks/benchmarks/stream):

- [`stream_benchmark.py`](https://github.com/vepe99/Odisseo/blob/main/notebooks/benchmarks/stream/stream_benchmark.py): initial conditions, timed runs, timing plot.
- [`stream_conservation.py`](https://github.com/vepe99/Odisseo/blob/main/notebooks/benchmarks/stream/stream_conservation.py): snapshot runs and the conservation plot.
- [`README.md`](https://github.com/vepe99/Odisseo/blob/main/notebooks/benchmarks/stream/README.md): the full description of the setup, the run conditions and the reproduction commands.
