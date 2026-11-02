"""Minimal comparison of losses for recovering the NFW M_vir from a GD1-like stream.

Same set-up as grad_NFW_Potential.ipynb, but:
  * the target is generated at the *actual* Mvir used in `params` (the notebook marks 1e12 as truth
    while the target is made with 4.37e11),
  * common random numbers: the same Plummer key is used for every Mvir, so the loss curve is smooth,
  * observational noise is applied once to the target and folded analytically into the likelihoods,
  * we parameterise with log10(Mvir) and scan loss and d loss / d log10 Mvir.
"""
import os
import shutil
if shutil.which("nvidia-smi"):
    from autocvd import autocvd
    autocvd(num_gpus=1)

import time
import jax
import jax.numpy as jnp
from jax import jit, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy import units as u

from odisseo import construct_initial_state
from odisseo.dynamics import DIRECT_ACC_MATRIX
from odisseo.option_classes import (SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams,
                                    PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL)
from odisseo.initial_condition import Plummer_sphere
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.utils import projection_on_GD1

OUT = os.environ.get("OUT", os.path.dirname(os.path.abspath(__file__)))

code_units = CodeUnits(10 * u.kpc, 1e4 * u.Msun, G=1, unit_time=3 * u.Gyr)

config = SimulationConfig(N_particles=int(os.environ.get("N_PARTICLES", 1000)),
                          return_snapshots=False,
                          num_snapshots=500,
                          num_timesteps=1000,
                          external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL),
                          acceleration_scheme=DIRECT_ACC_MATRIX,
                          softening=(0.1 * u.pc).to(code_units.code_length).value,)

params = SimulationParams(t_end=(3 * u.Gyr).to(code_units.code_time).value,
                          Plummer_params=PlummerParams(Mtot=(10**4.05 * u.Msun).to(code_units.code_mass).value,
                                                       a=(8 * u.pc).to(code_units.code_length).value),
                          MN_params=MNParams(M=(68_193_902_782.346756 * u.Msun).to(code_units.code_mass).value,
                                             a=(3.0 * u.kpc).to(code_units.code_length).value,
                                             b=(0.280 * u.kpc).to(code_units.code_length).value),
                          NFW_params=NFWParams(Mvir=(4.3683325e11 * u.Msun).to(code_units.code_mass).value,
                                               r_s=(16.0 * u.kpc).to(code_units.code_length).value,),
                          PSP_params=PSPParams(M=4501365375.06545 * u.Msun.to(code_units.code_mass),
                                               alpha=1.8,
                                               r_c=(1.9 * u.kpc).to(code_units.code_length).value),
                          G=code_units.G,)

config_com = config._replace(N_particles=1)
params_com = params._replace(t_end=-params.t_end)

pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
vel_com_final = jnp.array([[109.5, -254.5, -90.3]]) * (u.km / u.s).to(code_units.code_velocity)

MSUN = (1 * u.Msun).to(code_units.code_mass).value
LOG_M_TRUE = np.log10(4.3683325e11)

# R [kpc], phi1 [deg], phi2 [deg], vR [km/s], pm1 [mas/yr], pm2 [mas/yr]
noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.05])


def simulate(log_Mvir, key):
    Mvir = 10**log_Mvir * MSUN
    new_params = params._replace(NFW_params=params.NFW_params._replace(Mvir=Mvir))
    new_params_com = params_com._replace(NFW_params=params_com.NFW_params._replace(Mvir=Mvir))

    state_com = time_integration(construct_initial_state(pos_com_final, vel_com_final),
                                 jnp.array([params.Plummer_params.Mtot]), config=config_com, params=new_params_com)
    positions, velocities, mass = Plummer_sphere(key=key, params=new_params, config=config)
    state = construct_initial_state(positions + state_com[:, 0], velocities + state_com[:, 1])
    final_state = time_integration(state, mass, config=config, params=new_params)
    return projection_on_GD1(final_state, code_units=code_units)


# ---- target: fixed IC key, noise applied once ----
key_target, key_noise, key_sim = random.split(random.PRNGKey(42), 3)
stream_target = simulate(LOG_M_TRUE, key_target)
stream_target = stream_target + random.normal(key_noise, stream_target.shape) * noise_std
# fixed, data-derived scale per dimension (no gradient through it)
scale = jnp.std(stream_target, axis=0)


# ---------------------------------------------------------------- losses
def loss_mmd(sim, obs):
    """Baseline: multi-scale RBF MMD in whitened space with *fixed* bandwidths."""
    x, y = sim / scale, obs / scale
    def k(a, b, s):
        d2 = jnp.sum((a[:, None] - b[None]) ** 2, -1)
        return jnp.exp(-d2 / (2 * s**2))
    sig = jnp.array([0.1, 0.3, 1.0, 3.0])
    m = lambda s: k(x, x, s).mean() + k(y, y, s).mean() - 2 * k(x, y, s).mean()
    return jnp.sum(jax.vmap(m)(sig))


def loss_kde(sim, obs):
    """-log L with p(x_obs) = KDE of the simulated particles convolved with the observational errors."""
    # Silverman bandwidth on the simulation, frozen
    h = jax.lax.stop_gradient(1.06 * jnp.std(sim, 0) * sim.shape[0] ** (-1 / 5))
    var = h**2 + noise_std**2
    d2 = jnp.sum((obs[:, None] - sim[None]) ** 2 / var, -1)            # (N_obs, N_sim)
    logp = jax.nn.logsumexp(-0.5 * d2, axis=1) - jnp.log(sim.shape[0]) - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * var))
    return -jnp.mean(logp)


def loss_track(sim, obs, h_phi1=2.0):
    """Stream-track likelihood: p(phi1) * prod_d N(x_d | mu_d(phi1), s_d(phi1)^2 + err_d^2).

    mu_d, s_d are Gaussian-kernel weighted means/stds of the simulated particles around each observed phi1,
    i.e. a smooth (differentiable) version of binning in phi1.
    """
    # log-space throughout: w / sum(w) is inf*0 = NaN in reverse mode when an observed star has no particle nearby
    logw = -0.5 * ((obs[:, 1:2] - sim[None, :, 1]) / h_phi1) ** 2      # (N_obs, N_sim)
    # density along the stream: normalised 1D KDE in phi1
    log_p_phi1 = jax.nn.logsumexp(logw, 1) - jnp.log(sim.shape[0] * jnp.sqrt(2 * jnp.pi) * h_phi1)
    wn = jax.nn.softmax(logw, 1)
    other = jnp.array([0, 2, 3, 4, 5])
    mu = wn @ sim[:, other]
    s2 = wn @ sim[:, other] ** 2 - mu**2
    var = jnp.clip(s2, 1e-6) + noise_std[other] ** 2
    log_p_rest = -0.5 * jnp.sum((obs[:, other] - mu) ** 2 / var + jnp.log(2 * jnp.pi * var), 1)
    return -jnp.mean(log_p_phi1 + log_p_rest)


LOSSES = {"MMD (fixed bw, whitened)": loss_mmd, "KDE log-likelihood": loss_kde, "Track log-likelihood": loss_track}


def make_vg(loss_fn):
    return jit(jax.value_and_grad(lambda lm: loss_fn(simulate(lm, key_sim), stream_target)))


if __name__ == "__main__":
    log_M = np.linspace(LOG_M_TRUE - 0.3, LOG_M_TRUE + 0.3, int(os.environ.get("N_SCAN", 25)))
    res = {}
    for name, fn in LOSSES.items():
        vg = make_vg(fn)
        t0 = time.time()
        vals = [vg(jnp.asarray(lm)) for lm in log_M]
        res[name] = np.array([[float(v), float(g)] for v, g in vals])
        i = np.argmin(res[name][:, 0])
        print(f"{name:28s} argmin log10 M = {log_M[i]:.3f} (true {LOG_M_TRUE:.3f})  "
              f"[{time.time() - t0:.0f}s]", flush=True)
        # finite-difference check of the AD gradient at a point off the truth
        lm0, eps = LOG_M_TRUE - 0.15, 0.02  # large-scale slope, not the local ripple
        fd = (float(vg(jnp.asarray(lm0 + eps))[0]) - float(vg(jnp.asarray(lm0 - eps))[0])) / (2 * eps)
        print(f"{'':28s} grad @ {lm0:.3f}: AD {float(vg(jnp.asarray(lm0))[1]):.4g}  FD {fd:.4g}", flush=True)

    np.savez(os.path.join(OUT, "nfw_mvir_losses.npz"), log_M=log_M, **{k: v for k, v in res.items()})
    fig, axs = plt.subplots(2, len(LOSSES), figsize=(5 * len(LOSSES), 8), sharex=True)
    for j, (name, r) in enumerate(res.items()):
        axs[0, j].plot(log_M, r[:, 0], "o-")
        axs[0, j].set_title(name)
        axs[1, j].plot(log_M, r[:, 1], "o-")
        axs[1, j].axhline(0, c="k", lw=0.5)
        for a in axs[:, j]:
            a.axvline(LOG_M_TRUE, c="r", ls="--")
        axs[1, j].set_xlabel(r"$\log_{10} M_{vir}$")
    axs[0, 0].set_ylabel("loss")
    axs[1, 0].set_ylabel(r"$\partial$ loss / $\partial \log_{10} M_{vir}$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nfw_mvir_losses.png"), dpi=100)
