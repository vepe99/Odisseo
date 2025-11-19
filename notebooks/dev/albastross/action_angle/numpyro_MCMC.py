#!/usr/bin/env python3
"""
numpyro_MCMC.py

Run NUTS (NumPyro) inference on the mass and scale radius of the NFW halo
using your MockStreamGenerator simulator and the stream likelihood.

Usage:
    python numpyro_MCMC.py --num-warmup 500 --num-samples 1000 --num-chains 2

Edit priors and defaults below as needed.
"""
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from autocvd import autocvd
autocvd(num_gpus = 1)
import argparse
import time
import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, AIES

# Your domain packages: expect these to be importable in your env.
# If any import fails, fix your environment (these come from your codebase).
try:
    import galax.potential as gp
    import galax.dynamics as gd  # if present; your snippet used `gd` for ChenStreamDF & MockStreamGenerator
    import galax.dynamics as gd  # fallback alias
    import galax.coordinates as gc
except Exception as e:
    print("Error importing galax modules:", e)
    print("Make sure galax (gp, gd, gc) is installed and on PYTHONPATH.")
    raise

# Quantity may come from 'unxt' in your environment earlier; fallback to astropy.units.Quantity
try:
    from unxt import Quantity
except Exception:
    from astropy.units import Quantity

# --------------------------
# ---- coordinate helpers ---
# --------------------------
# (kept from your snippet, minimally modified)
@jax.jit
def halo_to_sun(Xhalo: jnp.ndarray) -> jnp.ndarray:
    sunx = 8.0
    xsun = sunx - Xhalo[0]
    ysun = Xhalo[1]
    zsun = Xhalo[2]
    return jnp.array([xsun, ysun, zsun])

@jax.jit
def sun_to_gal(Xsun: jnp.ndarray) -> jnp.ndarray:
    r = jnp.linalg.norm(Xsun)
    b = jnp.arcsin(Xsun[2] / r)
    l = jnp.arctan2(Xsun[1], Xsun[0])
    return jnp.array([r, b, l])

@jax.jit
def gal_to_equat(Xgal: jnp.ndarray) -> jnp.ndarray:
    dNGPdeg = 27.12825118085622
    lNGPdeg = 122.9319185680026
    aNGPdeg = 192.85948
    dNGP = dNGPdeg * jnp.pi / 180.0
    lNGP = lNGPdeg * jnp.pi / 180.0
    aNGP = aNGPdeg * jnp.pi / 180.0
    r = Xgal[0]
    b = Xgal[1]
    l = Xgal[2]
    sb = jnp.sin(b)
    cb = jnp.cos(b)
    sl = jnp.sin(lNGP - l)
    cl = jnp.cos(lNGP - l)
    cs = cb * sl
    cc = jnp.cos(dNGP) * sb - jnp.sin(dNGP) * cb * cl
    alpha = jnp.arctan(cs / cc) + aNGP
    delta = jnp.arcsin(jnp.sin(dNGP) * sb + jnp.cos(dNGP) * cb * cl)
    return jnp.array([r, alpha, delta])

def transform_velocity(transform_fn, X, V):
    J = jax.jacobian(transform_fn)(X)  # (3,3)
    return J @ V

def halo_to_equatorial(Xhalo):
    Xsun = halo_to_sun(Xhalo)
    Xgal = sun_to_gal(Xsun)
    Xeq  = gal_to_equat(Xgal)
    return Xeq

# batched helpers
halo_to_equatorial_batch = jax.vmap(halo_to_equatorial, in_axes=(0))
transform_velocity_batch = jax.vmap(transform_velocity, in_axes=(None, 0, 0))

def stream_to_array(stream):
    # expects stream.q.x etc. are astropy/unxt Quantity-like objects
    pos = jnp.array([stream.q.x.to('kpc').value, stream.q.y.to('kpc').value, stream.q.z.to('kpc').value])
    vel = jnp.array([stream.p.x.to('kpc/Myr').value, stream.p.y.to('kpc/Myr').value, stream.p.z.to('kpc/Myr').value])
    return jnp.concatenate([pos, vel], axis=0).T

# --------------------------
# ---- likelihood pieces ----
# --------------------------
def log_diag_multivariate_normal(x, mean, sigma_eff):
    diff = (x - mean) / sigma_eff
    D = x.shape[0]
    log_det = 2.0 * jnp.sum(jnp.log(sigma_eff))
    norm_const = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det)
    exponent = -0.5 * jnp.sum(diff**2)
    return norm_const + exponent

def stream_likelihood_diag(model_stream, obs_stream, obs_errors, smooth_sigma):
    sigma_eff = jnp.sqrt(obs_errors**2 + smooth_sigma**2)
    def obs_log_prob(obs):
        def model_log_prob(model_point):
            return log_diag_multivariate_normal(obs, model_point, sigma_eff)
        log_probs = jax.vmap(model_log_prob)(model_stream)
        log_p_stream = jax.scipy.special.logsumexp(log_probs) - jnp.log(model_stream.shape[0])
        return log_p_stream
    logL_values = jax.vmap(obs_log_prob)(obs_stream)
    return jnp.sum(logL_values)

@jax.jit
def log_multivariate_normal(x, mean, cov):
    D = x.shape[0]
    L = jnp.linalg.cholesky(cov)
    diff = x - mean
    solve = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    mahal = jnp.sum(solve**2)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    norm_const = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det)
    return norm_const - 0.5 * mahal

@jax.jit
def stream_likelihood_fullcov(model_stream, obs_stream, obs_errors, smooth_sigma):
    cov = smooth_sigma
    def obs_log_prob(obs):
        def model_log_prob(model_point):
            return log_multivariate_normal(obs, model_point, cov)
        log_probs = jax.vmap(model_log_prob)(model_stream)
        return jax.scipy.special.logsumexp(log_probs) - jnp.log(model_stream.shape[0])
    logL_values = jax.vmap(obs_log_prob)(obs_stream)
    return jnp.sum(logL_values)

# --------------------------
# ---- loss simulator ------
# --------------------------
@jax.jit
def loss(m_tot, r_s, key_sim=None):
    """
    Return log-likelihood for a given m_tot and r_s.
    This rebuilds the CompositePotential with the requested NFW parameters,
    generates a mock stream, transforms it and computes the stream likelihood
    against the precomputed stream_target (defined in main).
    """
    # build potential with given m and r_s
    pot_new = gp.CompositePotential(
            disk=gp.BovyMWPotential2014.disk,
            halo=gp.NFWPotential(m=m_tot, r_s=r_s, units="galactic"),
            bulge=gp.BovyMWPotential2014.bulge,
    )

    # re-create initial conditions used for the stream generator
    # Note: these are the values you used in your original snippet; keep consistent.
    w = gc.PhaseSpacePosition(q=Quantity([11.8, 0.79, 6.4], "kpc"),
                                p=Quantity([109.5,-254.5,-90.3], "km/s"),
                            )
    t_array = Quantity(-jnp.linspace(0, 3000, 1_000), "Myr")

    df = gd.ChenStreamDF()
    gen = gd.MockStreamGenerator(df, pot_new)

    # deterministic key: simulations should be deterministic wrt (m_tot, r_s)
    if key_sim is None:
        key_sim = jr.PRNGKey(0)

    stream_c25_new, prog_c25_new = gen.run(key_sim, t_array, w, prog_mass)

    stream = stream_to_array(stream_c25_new)
    stream_pos = stream[:, :3]
    stream_vel = stream[:, 3:]
    stream_eq_pos = halo_to_equatorial_batch(stream_pos)
    stream_eq_vel = transform_velocity_batch(halo_to_equatorial, stream_pos, stream_vel)
    stream = jnp.concatenate([stream_eq_pos, stream_eq_vel], axis=1)

    # no observational errors assumed here (you may change)
    noise_std = jnp.zeros(6)

    # construct model covariance (full-cov) from model stream
    stream_cov = 0.05 * jnp.cov(stream_target.T)

    # compute log-likelihood
    ll = stream_likelihood_fullcov(model_stream=stream,
                                   obs_stream=stream_target,
                                   obs_errors=noise_std,
                                   smooth_sigma=stream_cov)
    return ll

# --------------------------
# ---- NumPyro model -------
# --------------------------

def stream_model():
    """
    NumPyro model that infers log_m and log_r_s (log of NFW mass and scale radius).
    Both priors are uniform in log-space by default; change as you like.
    """
    
    base_m = gp.BovyMWPotential2014.halo.m.value.value
    base_r_s = gp.BovyMWPotential2014.halo.r_s.value.value

    # Priors in log-space (you can widen/narrow these as needed)
    log_m = numpyro.sample("log_m", dist.Uniform(jnp.log(base_m * 0.5), jnp.log(base_m * 1.5)))
    log_r_s = numpyro.sample("log_r_s", dist.Uniform(jnp.log(base_r_s * 0.5), jnp.log(base_r_s * 1.5)))

    m_tot = jnp.exp(log_m)
    r_s = jnp.exp(log_r_s)

    # expose deterministic values in trace for easier interpretation
    numpyro.deterministic("m_tot", m_tot)
    numpyro.deterministic("r_s", r_s)

    # Optionally: if your simulator uses randomness, you can sample a key and pass it into loss.
    # Here we use a fixed key for deterministic simulator behaviour.
    # key_sim = jr.PRNGKey(0)

    # Compute the log-likelihood (simulator must return a scalar logpdf)
    loglike = loss(m_tot, r_s, key_sim=None)

    # Insert into probabilistic model
    numpyro.factor("stream_loglike", loglike)

def stream_model_reparam():
    """
    NumPyro model that infers log_m and log_r_s (log of NFW mass and scale radius).
    Both priors are uniform in log-space by default; change as you like.
    """
    
    base_m = gp.BovyMWPotential2014.halo.m.value.value
    base_r_s = gp.BovyMWPotential2014.halo.r_s.value.value

    # hyperparams (center + scale) — set sensible values
    mu_logm = jnp.log(base_m)
    sigma_logm = 0.01  # tune: ~0.2-0.5

    mu_logrs = jnp.log(base_r_s)
    sigma_logrs = 0.01

    # Non-centered sampling
    z_m = numpyro.sample("z_m", dist.Normal(0., 1.))
    z_rs = numpyro.sample("z_rs", dist.Normal(0., 1.))

    log_m = mu_logm + sigma_logm * z_m
    log_r_s = mu_logrs + sigma_logrs * z_rs

    m_tot = jnp.exp(log_m)
    r_s = jnp.exp(log_r_s)

    numpyro.deterministic("m_tot", m_tot)
    numpyro.deterministic("r_s", r_s)

    # Optionally: if your simulator uses randomness, you can sample a key and pass it into loss.
    # Here we use a fixed key for deterministic simulator behaviour.
    # key_sim = jr.PRNGKey(0)

    # Compute the log-likelihood (simulator must return a scalar logpdf)
    loglike = loss(m_tot, r_s, key_sim=None)

    # Insert into probabilistic model
    numpyro.factor("stream_loglike", loglike)

# --------------------------
# ---- CLI / main ----------
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mcmc-method", type=str, default="NUTS", help="MCMC method to use (default: NUTS)")
    p.add_argument("--reparm", type=bool, default=False, )
    p.add_argument("--num-warmup", type=int, default=400, help="NUTS warmup steps")
    p.add_argument("--num-samples", type=int, default=1000, help="NUTS posterior samples")
    p.add_argument("--num-chains", type=int, default=2, help="Number of chains")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--out", type=str, default="mcmc_samples.npz", help="Output .npz file for samples")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    rng_key = jr.PRNGKey(args.seed)

    # --------------------------
    # --- Build observed data ---
    # --------------------------
    # We build stream_target once using the base potential (this is your 'observed' stream)
    base_pot = gp.CompositePotential(
            disk=gp.BovyMWPotential2014.disk,
            halo=gp.BovyMWPotential2014.halo,
            bulge=gp.BovyMWPotential2014.bulge,
    )

    # Ensure prog_mass exists: if your project defines it earlier, replace this default.
    # Default chosen conservatively — adjust to your real progenitor mass and units.
    try:
        prog_mass  # if present from earlier environment
    except NameError:
        prog_mass = Quantity(10**4.05, "Msun")  # default; change if needed

    df = gd.ChenStreamDF()
    gen = gd.MockStreamGenerator(df, base_pot)

    # initial conditions consistent with your snippet
    w = gc.PhaseSpacePosition(q=Quantity([11.8, 0.79, 6.4], "kpc"),
                                p=Quantity([109.5,-254.5,-90.3], "km/s"),
                            )
    t_array = Quantity(-jnp.linspace(0, 3000, 500), "Myr")

    print("Generating target (observed) stream with base potential (this may take a while)...")
    t0 = time.time()
    stream_c25, _ = gen.run(jr.PRNGKey(0), t_array, w, prog_mass)
    t1 = time.time()
    print(f"Stream generation done in {(t1-t0):.1f}s")

    stream_target = stream_to_array(stream_c25)
    pos_stream_target = stream_target[:, :3]
    vel_stream_target = stream_target[:, 3:]
    pos_eq_stream_target = halo_to_equatorial_batch(pos_stream_target)
    vel_eq_stream_target = transform_velocity_batch(halo_to_equatorial, pos_stream_target, vel_stream_target)
    stream_target = jnp.concatenate([pos_eq_stream_target, vel_eq_stream_target], axis=1)
    print("Prepared stream_target shape:", stream_target.shape)

    if args.reparm:
        model = stream_model_reparam
    else:
        model = stream_model

    # Run NUTS
    if args.mcmc_method == "NUTS":
        kernel = NUTS(model, 
                      max_tree_depth=4,
                      forward_mode_differentiation = True)
        mcmc = MCMC(kernel,
                    num_warmup=args.num_warmup,
                    num_samples=args.num_samples,
                    num_chains=args.num_chains,
                    progress_bar=True,)
        out = "mcmc_samples_NUTS.npz"
    else:
        print('Running AIES')
        kernel = AIES(model)
        mcmc = MCMC(kernel,
                    num_warmup=args.num_warmup,
                    num_samples=args.num_samples,
                    num_chains=args.num_chains,
                    chain_method='vectorized',
                    progress_bar=True)
        out = "mcmc_samples_AIES.npz"

    print("Starting MCMC...")
    mcmc.run(rng_key)
    print("MCMC done.")

    mcmc.print_summary()

    samples = mcmc.get_samples()
    # Save numeric results (convert to numpy)
    samples_np = {k: np.asarray(v) for k, v in samples.items()}
    np.savez(out, **samples_np)
    print(f"Saved samples to {out}")
