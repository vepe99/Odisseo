from autocvd import autocvd
autocvd(num_gpus = 1)
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1, 4'
# os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import matplotlib.pyplot as plt

import jax.random as jr

from unxt import Quantity
import galax.coordinates as gc
import galax.potential as gp
import galax.dynamics as gd

import agama

import time

import jax.numpy as jnp
import jax
from jax import jit
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from astropy import units as u

@jit
def log_prior_loguniform_logsigma(log_sigma, log_sigma_min, log_sigma_max):
    # uniform prior on log(sigma) between bounds -> p(log_sigma)=const inside
    # returns log p(sigma) up to additive constant
    in_bounds = (log_sigma >= log_sigma_min) & (log_sigma <= log_sigma_max)
    # if outside, return -inf
    return jnp.where(in_bounds, -jnp.log(log_sigma_max - log_sigma_min), -jnp.inf)


@jit
def log_diag_multivariate_normal(x, mean, sigma_eff):
    """
    Log PDF of a multivariate Gaussian with diagonal covariance.
    sigma_eff : (D,)  # effective standard deviation per dimension
    """
    print('x shape:', x.shape)
    print('mean shape:', mean.shape)
    print('sigma_eff shape:', sigma_eff.shape)
    diff = (x - mean) / sigma_eff
    D = x.shape[0]
    log_det = 2.0 * jnp.sum(jnp.log(sigma_eff))
    norm_const = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det)
    exponent = -0.5 * jnp.sum(diff**2)
    return norm_const + exponent

@jit
def stream_likelihood_diag(model_stream, obs_stream, obs_errors, smooth_sigma):
    """
    Log-likelihood of observed stars given simulated stream (diagonal covariance),
    including model smoothing variance term Σ_k^2.
    
    Parameters
    ----------
    model_stream : (K, D)
    obs_stream : (N, D)
    obs_errors : (D,) or (N, D)
    smooth_sigma : (D,)  # per-dimension smoothing std deviation
    """
    sigma_eff = jnp.sqrt(obs_errors**2 + smooth_sigma**2)

    def obs_log_prob(obs):
        def model_log_prob(model_point):
            return log_diag_multivariate_normal(obs, model_point, sigma_eff)
        log_probs = jax.vmap(model_log_prob)(model_stream)
        log_p_stream = jax.scipy.special.logsumexp(log_probs) - jnp.log(model_stream.shape[0])
        return log_p_stream

    logL_values = jax.vmap(obs_log_prob)(obs_stream)
    return jnp.sum(logL_values)  # sum, not mean



@jit
def log_multivariate_normal(x, mean, cov):
    """
    Log PDF of a multivariate Gaussian with full covariance.

    Parameters
    ----------
    x : (D,)
    mean : (D,)
    cov : (D, D)  # covariance matrix (must be symmetric positive definite)
    """
    D = x.shape[0]
    L = jnp.linalg.cholesky(cov)
    diff = x - mean
    solve = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    mahal = jnp.sum(solve**2)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    norm_const = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det)
    return norm_const - 0.5 * mahal


def make_smoothing_covariance(smooth_sigma, rho=0.3):
    """
    Build correlated covariance matrix for smoothing kernel.
    rho: correlation coefficient between adjacent coordinates (0=no correlation, 1=perfect).
    smooth_sigma: (D,)
    """
    D = smooth_sigma.shape[0]
    Sigma = jnp.outer(smooth_sigma, smooth_sigma)
    # Exponential correlation by distance between indices
    idx = jnp.arange(D)
    corr = rho ** jnp.abs(idx[:, None] - idx[None, :])
    return Sigma * corr


@jit
def stream_likelihood_fullcov(model_stream, obs_stream, obs_errors, smooth_sigma, rho=0.3):
    """
    Log-likelihood of observed stars given simulated stream (full covariance version).
    """
    cov = make_smoothing_covariance(jnp.sqrt(obs_errors**2 + smooth_sigma**2), rho=rho)

    def obs_log_prob(obs):
        def model_log_prob(model_point):
            return log_multivariate_normal(obs, model_point, cov)
        log_probs = jax.vmap(model_log_prob)(model_stream)
        return jax.scipy.special.logsumexp(log_probs) - jnp.log(model_stream.shape[0])

    logL_values = jax.vmap(obs_log_prob)(obs_stream)
    return jnp.sum(logL_values)


def marginalize_sigma_rho_grid(stream_loglikelihood_fn,
                               model_stream, obs_stream, obs_errors,
                               log_sigma_min=-4.0, log_sigma_max=0.0, n_sigma=32,
                               rho_min=0.0, rho_max=0.9, n_rho=16,
                               prior_sigma_fn=log_prior_loguniform_logsigma):
    log10_grid = jnp.linspace(log_sigma_min, log_sigma_max, n_sigma)
    ln10 = jnp.log(10.0)
    log_sigma_grid_nat = log10_grid * ln10
    sigma_grid = jnp.exp(log_sigma_grid_nat)
    rho_grid = jnp.linspace(rho_min, rho_max, n_rho)

    def eval_one(s, rho):
        ll = stream_loglikelihood_fn(model_stream, obs_stream, obs_errors, s, rho)
        lp = prior_sigma_fn(jnp.log(s), log_sigma_min * ln10, log_sigma_max * ln10)
        return ll + lp

    batched_eval = jax.vmap(lambda s:
        jax.vmap(lambda rho: eval_one(s, rho))(rho_grid)
    )(sigma_grid)

    # marginalize numerically over ln(sigma) and rho
    dx = (log_sigma_grid_nat[1] - log_sigma_grid_nat[0])
    drho = (rho_grid[1] - rho_grid[0])
    log_marginal = jax.scipy.special.logsumexp(batched_eval + jnp.log(dx) + jnp.log(drho))
    return log_marginal




def stream_to_array(stream):
    pos = jnp.array([stream.q.x.to('kpc').value, stream.q.y.to('kpc').value, stream.q.z.to('kpc').value])
    vel = jnp.array([stream.p.x.to('km/s').value, stream.p.y.to('km/s').value, stream.p.z.to('km/s').value])
    return pos.T, vel.T


from functools import partial

# Example: stream_loglikelihood(model_stream, obs_stream, obs_errors, smooth_sigma)
# should return scalar log p(data | theta, smooth_sigma).

def log_prior_loguniform_logsigma(log_sigma, log_sigma_min, log_sigma_max):
    # uniform prior on log(sigma) between bounds -> p(log_sigma)=const inside
    # returns log p(sigma) up to additive constant
    in_bounds = (log_sigma >= log_sigma_min) & (log_sigma <= log_sigma_max)
    # if outside, return -inf
    return jnp.where(in_bounds, -jnp.log(log_sigma_max - log_sigma_min), -jnp.inf)


def marginalize_sigma_grid(stream_loglikelihood_fn,
                           model_stream, obs_stream, obs_errors,
                           log_sigma_min=-6.0, log_sigma_max=1.0, n_grid=128,
                           prior_fn=log_prior_loguniform_logsigma):
    """
    Numerically marginalize the likelihood over sigma using a log-grid.

    Arguments
    ---------
    stream_loglikelihood_fn: function(model_stream, obs_stream, obs_errors, smooth_sigma) -> log-likelihood scalar
    model_stream, obs_stream, obs_errors: as in your likelihood
    log_sigma_min, log_sigma_max: bounds in log10(sigma) **for the unit system of sigma**
                                (we'll work in natural log inside)
    n_grid: number of grid points in log10-space
    prior_fn: function(log_sigma_nat) -> log prior density (natural-log)
              The function receives natural-log sigma (ln(s)), not log10.
    Returns
    -------
    log_marginal_likelihood: scalar (natural log)
    """

    # grid in log10 space, but do arithmetic in natural log for accuracy
    log10_grid = jnp.linspace(log_sigma_min, log_sigma_max, n_grid)  # base-10 exponents
    # convert to natural log of sigma
    ln10 = jnp.log(10.0)
    log_sigma_grid_nat = log10_grid * ln10  # ln(sigma)
    sigma_grid = jnp.exp(log_sigma_grid_nat)  # sigma values

    # vectorized log-likelihood evaluation over grid of sigma
    # we expect stream_loglikelihood_fn to accept sigma either scalar or array
    batched_ll = jax.vmap(lambda s: stream_loglikelihood_fn(model_stream,
                                                             obs_stream,
                                                             obs_errors,
                                                             s))(sigma_grid)  # shape (n_grid,)

    # compute log prior for each grid point (prior on ln(sigma))
    log_prior_vals = jax.vmap(lambda ln_s: prior_fn(ln_s, log_sigma_min * ln10, log_sigma_max * ln10))(log_sigma_grid_nat)
    # Note: prior_fn expects natural-log bounds if you implemented it that way.

    # integration weight: dx = delta(ln sigma) when integrating over ln(sigma).
    # We're integrating p(data|s) p(s) ds. If prior is on ln(s), integral = ∫ p(data|s) p(ln s) e^{ln s} d(ln s)
    # Simpler: do the integral over ln(s): ∫ p(data|s(lns)) p(s(lns)) * exp(lns) dlns
    # If prior_fn returns log p(ln s) (i.e. prior on lns), adjust accordingly.
    # For a log-uniform prior p(s) ∝ 1/s -> p(ln s) is constant and p(s) = exp(-ln s + const), but easier:
    # We'll compute log integrand as: log p(data|s) + log p(s) and convert ds via trapezoid in ln(s).

    # Here we will assume prior_fn returns log p(ln s) (i.e., prior density per d(ln s)). If your prior is
    # uniform in ln s, log_p(ln s) is constant. For convenience, implement prior on ln(s) (natural log).
    # So integrand in d(ln s) is: p(data | s(lns)) * p(ln s)  (and integral over dlns)
    # That avoids extra Jacobian factors.

    # If prior_fn is log p(ln s), we can compute:
    log_integrand = batched_ll + log_prior_vals  # log of the integrand over d(ln s)

    # integrate over ln(s) with the trapezoid rule in log-space for numerical stability:
    # convert to linear weights with exp and trapezoid spacing, but do it with logsumexp stabilization
    # using the log-sum-exp plus log(dx) approach.

    # dx in ln(s) space (natural log)
    dx = (log_sigma_grid_nat[1] - log_sigma_grid_nat[0])
    # log of integrand plus log(dx)
    log_integrand_plus_dx = log_integrand + jnp.log(dx)

    # stable sum in log: log ∑ exp(log_integrand_plus_dx)
    log_marginal = jax.scipy.special.logsumexp(log_integrand_plus_dx)

    return log_marginal

# def marginalize_sigma_grid(stream_loglikelihood_fn,
#                            model_stream, obs_stream, obs_errors,
#                            log_sigma_min=-6.0, log_sigma_max=1.0, n_grid=128,
#                            prior_fn=log_prior_loguniform_logsigma):
#     """
#     Numerically marginalize the likelihood over sigma using a log-grid,
#     where each smoothing sigma is scaled by the model stream's dispersion.

#     The effective smoothing at each grid point is:
#         smooth_sigma = sigma_grid * std(model_stream, axis=0)

#     This ties the smoothing strength to the model stream's natural scale.

#     Parameters
#     ----------
#     stream_loglikelihood_fn : callable
#         Function (model_stream, obs_stream, obs_errors, smooth_sigma) -> log-likelihood scalar
#     model_stream, obs_stream, obs_errors : array-like
#         As in your likelihood
#     log_sigma_min, log_sigma_max : float
#         Bounds in log10(sigma), controlling the *fractional* smoothing (e.g., 1e-3–1)
#     n_grid : int
#         Number of grid points in log10 space
#     prior_fn : callable
#         Function(log_sigma_nat) -> log prior density (in natural log sigma space)

#     Returns
#     -------
#     log_marginal_likelihood : float
#         The marginalized log-likelihood over sigma.
#     """

#     # Compute characteristic scale of the model stream
#     model_std = jnp.std(model_stream, axis=0)  # (D,)
#     ln10 = jnp.log(10.0)

#     # grid in log10-space
#     log10_grid = jnp.linspace(log_sigma_min, log_sigma_max, n_grid)
#     log_sigma_grid_nat = log10_grid * ln10
#     sigma_grid = jnp.exp(log_sigma_grid_nat)  # scalar multipliers

#     # Vectorized likelihood evaluation
#     def logL_for_sigma(s):
#         # scale the model-dependent smoothing
#         smooth_sigma = s * model_std
#         return stream_loglikelihood_fn(model_stream, obs_stream, obs_errors, smooth_sigma)

#     batched_ll = jax.vmap(logL_for_sigma)(sigma_grid)

#     # Compute log prior (prior on ln σ)
#     log_prior_vals = jax.vmap(
#         lambda ln_s: prior_fn(ln_s, log_sigma_min * ln10, log_sigma_max * ln10)
#     )(log_sigma_grid_nat)

#     # Combine likelihood and prior
#     log_integrand = batched_ll + log_prior_vals

#     # Integration over ln(sigma)
#     dx = log_sigma_grid_nat[1] - log_sigma_grid_nat[0]
#     log_integrand_plus_dx = log_integrand + jnp.log(dx)
#     log_marginal = jax.scipy.special.logsumexp(log_integrand_plus_dx)

#     return log_marginal

# @jit
# def stream_likelihood_diag(model_stream, obs_stream, obs_errors, smooth_sigma,
#                            outlier_sigma=3.0):
#     """
#     Log-likelihood of observed stars given simulated stream (diagonal covariance),
#     with outlier rejection implemented via jnp.where (median - Nσ rule).

#     Parameters
#     ----------
#     model_stream : (K, D)
#     obs_stream : (N, D)
#     obs_errors : (D,) or (N, D)
#     smooth_sigma : (D,)
#     outlier_sigma : float
#         Threshold multiplier for standard deviation below median to reject outliers.
#     """
#     sigma_eff = jnp.sqrt(obs_errors**2 + smooth_sigma**2)

#     def obs_log_prob(obs):
#         def model_log_prob(model_point):
#             return log_diag_multivariate_normal(obs, model_point, sigma_eff)
#         log_probs = jax.vmap(model_log_prob)(model_stream)
#         return jax.scipy.special.logsumexp(log_probs) - jnp.log(model_stream.shape[0])

#     # Per–observation log-likelihoods
#     logL_values = jax.vmap(obs_log_prob)(obs_stream)

#     # Replace non-finite values with very low numbers
#     logL_values = jnp.where(jnp.isfinite(logL_values), logL_values, -jnp.inf)

#     # Compute robust stats (ignoring -inf implicitly)
#     finite_mask = jnp.isfinite(logL_values)
#     valid_vals = jnp.where(finite_mask, logL_values, 0.0)
#     count_valid = jnp.sum(finite_mask)
#     mean = jnp.sum(valid_vals) / count_valid
#     std = jnp.sqrt(jnp.sum((jnp.where(finite_mask, logL_values - mean, 0.0))**2) / count_valid)
#     median = jnp.median(jnp.where(finite_mask, logL_values, mean))

#     # Outlier limit
#     limit = median - outlier_sigma * std

#     # Use jnp.where to zero out outliers
#     logL_clipped = jnp.where(logL_values >= limit, logL_values, 0.0)

#     # Sum all contributions (outliers contribute 0)
#     total_logL = jnp.sum(logL_clipped)

#     return total_logL

@jax.jit
def halo_to_sun(Xhalo: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from simulation frame to cartesian frame centred at Sun
    Args:
      Xhalo: 3d position (x [kpc], y [kpc], z [kpc]) in simulation frame
    Returns:
      3d position (x_s [kpc], y_s [kpc], z_s [kpc]) in Sun frame
    Examples
    --------
    >>> halo_to_sun(jnp.array([1.0, 2.0, 3.0]))
    """
    sunx = 8.0
    xsun = sunx - Xhalo[0]
    ysun = Xhalo[1]
    zsun = Xhalo[2]
    return jnp.array([xsun, ysun, zsun])


@jax.jit
def sun_to_gal(Xsun: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from sun cartesian frame to galactic co-ordinates
    Args:
      Xsun: 3d position (x_s [kpc], y_s [kpc], z_s [kpc]) in Sun frame
    Returns:
      3d position (r [kpc], b [rad], l [rad]) in galactic frame
    Examples
    --------
    >>> sun_to_gal(jnp.array([1.0, 2.0, 3.0]))
    """
    r = jnp.linalg.norm(Xsun)
    b = jnp.arcsin(Xsun[2] / r)
    l = jnp.arctan2(Xsun[1], Xsun[0])
    return jnp.array([r, b, l])


@jax.jit
def gal_to_equat(Xgal: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from galactic co-ordinates to equatorial co-ordinates
    Args:
      Xgal: 3d position (r [kpc], b [rad], l [rad]) in galactic frame
    Returns:
      3d position (r [kpc], alpha [rad], delta [rad]) in equatorial frame
    Examples
    --------
    >>> gal_to_equat(jnp.array([1.0, 2.0, 3.0]))
    """
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
    """
    Generic velocity transformation through coordinate mapping.

    Args:
      transform_fn: function R^3 → R^3 mapping positions to new coordinates
      X: position vector in original coordinates (3,)
      V: velocity vector in original coordinates (3,)

    Returns:
      velocity vector in transformed coordinates (3,)
    """
    J = jax.jacobian(transform_fn)(X)  # (3,3) Jacobian
    return J @ V

def halo_to_equatorial(Xhalo):
    Xsun = halo_to_sun(Xhalo)
    Xgal = sun_to_gal(Xsun)
    Xeq  = gal_to_equat(Xgal)
    return Xeq


#vamp functions
halo_to_equatorial_batch = jax.vmap(halo_to_equatorial, in_axes=(0))
transform_velocity_batch = jax.vmap(transform_velocity, in_axes=(None, 0, 0))


if __name__ == "__main__":
    key = jr.PRNGKey(0)

    milky_way_pot = gp.BovyMWPotential2014()
    # print(milky_way_pot.halo.r_s)
    # print(milky_way_pot.disk.a)
    # print(milky_way_pot.disk.b)

    # milky_way_pot = gp.CompositePotential(
    #         halo = gp.TriaxialNFWPotential(
    #             m=1.09 * 10**12 * u.Msun,
    #             r_s=16.0 * u.kpc,   
    #             units='galactic'
    #         )
    #     )
    # exit()

    w = gc.PhaseSpacePosition(q=Quantity([11.8, 0.79, 6.4], "kpc"),
                            p=Quantity([109.5,-254.5,-90.3], "km/s"),
                        )

    t_array = Quantity(-np.linspace(0, 3000, 2000), "Myr")
    prog_mass = Quantity(10**4.05, "Msun")
    pot_target = milky_way_pot
    df = gd.ChenStreamDF()
    gen = gd.MockStreamGenerator(df, pot_target)
    stream_c25, _ = gen.run(key, t_array, w, prog_mass)

    pos_target, vel_target = stream_to_array(stream_c25)
    pos_eq_target = pos_target
    vel_eq_target = vel_target
    print('pos_target shape:', pos_target.shape)
    print('vel_target shape:', vel_target.shape)

    # Transform to equatorial coordinates
    # pos_eq_target = halo_to_equatorial_batch(pos_target)
    # vel_eq_target = transform_velocity_batch(halo_to_equatorial, pos_target, vel_target)
    stream_target = jnp.concatenate([pos_eq_target, vel_eq_target], axis=1)
    print('stream_target shape:', stream_target.shape)
    # exit()

    @jit
    def loss(m_tot):

        pot_new = gp.CompositePotential(
            disk=milky_way_pot.disk,
            bulge=milky_way_pot.bulge,
            halo=gp.NFWPotential(
                m=m_tot,
                r_s=milky_way_pot.halo.r_s.value.value,
                units='galactic'
            )
        )

        t_array = Quantity(-jnp.linspace(0, 3000, 2000), "Myr")
        prog_mass = Quantity(10**4.05, "Msun")
        df = gd.ChenStreamDF()
        gen = gd.MockStreamGenerator(df, pot_new)
        stream_c25_new, _ = gen.run(key, t_array, w, prog_mass)

        stream_pos, stream_vel = stream_to_array(stream_c25_new)
        pos_eq, vel_eq = stream_pos, stream_vel
        # pos_eq = halo_to_equatorial_batch(stream_pos)
        # vel_eq = transform_velocity_batch(halo_to_equatorial, stream_pos, stream_vel)
        stream = jnp.concatenate([pos_eq, vel_eq], axis=1)

        # noise_std = jnp.array([10, 5, 0.1, 10.0, 5.0, 5.0])  # minimal observational noise
        noise_std = jnp.zeros(6)  # no observational noise
        pos_std = jnp.std(pos_eq, axis=0)
        vel_std = jnp.std(vel_eq, axis=0)

        # jax.debug.print('{pos_std}', pos_std=pos_std)
        # jax.debug.print('{vel_std}', vel_std=vel_std)
        def smooth_sigma_fn(perc):
            return jnp.concatenate([perc * pos_std, perc * vel_std])

        # now marginalize only over σ, no ρ
        # log_marginal = marginalize_sigma_grid(
        #     stream_likelihood_diag,
        #     stream, stream_target, noise_std,
        #     log_sigma_min=-6, log_sigma_max=2,
        #     n_grid=100
        # )
        log_marginal = stream_likelihood_diag(
            stream, stream_target, noise_std,
            smooth_sigma=smooth_sigma_fn(0.1)
        )
        return log_marginal



    true_NFW_mass = milky_way_pot.halo.m.value.value
    # true_NFW_mass = 1.09e12
    # true_NFW_mass = milky_way_pot.disk.m_tot.value.value
    mass_NFW = jnp.sort(jnp.concatenate([jnp.linspace(true_NFW_mass * 0.25, true_NFW_mass * 2, 1200), jnp.array([true_NFW_mass])]))
    mesh = Mesh(np.array(jax.devices()), ("i",))
    mass_NFW = jax.device_put(mass_NFW, NamedSharding(mesh, PartitionSpec("i")))
    # loss_value = jax.vmap(loss)(mass_NFW)
    loss_value = jax.lax.map(loss, mass_NFW, batch_size=300)
    grad_value = jax.lax.map(jax.jacfwd(loss), mass_NFW, batch_size=300)
    # grad_value = jax.lax.map(jax.grad(loss), mass_NFW, batch_size=7)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(2,2,1)
    ax.plot(mass_NFW, loss_value)
    ax.axvline(true_NFW_mass, color='red', linestyle='--', label='True NFW Mass')
    ax.scatter(mass_NFW[jnp.argmax(loss_value)], jnp.max(loss_value), color='green', label='Max Likelihood')
    ax.set_xlabel('NFW Halo Mass [Msun]')
    ax.set_ylabel('Log-Likelihood')

    ax = fig.add_subplot(2,2,2, projection='3d')
    pot_plot = gp.CompositePotential(
            disk=milky_way_pot.disk,
            # disk = gp.MiyamotoNagaiPotential(m_tot=mass_NFW[jnp.argmax(loss_value)], a=6.5, b=0.26, units="galactic" ),
            halo=gp.NFWPotential(m=mass_NFW[jnp.argmax(loss_value)], r_s=16, units="galactic"),
            # halo=milky_way_pot.halo,
            bulge=milky_way_pot.bulge,
        )
    # pot = gp.CompositePotential(
    #         halo = gp.TriaxialNFWPotential(
    #             m=mass_NFW[jnp.argmax(loss_value)],
    #             r_s=16.0 * u.kpc,   
    #             units='galactic'
    #         )
    #     )

    df = gd.ChenStreamDF()
    gen = gd.MockStreamGenerator(df, pot_plot)
    best_stream, _ = gen.run(jr.key(0), t_array, w, prog_mass)
    stream_pos, stream_vel = stream_to_array(best_stream)

    ax.scatter(stream_pos[:,0], stream_pos[:,1], stream_pos[:,2], c='b', s=5, label='Best-fit Stream')
    ax.scatter(pos_target[:,0], pos_target[:,1], pos_target[:,2], c='r', s=10,  label='Target Stream')
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.set_zlabel('Z [kpc]')
    ax.legend()

    ax = fig.add_subplot(2,2,3)
    ax.scatter(mass_NFW, jnp.sqrt(grad_value**2), c=np.where(grad_value>0, 'r', 'b'))
    ax.axvline(mass_NFW[jnp.argmax(loss_value)], color='r', label='True $M_{tot}$')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig('./plot/loss_landscape/loss_landscale_M_NFW_chen25_diagcovariance.png')