# from autocvd import autocvd
# autocvd(num_gpus = 1)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import numpy as np
import matplotlib.pyplot as plt

import jax.random as jr
from functools import partial
from unxt import Quantity
import galax.coordinates as gc
import galax.potential as gp
import galax.dynamics as gd

from tqdm import tqdm

import time

import jax.numpy as jnp
import jax
from jax import jit, random
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from astropy import units as u

@jit
def stream_to_array(stream):
    pos = jnp.array([stream.q.x.to('kpc').value, stream.q.y.to('kpc').value, stream.q.z.to('kpc').value])
    vel = jnp.array([stream.p.x.to('km/s').value, stream.p.y.to('km/s').value, stream.p.z.to('km/s').value])
    return pos.T, vel.T


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

@jit
def run_simulation(params):
    w = gc.PhaseSpacePosition(q=Quantity([params[6], params[7], params[8]], "kpc"),
                            p=Quantity([params[9], params[10], params[11]], "km/s"),
                        )
    milky_way_pot = gp.BovyMWPotential2014()
    t_array = Quantity(-np.linspace(0, 3000, 1000), "Myr")
    prog_mass = Quantity(params[0], "Msun")
    pot= gp.CompositePotential(
            halo = gp.NFWPotential(m=params[1], 
                                   r_s=params[2], units="galactic"),
            disk = gp.MiyamotoNagaiPotential(m_tot=params[3],
                                             a=params[4],
                                             b=params[5], units="galactic"),
            bulge=milky_way_pot.bulge,
        )
    df = gd.ChenStreamDF()
    gen = gd.MockStreamGenerator(df, pot)
    stream_c25_new, _ = gen.run(random.PRNGKey(0), t_array, w, prog_mass)

    stream_pos, stream_vel = stream_to_array(stream_c25_new)
    stream = jnp.concatenate([stream_pos, stream_vel], axis=1)
    # pos_eq = halo_to_equatorial_batch(stream_pos)
    # vel_eq = transform_velocity_batch(halo_to_equatorial, stream_pos, stream_vel)
    # stream = jnp.concatenate([pos_eq, vel_eq], axis=1)

    return stream

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation parameters")
    parser.add_argument('--batch-size', type=int, default=2500, help='Batch size for sampling')
    parser.add_argument('--num-chunks', type=int, default=1_055_000, help='Number of chunks to process')
    parser.add_argument('--start-idx', type=int, default=0, help='Starting index')
    args = parser.parse_args()
    print(args)

    print('beginning sampling...')

    milky_way_pot = gp.BovyMWPotential2014()
    params_true = {'prog_mass': 10**4.05,
                   'halo_mass': milky_way_pot.halo.m.value.value,
                   'halo_r_s': milky_way_pot.halo.r_s.value.value,
                   'disk_mass': milky_way_pot.disk.m_tot.value.value,
                   'disk_a': milky_way_pot.disk.a.value.value,
                   'disk_b': milky_way_pot.disk.b.value.value,
                  }
    # params_true_array =jnp.array([params_true['prog_mass'],
    #                               params_true['halo_mass'],
    #                               params_true['halo_r_s'],
    #                               params_true['disk_mass'],
    #                               params_true['disk_a'],
    #                               params_true['disk_b'],
    #                               11.8, 
    #                               0.79,
    #                               6.4,
    #                               9.5,
    #                               -254.5,
    #                               -90.3])

    # true_simulation = run_simulation(params_true_array)
    # np.savez('/export/data/vgiusepp/galax_data/data_varying_position_uniform_prior/sbi_sim/data/sbi-benchmarks/galax_AllParametersPosition_uniformprior/true.npz',
    #          x = true_simulation,
    #          theta = params_true_array
    #          )
    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(121, projection='3d')
    # ax.scatter(true_simulation[:,0], true_simulation[:,1], true_simulation[:,2], s=1)
    # ax.set_xlabel('R [kpc]')
    # ax.set_ylabel('alpha [rad]')
    # ax.set_zlabel('delta [rad]')
    
    # ax = fig.add_subplot(122, projection='3d')
    # ax.scatter(true_simulation[:,3], true_simulation[:,4], true_simulation[:,5], s=1)
    # ax.set_xlabel('v_R [km/s]')
    # ax.set_ylabel('v_alpha [km/s]')
    # ax.set_zlabel('v_delta [km/s]')
    # plt.savefig('./plot/true_stream.png', dpi=300)
    # plt.close()
    # print(params_true)
    # exit()

    start_time = time.time()
    batch_size = args.batch_size
    num_chunks = args.num_chunks
    name_str = args.start_idx

    for i in tqdm(range(name_str, num_chunks, batch_size)):
        rng_key = random.PRNGKey(int(i))
        params_values = jax.random.uniform(
            rng_key,
            shape=(batch_size, 12),
            minval = jnp.array([10**3,
                                0.5 * params_true['halo_mass'],
                                0.5 * params_true['halo_r_s'],
                                0.5 * params_true['disk_mass'],
                                0.5 * params_true['disk_a'],
                                0.5 * params_true['disk_b'],
                                10.0, #x
                                0.1, #y
                                6.0, #z
                                90.0, #vx
                                -280.0, #vy
                                -120.0]), #vz
            maxval = jnp.array([10**4.5,
                                2 * params_true['halo_mass'],
                                2 * params_true['halo_r_s'],
                                2 * params_true['disk_mass'],
                                2 * params_true['disk_a'],
                                2 * params_true['disk_b'],
                                14.0, #x
                                2.5,  #y
                                8.0,  #z
                                115.0, #vx
                                -230.0, #vy
                                -80.0])) #vz

        stream_samples = jax.vmap(run_simulation)(params_values)
        for j in range(batch_size):
            np.savez_compressed(f"/export/data/vgiusepp/galax_data/data_1000carthesian_varying_position_uniform_prior/file_{name_str:08d}.npz",
                                x = stream_samples[j],
                                theta = params_values[j],)
            name_str += 1


end_time = time.time()
print("Time taken to sample in seconds:", end_time - start_time)
        

    

