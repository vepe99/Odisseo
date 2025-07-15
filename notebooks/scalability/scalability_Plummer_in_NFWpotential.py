import os
from math import pi

os.environ["CUDA_VISIBLE_DEVICES"] = "7, 6"  # Use only the first GPUos.environ["CUDA_VISIBLE_DEVICES"] = "5, 4, 3, 1"  # Use only the first GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Use only the first GPU

from tqdm import tqdm
from typing import Optional, Tuple, Callable, Union, List
from functools import partial

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# jax.config.update("jax_enable_x64", True)

import numpy as np
from astropy import units as u
from astropy import constants as c

import odisseo
from odisseo import construct_initial_state
from odisseo.integrators import leapfrog
from odisseo.dynamics import direct_acc, DIRECT_ACC, DIRECT_ACC_MATRIX, DIRECT_ACC_LAXMAP
from odisseo.option_classes import SimulationConfig, SimulationParams, NFWParams, PlummerParams, NFW_POTENTIAL
from odisseo.initial_condition import Plummer_sphere, ic_two_body
from odisseo.utils import center_of_mass
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.visualization import create_3d_gif, create_projection_gif, energy_angular_momentum_plot

import time


plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 15,
})

code_length = 10.0 * u.kpc
code_mass = 1e4 * u.Msun
G = 1 
code_units = CodeUnits(code_length, code_mass, G=G, unit_time=(3 * u.Gyr))

runtime_list_direct_acc = []
N_particles_list = [1_000, 10_000, 100_000, 200_000, 500_000 ]
print("N devices:", len(jax.devices()))
print("saving to:", f'./kartick_test_data/h100/100_time_step/multi_gpu_{len(jax.devices())}_laxmap.npy')
print(N_particles_list)

batch_size = 10_000 * len(jax.devices())  # Adjust batch size based on the number of devices

for N_particles in tqdm(N_particles_list):

    config = SimulationConfig(N_particles=N_particles, 
                          return_snapshots=False, 
                          num_timesteps=100, 
                          external_accelerations=(NFW_POTENTIAL,  ), 
                          acceleration_scheme=DIRECT_ACC_LAXMAP,
                          batch_size=batch_size,
                          softening=(0.1 * u.kpc).to(code_units.code_length).value) #default values

    params = SimulationParams(t_end = (10 * u.Gyr).to(code_units.code_time).value,  
                            Plummer_params= PlummerParams(Mtot=(1e8 * u.Msun).to(code_units.code_mass).value,
                                                            a=(1 * u.kpc).to(code_units.code_length).value),
                            NFW_params = NFWParams(Mvir=(1e12 * u.Msun).to(code_units.code_mass).value,
                                                    r_s = (20 * u.kpc).to(code_units.code_length).value),
                            G=G, ) 
    
    #set up the particles in the initial state
    positions, velocities, mass = Plummer_sphere(key=random.PRNGKey(4), params=params, config=config)
    #put the Plummer sphere in a ciruclar orbit around the NFW halo
    rp=200*u.kpc.to(code_units.code_length)

    if len(config.external_accelerations)>0:
        pos, vel, _ = ic_two_body(params.NFW_params.Mvir, params.Plummer_params.Mtot, rp=rp, e=0., params=params)
        velocities = velocities + vel[1]
        positions = positions + pos[1]

    #initialize the initial state
    initial_state = construct_initial_state(positions, velocities)
    mesh = Mesh(np.array(jax.devices()), ("i",))
    initial_state = jax.device_put(initial_state, NamedSharding(mesh, PartitionSpec("i")))
    mass = jax.device_put(mass, NamedSharding(mesh, PartitionSpec("i")))
    
    times = []
    for i in range(3):
        start_time = time.time()
        snapshots = jax.block_until_ready( time_integration(initial_state, mass, config, params) )
        end_time = time.time()
        runtime = end_time - start_time
        times.append(runtime)

    # Where you're currently using timeit.timeit()
    # times = timeit.repeat(
    #     lambda: jax.block_until_ready(time_integration(initial_state, mass, config, params)),
    #     repeat=1,  # Number of times to repeat the measurement 
    # )

    mean_runtime = np.mean(times)
    std_runtime = np.std(times)
    runtime_list_direct_acc.append((mean_runtime, std_runtime))
    np.save(f'./kartick_test_data/h100/100_time_step/multi_gpu_{len(jax.devices())}_laxmap_{batch_size}.npy', np.array(runtime_list_direct_acc))