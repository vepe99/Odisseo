print("Running reflex motion test")

from autocvd import autocvd
autocvd(num_gpus = 1)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax.numpy as jnp
from jax import random

import plotly.graph_objects as go

from astropy import units as u

from odisseo import construct_initial_state
from odisseo.dynamics import DIRECT_ACC_MATRIX
from odisseo.option_classes import SimulationConfig, SimulationParams, PlummerParams
from odisseo.option_classes import NFW_POTENTIAL, DIFFRAX_BACKEND
from odisseo.initial_condition import Plummer_sphere
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.visualization import energy_angular_momentum_plot

print("Imports successful")

code_length = 10 * u.kpc
code_mass = 1e4 * u.Msun
G = 1
code_time = 1 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  

config = SimulationConfig(N_particles=100, 
                          return_snapshots = True, 
                          num_snapshots = 1000, 
                          fixed_timestep=False,
                          num_timesteps=10,
                          softening = (0.1 * u.pc).to(code_units.code_length).value,
                          integrator=DIFFRAX_BACKEND,
                          acceleration_scheme=DIRECT_ACC_MATRIX,
                          external_accelerations=(NFW_POTENTIAL,), 
                          reflex_motion = True)

params = SimulationParams(t_end = (4 * u.Gyr).to(code_units.code_time).value,
                          Plummer_params=PlummerParams(Mtot=(2.5e4 * u.Msun).to(code_units.code_mass).value,a=(8 * u.pc).to(code_units.code_length).value),  
                          G=code_units.G) 

key = random.PRNGKey(1)

#set up the particles in the initial state
positions, velocities, mass = Plummer_sphere(key=key, params=params, config=config)

#the center of mass needs to be integrated backwards in time first 
config_com = config._replace(N_particles=1,)
params_com = params._replace(t_end=-params.t_end,)

#this is the final position of the cluster, we need to integrate backwards in time 
pos_com_final = jnp.array([[7.86390455, 0.22748727, 16.41622487]]) * u.kpc.to(code_units.code_length)
vel_com_final = jnp.array([[-42.35458106, -103.69384675, -15.48729026]]) * (u.km/u.s).to(code_units.code_velocity)
mass_com = jnp.array([params_com.Plummer_params.Mtot])
final_state_com = construct_initial_state(pos_com_final, vel_com_final) # state is a (N_particles x 2 x 3)

# Add the Milky Way as particle in index 0
MW_position = jnp.array([[0., 0., 0.]]) * u.kpc.to(code_units.code_length)
MW_velocity = jnp.array([[10., 0., 0.]]) * (u.km/u.s).to(code_units.code_velocity)
MW_mass = jnp.array([0. * u.Msun.to(code_units.code_mass)])
final_state_com = jnp.concatenate([construct_initial_state(MW_position, MW_velocity), final_state_com], axis=0)
mass_com = jnp.concatenate([jnp.zeros(1), mass_com], axis=0) # Set its mass to zero, as its effect is included as an external potential
print("Setup successful")

#evolution in time
snapshots_com = time_integration(final_state_com, mass_com, config_com, params_com)
print("Time integration successful")

#we can plot the snapshots of simulations, the snapshot are NameTuple with states=(N_snapshots x N_particles x 2 x 3) array

##### CoM orbit plot####
scale = code_units.code_length.to(u.kpc)

fig = go.Figure()

# Particle trajectory
fig.add_trace(go.Scatter3d(
    x=snapshots_com.states[:, 1, 0, 0] * scale,
    y=snapshots_com.states[:, 1, 0, 1] * scale,
    z=snapshots_com.states[:, 1, 0, 2] * scale,
    mode='lines', name='Particle trajectory', line=dict(color='black')
))
fig.add_trace(go.Scatter3d(
    x=[snapshots_com.states[0, 1, 0, 0] * scale],
    y=[snapshots_com.states[0, 1, 0, 1] * scale],
    z=[snapshots_com.states[0, 1, 0, 2] * scale],
    mode='markers', name='Initial position Particle', marker=dict(color='blue')
))
fig.add_trace(go.Scatter3d(
    x=[snapshots_com.states[-1, 1, 0, 0] * scale],
    y=[snapshots_com.states[-1, 1, 0, 1] * scale],
    z=[snapshots_com.states[-1, 1, 0, 2] * scale],
    mode='markers', name='Final position Particle', marker=dict(color='red')
))

# MW trajectory
fig.add_trace(go.Scatter3d(
    x=snapshots_com.states[:, 0, 0, 0] * scale,
    y=snapshots_com.states[:, 0, 0, 1] * scale,
    z=snapshots_com.states[:, 0, 0, 2] * scale,
    mode='lines', name='MW trajectory', line=dict(color='gray')
))
fig.add_trace(go.Scatter3d(
    x=[snapshots_com.states[0, 0, 0, 0] * scale],
    y=[snapshots_com.states[0, 0, 0, 1] * scale],
    z=[snapshots_com.states[0, 0, 0, 2] * scale],
    mode='markers', name='Initial position MW', marker=dict(color='green')
))
fig.add_trace(go.Scatter3d(
    x=[snapshots_com.states[-1, 0, 0, 0] * scale],
    y=[snapshots_com.states[-1, 0, 0, 1] * scale],
    z=[snapshots_com.states[-1, 0, 0, 2] * scale],
    mode='markers', name='Final position MW', marker=dict(color='yellow')
))

fig.update_layout(scene=dict(
    xaxis_title='X [kpc]',
    yaxis_title='Y [kpc]',
    zaxis_title='Z [kpc]'
))

fig.write_html("./tests/reflex_motion_3d_orbit.html")

energy_angular_momentum_plot(snapshots_com, code_units, "./tests/reflex_motion_energy_angular_momentum.png")