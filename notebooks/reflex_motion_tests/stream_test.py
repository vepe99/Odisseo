print("Running reflex motion test")

# Only make a single GPU visible
from autocvd import autocvd

autocvd(num_gpus=1)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import numpy as np

import jax.numpy as jnp
from jax import random

import plotly.graph_objects as go

from astropy import units as u

from odisseo import construct_initial_state
from odisseo.dynamics import DIRECT_ACC_MATRIX
from odisseo.option_classes import (
    SimulationConfig,
    SimulationParams,
    HernquistParams,
    NFWParams,
    MNParams,
    PSPParams,
    PointMassParams,
    DynamicalFrictionParams,
    PlummerParams,
)
from odisseo.option_classes import (
    NFW_POTENTIAL,
    DIFFRAX_BACKEND,
    MN_POTENTIAL,
    PSP_POTENTIAL,
    HERNQUIST_POTENTIAL,
    DOPRI8,
)
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.initial_condition import Plummer_sphere
from odisseo.visualization import MW_pot_representation

print("Imports successful")

# Setup
code_length = 10 * u.kpc
code_mass = 1e11 * u.Msun
code_time = 1 * u.Gyr
code_units = CodeUnits(
    unit_length=code_length, unit_mass=code_mass, unit_time=code_time, G=None
)

config = SimulationConfig(
    N_particles=10000,
    return_snapshots=True,
    num_snapshots=1000,
    fixed_timestep=False,
    num_timesteps=1000,
    softening=0.1 * u.pc.to(code_units.code_length),
    integrator=DIFFRAX_BACKEND,
    acceleration_scheme=DIRECT_ACC_MATRIX,
    external_accelerations=(
        (NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL),
        (HERNQUIST_POTENTIAL,),
    ),
    # external_accelerations = ((POINT_MASS,),(HERNQUIST_POTENTIAL,)),
    reflex_motion=True,
    diffrax_solver=DOPRI8,
)

params = SimulationParams(
    t_end=1 * u.Gyr.to(code_units.code_time),
    G=code_units.G,
    Hernquist_params=HernquistParams(
        M=15e10 * u.Msun.to(code_units.code_mass),
        r_s=17.14 * u.kpc.to(code_units.code_length),
    ),
    NFW_params=NFWParams(
        Mvir=4.3683325e11 * u.Msun.to(code_units.code_mass),
        r_s=15.3 * u.kpc.to(code_units.code_length),
    ),
    MN_params=MNParams(
        M=68_193_902_782.346756 * u.Msun.to(code_units.code_mass),
        a=3.0 * u.kpc.to(code_units.code_length),
        b=0.28 * u.kpc.to(code_units.code_length),
    ),
    PSP_params=PSPParams(
        M=4501365375.06545 * u.Msun.to(code_units.code_mass),
        alpha=1.8,
        r_c=1.9 * u.kpc.to(code_units.code_length),
    ),
    PointMass_params=PointMassParams(M=15e11 * u.Msun.to(code_units.code_mass)),
    DynamicalFriction_params=DynamicalFrictionParams(
        sigma_MW=120.0 * (u.km / u.s).to(code_units.code_velocity),
        lambda_df=0.001,
        coulomb_log_numerator=100.0 * u.kpc.to(code_units.code_length),
    ),
    Plummer_params=PlummerParams(
        Mtot=10**4.05 * u.Msun.to(code_units.code_mass),
        a=100 * u.pc.to(code_units.code_length),
    ),
)

key = random.PRNGKey(1)

# Setup COM backwards integration
config_com = config._replace(
    N_particles=1,
)
params_com = params._replace(
    t_end=-params.t_end,
)

# Setup COM of Plummer sphere as a particle
pos_com = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
vel_com = jnp.array([[109.5, -254.5, -90.3]]) * (u.km / u.s).to(
    code_units.code_velocity
)
mass_com = jnp.array([params_com.Plummer_params.Mtot])
state_com = construct_initial_state(
    pos_com, vel_com
)  # state is a (N_particles x 2 x 3)

# Add the Milky Way as particle in index 0
MW_pos_com = jnp.array([[0.0, 0.0, 0.0]]) * u.kpc.to(code_units.code_length)
MW_vel_com = jnp.array([[0.0, 0.0, 0.0]]) * (u.km / u.s).to(code_units.code_velocity)
MW_mass = jnp.array([15e11]) * u.Msun.to(code_units.code_mass)
# MW_mass = jnp.array([4.3683325e11]) * u.Msun.to(code_units.code_mass)
MW_state_com = construct_initial_state(MW_pos_com, MW_vel_com)

# Add the LMC as particle in index 1
LMC_pos_com = jnp.array([[-0.6, -41.3, -27.1]]) * u.kpc.to(code_units.code_length)
LMC_vel_com = jnp.array([[-63.9, -213.8, 206.6]]) * (u.km / u.s).to(
    code_units.code_velocity
)
LMC_mass = jnp.array([15e10]) * u.Msun.to(code_units.code_mass)
# LMC_mass = jnp.array([4.3683325e11]) * u.Msun.to(code_units.code_mass)
LMC_state_com = construct_initial_state(LMC_pos_com, LMC_vel_com)

state_com = jnp.concatenate([MW_state_com, LMC_state_com, state_com], axis=0)
mass_com = jnp.concatenate([MW_mass, LMC_mass, mass_com], axis=0)

print("Setup successful")

# Backwards simulation only using COM of Plummer sphere
snapshots_com = time_integration(
    state_com, mass_com, config_com, params_com
)  # snapshots is a (N_snapshots x N_particles x 2 x 3)
print("Backward time integration successful")

# Final (earlier) state of all particles
MW_state_stream = jnp.array([snapshots_com.states[-1, 0]])
LMC_state_stream = jnp.array([snapshots_com.states[-1, 1]])
sphere_state_stream = snapshots_com.states[-1, 2]

# Add the final COM position and velocity to the Plummer sphere particles
sphere_pos, sphere_vel, sphere_mass = Plummer_sphere(
    key=key, params=params, config=config
)
rel_sphere_pos = sphere_pos * u.kpc.to(code_units.code_length) + sphere_state_stream[0]
rel_sphere_vel = (
    sphere_vel * (u.km / u.s).to(code_units.code_velocity) + sphere_state_stream[1]
)
sphere_mass = sphere_mass * u.Msun.to(code_units.code_mass)

# Setup forward integration with the Plummer sphere particles
particle_state_stream = construct_initial_state(rel_sphere_pos, rel_sphere_vel)

state_stream = jnp.concatenate(
    [MW_state_stream, LMC_state_stream, particle_state_stream], axis=0
)
mass = jnp.concatenate([MW_mass, LMC_mass, sphere_mass], axis=0)

# Forward simulation with the Plummer sphere particles
snapshots = time_integration(state_stream, mass, config, params)
print("Forward time integration successful")

# Visualization
length_scale = code_units.code_length.to(u.kpc)
time_scale = code_units.code_time.to(u.Gyr)
MW_trajectory = snapshots.states[:, 0, 0, :] * length_scale
LMC_trajectory = snapshots.states[:, 1, 0, :] * length_scale - MW_trajectory
PlummerSphere_trajectory = (
    snapshots.states[:, 2:, 0, :] * length_scale - MW_trajectory[:, None, :]
)

(x_halo, y_halo, z_halo), (x_bulge, y_bulge, z_bulge), (x_disk, y_disk, z_disk) = (
    MW_pot_representation(params)
)

fig = go.Figure()

# Halo
fig.add_trace(
    go.Surface(
        x=x_halo,
        y=y_halo,
        z=z_halo,
        opacity=0.15,
        colorscale=[[0, "gray"], [1, "gray"]],
        showscale=False,
        name="Halo",
    )
)

# Disk
fig.add_trace(
    go.Surface(
        x=x_disk,
        y=y_disk,
        z=z_disk,
        opacity=0.5,
        colorscale=[[0, "darkgray"], [1, "darkgray"]],
        showscale=False,
        name="Disk",
    )
)

# Bulge
fig.add_trace(
    go.Surface(
        x=x_bulge,
        y=y_bulge,
        z=z_bulge,
        opacity=0.9,
        colorscale=[[0, "dimgray"], [1, "dimgray"]],
        showscale=False,
        name="Bulge",
    )
)

# Plummer sphere Particle position
fig.add_trace(
    go.Scatter3d(
        x=PlummerSphere_trajectory[0, :, 0],
        y=PlummerSphere_trajectory[0, :, 1],
        z=PlummerSphere_trajectory[0, :, 2],
        mode="markers",
        name="Plummer Sphere Start",
        marker=dict(color="green", size=2),
    )
)
fig.add_trace(
    go.Scatter3d(
        x=PlummerSphere_trajectory[-1, :, 0],
        y=PlummerSphere_trajectory[-1, :, 1],
        z=PlummerSphere_trajectory[-1, :, 2],
        mode="markers",
        name="Plummer Sphere End",
        marker=dict(color="red", size=2),
    )
)

# MW trajectory
# fig.add_trace(go.Scatter3d(
#     x = MW_trajectory[:, 0],
#     y = MW_trajectory[:, 1],
#     z = MW_trajectory[:, 2],
#     mode='lines', showlegend=False, line=dict(
#         color=snapshots.times * time_scale,
#         colorscale='Viridis',
#         width=4
#     )
# ))
# fig.add_trace(
#     go.Scatter3d(
#         x=[MW_trajectory[0, 0]],
#         y=[MW_trajectory[0, 1]],
#         z=[MW_trajectory[0, 2]],
#         mode="markers",
#         name="MW",
#         marker=dict(color="purple"),
#     )
# )
# fig.add_trace(
#     go.Scatter3d(
#         x=[MW_trajectory[-1, 0]],
#         y=[MW_trajectory[-1, 1]],
#         z=[MW_trajectory[-1, 2]],
#         mode="markers",
#         showlegend=False,
#         marker=dict(color="purple"),
#     )
# )

# LMC trajectory
fig.add_trace(
    go.Scatter3d(
        x=LMC_trajectory[:, 0],
        y=LMC_trajectory[:, 1],
        z=LMC_trajectory[:, 2],
        mode="lines",
        showlegend=False,
        line=dict(
            color=snapshots.times * time_scale,
            colorscale="Viridis",
            width=4,
        ),
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[LMC_trajectory[0, 0]],
        y=[LMC_trajectory[0, 1]],
        z=[LMC_trajectory[0, 2]],
        mode="markers",
        name="LMC",
        marker=dict(color="blue"),
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[LMC_trajectory[-1, 0]],
        y=[LMC_trajectory[-1, 1]],
        z=[LMC_trajectory[-1, 2]],
        mode="markers",
        showlegend=False,
        marker=dict(color="blue"),
    )
)

fig.update_layout(
    scene=dict(
        xaxis_title="X [kpc]",
        yaxis_title="Y [kpc]",
        zaxis_title="Z [kpc]",
        aspectmode="data",
    )
)

fig.write_html("./notebooks/reflex_motion_tests/orbits_PlummerSphere_MWcof.html")
