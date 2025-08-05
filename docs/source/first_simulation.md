Running a simulation
====================

# A simple approach

In Odisseo the set-up of the simulation is splitted in two parts:
- Parameters: physical parameters value. The gradient of the simulation can be taken with respect to them. Changing parameters does not trigger `jax.jit` recompilation. 
- Configuration: set the functions that are called by the simulation. Since it controlls also the shapes of the `jnp.arrays`, changing the configuration triggers `jax.jit` recompilation. 


## Configuration:

The simulation configuration are set by the `odisseo.option_classes.SimulationConfig` class. The main configuration that can be set are:
- `N_particles` [int]: set the *number of particles* in the simulation. It sets the first dimension of `state` and `mass` array.
- `return_snapshots` [bool]: set if the simulation will return only the final snapshot (*False*) or also the snapshots (*True*). Along with the `state` of the system, also the  `Total Energy`, `Angular Momentum` and `time` are returned.
- `num_snapshots` [int]: set the number of snapshots that have to be returned if `return_snapshots = True`.
- `fixed_timestep` [bool]: set if the simulation will run with fixed size time steps (*True*) or adopt an adaptive time step (*False*, not yet implemented !)
- `num_timesteps` [int]: the number of time steps to be used. 
- `softening` [float]: the value of the *softening length* defined as:
$\Phi(\mathbf{r}) = -\frac{G m}{\sqrt{|\mathbf{r}|^2 + \epsilon^2}}$. It is used to avoid numerical issue due to finite time stepping and close encouters between particles leading to the unphysically large accelerations.
- `integrator` [int]: set which explicit numerical integrator scheme to evolve the `state`. The available numerical scheme are:
    - LEAPFROG: second-order sympletic integrator (also know as Velocity Verlet). 
    - RK4: fourth-order Runghe-Kutta integrator.
    - DIFFRAX_BACKEND: rely to the available `diffrax` solvers. For more information check the [documentation](https://docs.kidger.site/diffrax/)
- `diffrax_solver` [int]: set which diffrax solver will be used to evolve the `state`. The implemented solver are:
    - DOPRI5 
    - TSIT5 
    - SEMIIMPLICITEULER 
    - REVERSIBLEHEUN 
    - LEAPFROGMIDPOINT 
- `acceleation_scheme` [int]: set the strategy to calculate the pairwise distance between particles. This is usually the bottle-neck of direct N-body simulations, both in terms of computational time and memory requirment. The implemented strategy are:
    - DIRECT_ACC: calculate the pair-wise distance matrix using a double `jax.vmap` on the particles `state`.
    - DIRECT_ACC_LAXMAP: calculate the pair-wise distance matrix using `jax.lax.map` and `jax.vmap`. The batch size is set by `batch_size` configuration (see below). It can also be set to use a double `jax.lax.map` by setting the configuration `double_map = True`. This strategies are generally the slowest but also the most memory efficient. 
    - DIRECT_ACC_MATRIX: calculate the pair-wise distance matrix using array broadcasting operation. This is the fastest strategy.
- `batch_size` [int]: set the batch size of the `jax.lax.map` in the pair-wise distance calculation. It is used if and only if `acceleation_scheme = DIRECT_ACC_LAXMAP`.
- `double_map` [bool]: set if a double `jax.lax.map` (*True*), or a `jax.lax.map` and `jax.vmap` are used to calculate the pair-wise distance. It is used if and only if `acceleation_scheme = DIRECT_ACC_LAXMAP`.
- `external_accelerations` [tuple]: set the external potential functions, hence the external acceleration, that will be used during the simulation. The value of the parameters of this analytic function are part of the `odisseo.option_classes.SimulationParams` and are described in the `Parameters` section below. The implemented external acceleration:
    - NFW_POTENTIAL: Navarro-Frank-White halo potential. It is characterized by two parameters: the virial Mass `Mvir` and the scale radius `r_s`.
    - POINT_MASS: Point Mass potential. It is characterized by one parameter: the mass `M`. 
    - MN_POTENTIAL: Miamoto-Nagai disk potential. It is characterized by three parameters:
    the mass `M`, the scale length `a` and the scale height `b`.
    - PSP_POTENTIAL: Power Spherical Potential Cutoff bulge potential. It is characterized by three parameters: the mass `M`, the inner power `alpha` and the cut-off radius `r_c`.
- `num_checkpoints` [int]: set the number of checkpoints to be used by the `checkpointed_while_loop` in the `odisseo.time_integration._time_integration_fixed_steps_snapshot` function. For more information check the related function in the `equinox` module at the following [link](https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/checkpointed.py).
- `progress_bar` [bool]: set if a progress bar is shown (*True*) or not (*False*) during the integration.

## Parameters:
The simulation parameters are set by the odisseo.option_classes.SimulationParams class. These parameters define the physical characteristics of the system and can be used for differentiation (e.g., computing gradients with respect to parameters). Changing parameters does not trigger JAX recompilation, so they are efficient to update between runs.
The main parameters that can be set are:

- `G` [float]: the gravitational constant used in the simulation. It is suggested to use the `odisseo.units.CodeUnits` class to return the value of `G` in the code units.
- `t_end` [float]: the final simulation time, in code units. The total duration of the simulation will run from `t=0` to `t=t_end`.
- `Plummer Potential` [PlummerParams]: set the parameters for the `odisseo.initial_condition.Plummer` function to generate a self gravitating Plummer sphere (model for dwarf galaxies and Globular Clusters). The parameters that needs to be set are:
    -`a` [float]: scale length of the Plummer sphere.
    -`Mtot` [float]: total mass of the Plummer sphere.

`External Potential Parameters`
Depending on the choice of external potential(s) in the configuration, the corresponding parameter set will be used. Each potential has its own NamedTuple class:
- `NFW Potential` [NFWParams]:
    - `Mvir` [float]: virial mass of the halo.
    - `r_s` [float]: scale radius of the halo.
- `Point Mass `[PointMassParams]:
    - `M` [float]: mass of the point mass.
- `Miyamoto-Nagai Potential` [MNParams]:
    -`M` [float]: mass of the disk.
    -`a` [float]: scale length.
    -`b` [float]: scale height.
- `Power Spherical Potential with Cutoff` [PSPParams]:
    -`M` [float]: mass of the bulge.
    -`alpha` [float]: inner slope of the density profile.
    -`r_c` [float]: cutoff radius of the bulge.

> **📌 Important**: All parameter values must be explicitly converted to code units before being passed to the simulation. This ensures consistency and correctness during integration. For more information on how to define and convert physical quantities, see the Units Conversion section.

