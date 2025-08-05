Running a simulation
====================

# A simple approach

In Odisseo the set-up of the simulation is splitted in two parts:
- Parameters: physical parameters value. The gradient of the simulation can be taken with respect to them. Changing parameters does not trigger `jax.jit` recompilation. 
- Configuration: set the functions that are called by the simulation. Since it controlls also the shapes of the `jnp.arrays`, changing the configuration triggers `jax.jit` recompilation. 


# Configuration:

The simulation configuration are set by the `odisseo.option_classes.SimulationConfig` class. The main configuration that can be set are:
- `N_particles` [int]: set the *number of particles* in the simulation. It sets the first dimension of `state` and `mass` array.
- `return_snapshots` [bool]: set if the simulation will return only the final snapshot (*False*) or also the snapshots (*True*). Along with the `state` of the system, also the  `Total Energy`, `Angular Momentum` and `time` are returned.
- `num_snapshots` [int]: set the number of snapshots that have to be returned if `return_snapshots = True`.
- `fixed_timestep` [bool]: set if the simulation will run with fixed size time steps (*True*) or adopt an adaptive time step (*False*, not yet implemented !)
- `num_timesteps` [int]: the number of time steps to be used. 
- `softening` [float]: the value of the *softening length* defined as:
$$\Phi(\mathbf{r}) = -\frac{G m}{\sqrt{|\mathbf{r}|^2 + \epsilon^2}}$$. It is used to avoid numerical issue due to finite time stepping and close encouters between particles leading to the unphysically large accelerations.
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
    - DIRECT_ACC:
    - DIRECT_ACC_LAXMAP:
    - DIRECT_ACC_MATRIX:
