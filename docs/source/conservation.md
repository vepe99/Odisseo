Sanity Check
============

# Energy and Angular momentum conservation

Accurate long-term simulations in N-body dynamics hinge on the precise computation of forces and the integration of particle trajectories. One of the most fundamental validation steps in any N-body code is verifying the conservation of physical quantities—most importantly, total energy and angular momentum.

## ⚠️ Why Check for Conservation?
In a system governed by Newtonian gravity or other *time-independent* conservative forces, total energy and angular momentum should be conserved over time. These quantities serve as diagnostic tools for numerical accuracy and physical consistency:

- Total Energy conservation: in an isolated system the forces are conservative, the total mechanical energy (kinetic + potential) must remain constant.
- Angular Momentum conservation: for systems with rotational symmetry (e.g., central potentials or disk galaxies), angular momentum must be conserved.

In order to access the conservation of quantities, in Odisseo the function `energy_angular_momentum_plot` is implemented to check the relative error (a minimal example is shown below) across the time steps as a post-processing operation.

$$
\delta E = \frac{|E(t) - E(0)|}{|E(0)|}, \quad 
\delta L = \frac{|\vec{L}(t) - \vec{L}(0)|}{|\vec{L}(0)|}
$$

## Example

```python
from odisseo.option_classes import SimulationConfig
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits


code_length = 10.0 * u.kpc
code_mass = 1e8 * u.Msun
G = 1 
code_units = CodeUnits(code_length, code_mass, G=G)

config = SimulationConfig(N_particles=10_000, 
                          return_snapshots=True,                        #THE SNAPSHOTS NEED TO BE RETURNED 
                          num_snapshots=100,                            #THE NUMBER OF SNAPSHOTS THAT ARE USED 
                          num_timesteps=1_000, 
                          external_accelerations=(MN_POTENTIAL,  ), 
                          acceleration_scheme=DIRECT_ACC,
                          softening=(0.1 * u.kpc).to(code_units.code_length).value) 
                        
#### CODE TO RUN THE SIMULATION #####

snapshots = time_integration(initial_state, mass, config, params)      
energy_angular_momentum_plot(snapshots, 
                            code_units,                                                     #UNITS FOR CONVERSION
                            filename='./visualization/image/E_L_Plummer_in_MNpotential.pdf' #WHERE TO SAVE THE PLOT
                            )
```
An example of the previous function is shown below:

<img src="./notebooks/visualization/image/E_L_Plummer_in_MNpotential.png" alt="Energy and Angular Momentum Conservation" width="100%">



