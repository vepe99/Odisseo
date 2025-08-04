Sanity Check
============

# Energy and Angular momentum conservation

Accurate long-term simulations in N-body dynamics hinge on the precise computation of forces and the integration of particle trajectories. One of the most fundamental validation steps in any N-body code is verifying the conservation of physical quantities—most importantly, total energy and angular momentum.

## Why Check for Conservation?
In a system governed by Newtonian gravity or other *time-independent* conservative forces, total energy and angular momentum should be conserved over time. These quantities serve as diagnostic tools for numerical accuracy and physical consistency:

- Total Energy conservation: in an isolated system the forces are conservative, the total mechanical energy (kinetic + potential) must remain constant.
- Angular Momentum conservation: for systems with rotational symmetry (e.g., central potentials or disk galaxies), angular momentum must be conserved.

In order to access the conservation of quantities, in Odisseo the function `` is implemented to check the relative error (defined as below) across the time steps.

$$
\delta E = \frac{|E(t) - E(0)|}{|E(0)|}, \quad 
\delta L = \frac{|\vec{L}(t) - \vec{L}(0)|}{|\vec{L}(0)|}
$$

## Example

```python



```

