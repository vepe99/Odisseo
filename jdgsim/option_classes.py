from typing import NamedTuple
from jdgsim.integrators import LEAPFROG
from jdgsim.dynamics import DIRECT_ACC
from jdgsim.potentials import NFW_POTENTIAL



class SimulationParams(NamedTuple):
    """
    NamedTuple containing the parameters for the simulation. This parameter do not require recompilation
    """
    
    G: float =  4.498*10**(-6) #kpc³ / (M☉ Gyr²)
    
    t_end: float = 1.0 #Gyr
    
    NFW_params = {'Mvir': 1.62*1e11, 'r_s': 15.3, 'd_c': 7.18, 'c':None} #M☉, kpc
    
    Plummer_a = 7 #kpc
    

class SimulationConfig(NamedTuple):
    """
    NamedTuple containing the configuration for the simulation. This parameter require recompilation
    """
    
    N_particles: int = 1000
    
    dimensions: int = 3
    
    return_snapshots: bool = False
    
    numb_snapshots: int = 10
    
    fixed_timestep: bool = True
    
    num_timesteps = 1000
    
    softening = 1e-10
    
    integrator: int = LEAPFROG
    
    acceleration_scheme = DIRECT_ACC
    
    external_accelerations = (NFW_POTENTIAL, )
    
