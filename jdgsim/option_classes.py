from typing import NamedTuple
from jdgsim.integrators import LEAPFROG
from jdgsim.dynamics import DIRECT_ACC
from jdgsim.potentials import NFW_POTENTIAL

class NFWParams(NamedTuple):
    """
    NamedTuple containing the parameters for the NFW profile
    """
    
    Mvir: float = 1.62*1e11 #M☉
    
    r_s: float = 15.3 #kpc
    
    d_c: float = 7.18
    
    c: float = None

class SimulationParams(NamedTuple):
    """
    NamedTuple containing the parameters for the simulation. This parameter do not require recompilation
    """
    
    G: float = 4.498*10**(-6) #kpc³ / (M☉ Gyr²)
    
    t_end: float = 1.0 #Gyr
    
    NFW_params: NFWParams = NFWParams()
    
    Plummer_a:float = 7 #kpc
    

class SimulationConfig(NamedTuple):
    """
    NamedTuple containing the configuration for the simulation. This parameter require recompilation
    """
    
    N_particles: int = 1000
    
    dimensions: int = 3
    
    return_snapshots: bool = False
    
    num_snapshots: int = 10
    
    fixed_timestep: bool = True
    
    num_timesteps: int = 1000
    
    softening:float = 1e-10
    
    integrator: int = LEAPFROG
    
    acceleration_scheme: int = DIRECT_ACC
    
    external_accelerations: tuple = (NFW_POTENTIAL, )
    
