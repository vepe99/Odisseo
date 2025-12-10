from astropy import units as u
from astropy import constants as c

class CodeUnits:
    """
    Class to define code units to convert physical units into simulations units.
    The user needs to provid length, mass and either time units or the gravitational constant G.
    If time is not provided, it will be calculated from the gravitational constant.
    
    """

    def __init__(self, unit_length, unit_mass, G, unit_time=None):
        self.code_length = u.def_unit('code_length', unit_length)
        self.code_mass = u.def_unit('code_mass', unit_mass)
        self.code_density = u.def_unit('code_density', unit_mass/unit_length**3)
        if unit_time is not None:
            self.code_time = u.def_unit('code_time', unit_time)
            self.G = c.G.to(self.code_length**3 / (self.code_mass * self.code_time**2)).value
        else:
            self.G = c.G.to(self.code_length**3 / (self.code_mass * u.s**2)) 
            self.G = self.G * (self.code_mass/self.code_length**3)
            self.code_time = u.def_unit('code_time', (G/self.G)**(1/2) )
            self.G = G
        self.code_velocity = self.code_length / self.code_time
        self.code_force = self.code_mass * self.code_length / self.code_time**2


    

        
