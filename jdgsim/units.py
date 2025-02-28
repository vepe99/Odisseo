from astropy import units as u
from astropy import constants as c

class CodeUnits:

    def __init__(self, unit_length, unit_mass, G, unit_time=None):
        self.code_length = u.def_unit('code_length', unit_length)
        self.code_mass = u.def_unit('code_mass', unit_mass)
        self.code_density = u.def_unit('code_density', unit_mass/unit_length**3)
        if unit_time is not None:
            self.code_time = u.def_unit('code_time', unit_time)
        else:
            self.G = c.G.to(self.code_length**3 / (self.code_mass * u.s**2)) 
            self.G = self.G * (self.code_mass/self.code_length**3)
            self.code_time = u.def_unit('code_time', (G/self.G)**(1/2) )
        self.code_velocity = self.code_length / self.code_time


    

        
