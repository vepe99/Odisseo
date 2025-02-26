from astropy import units as u
from astropy import constants as c

class CodeUnits:

    def __init__(self, unit_length, unit_mass, unit_time):
        self.code_length = u.def_unit('code_length', unit_length)
        self.code_mass = u.def_unit('code_mass', unit_mass)
        self.code_time = u.def_unit('code_unit', unit_time)
        

    

        
