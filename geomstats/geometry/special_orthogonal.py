from geomstats.geometry.general_linear import GeneralLinear

class SpecialOrthogonal(GeneralLinear): 
    
    @classmethod
    def inv(cls, point):
        return cls.transpose(point)
