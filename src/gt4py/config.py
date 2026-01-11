from dataclasses import dataclass, field
from sl_gt4py.build import dtype

import numpy as np

@dataclass
class Config:
    
    dt: dtype
    dth: dtype = field(init=False)
    
    xmin: dtype
    xmax: dtype
    nx: int
        
    ymin: dtype
    ymax: dtype
    ny: int
    
    nz: int
    
    bcx_kind: int
    bcy_kind: int
    
    filter: bool
    
    model_starttime: float 
    model_endtime: float
    
    def __post_init__(self):
        
        self.dth = self.dt / 2
        self.indices()
        self.coordinates()
                
    def indices(self):
        
        self.i_indices = np.arange(self.nx)
        self.j_indices = np.arange(self.ny)
        
        I, J = np.meshgrid(self.i_indices, self.j_indices)
        self.I, self.J = I.T, J.T
        
    def coordinates(self):
    
        self.xc = np.linspace(self.xmin, self.xmax, self.nx)
        self.yc = np.linspace(self.ymin, self.ymax, self.ny)
        
        self.dx = self.xc[1] - self.xc[0]
        self.dy = self.yc[1] - self.yc[0]
        
        xcr, ycr = np.meshgrid(self.xc, self.yc)
        self.xcr, self.ycr = xcr.T, ycr.T 
                
        
if __name__ == "__main__":
    
    
    config = Config(
        dt=1,
        xmin=0,
        xmax=100,
        ymin=0,
        ymax=100,
        nx=50,
        ny=50,
        bcx_kind=0,
        bcy_kind=0
    )
    
    print(config.xc[0], config.xc[-1], len(config.xc))
    print(config.yc[0], config.yc[-1], len(config.yc))
    
    print(config.I[0, 0], config.I[-1, 0], config.I.shape)
    print(config.J[0, 0], config.J[0, -1], config.J.shape)