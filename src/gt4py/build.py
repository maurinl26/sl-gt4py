import numpy as np

dtype = np.float64
dtype_int = np.int32

backend = "gt:cpu_ifirst"
backend_opts = {"verbose": True} if backend.startswith("gt") else {}
origin = (0, 0, 0)
rebuild = False

# Functions
externals = {
    
}


update_periodic_layers_using_copy_stencil = False
if backend == "dace:gpu":  # initial experiments showed higher performance
    update_periodic_layers_using_copy_stencil = True
