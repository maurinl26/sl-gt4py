
import numpy as np


# gridtools4py settings
backend = "numpy"  # options: "numpy", "gt:cpu_ifirst", "gt:cpu_kfirst", "gt:gpu", "dace:cpu", "dace:gpu"
backend_opts = {'verbose': True} if backend.startswith('gt') else {}
dtype = np.float64
dtype_int = np.int32
origin = (0, 0, 0)
rebuild = True

# externals = {"rd": rd, "g": g, "p_ref": p_ref, "cp": cp}
externals = {
    
}