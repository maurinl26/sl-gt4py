import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

def plot_2D_scalar_field(xcr: np.ndarray, ycr: np.ndarray, scalar_field: np.ndarray, levels: int):
    """Plot 2D scalar field

    Args:
        xc (np.ndarray): x coordinates
        yc (np.ndarray): y coordinares
        scalar_field (np.ndarray): scalar field over xc, yc
    """
    # Shape

    fig, ax = plt.subplots(nrows=1)

    # Vent vertical
    cnt = ax.contourf(
        xcr,
        ycr,
        scalar_field,
        vmin=-15,
        vmax=15,
        cmap="bwr",
        extend="neither",
        levels=levels,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(cnt, ax=ax, extend="neither")
    plt.show()
    
def plot_blossey(
    xcr: np.ndarray,
    ycr: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    tracer: np.ndarray,
    output_file: str,
):
    # Shape
    fig, ax = plt.subplots()

    # Vent
    ax.quiver(
        xcr[::3, ::3], ycr[::3, ::3], vx[::3, ::3], vy[::3, ::3],
        color="C0",
        angles="xy",
        scale_units="xy",
        scale=100,
        width=0.0005,
    )
    ax.set(xlim=(0, 1), ylim=(0, 1))

    levels = [0.05 + i * 0.1 for i in range(0, 10)]
    ax.contour(xcr, ycr, tracer, colors="black", vmin=0.05, vmax=0.95, levels=levels)

    fig.savefig(output_file)
    
def plot_lipschitz(
    xcr: np.ndarray,
    ycr: np.ndarray,
    lipschitz: np.ndarray,
    tracer: np.ndarray,
    output_file: str,
):
    # Shape
    fig, ax = plt.subplots()

    levels = [0.05 + i * 0.1 for i in range(0, 10)]
    ax.contour(xcr, ycr, tracer, colors="black", vmin=0.05, vmax=0.95, levels=levels)

    ax.contour(xcr, ycr, lipschitz, colors="blue")

    fig.savefig(output_file)
    
def plot_tracer_against_reference(
    xcr: np.ndarray,
    ycr: np.ndarray,
    tracer: np.ndarray,
    tracer_ref: float,
    e2: float,
    einf: float,
    output_file: str,
    max_cfl: float,
    dx: float    
    ):
    
    # Shape
    fig, ax = plt.subplots()

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("equal")

    levels = [0.05 + i * 0.1 for i in range(0, 10)]
    ax.contour(xcr, ycr, tracer, colors="black", vmin=0.05, vmax=0.95, levels=levels, linewidths=1)
    
    levels_ref = [0.05, 0.75]
    ax.contour(xcr, ycr, tracer_ref, colors="blue", vmin=0.05, vmax=0.95, levels=levels_ref, linewidths=1)
    
    
    ax.text(0.0 ,0.01, r" $E_2$" + f" = {e2:.03f} \n" + r" $E_{\infty}$" +f" = {einf:.03f}")
    ax.text(0.90, 0.01, f" Min = {np.min(tracer):.03f} \n Max = {np.max(tracer):.03f}")
    
    ax.set_title(f"max(CFL) = {max_cfl:.02f}")
    ax.set_ylabel(f"$\Delta$x = {dx:.04f}")

    fig.savefig(output_file)
    
    
if __name__ ==  "__main__":
    
    data_ref = "./figures/blossey/20231116_mpdata_dx_0.005_dt_0.000125/data_0.nc"
    data_4 = "./figures/blossey/20231116_mpdata_dx_0.005_dt_0.000125/data_4.nc"
    output = "./figures/blossey/20231116_mpdata_dx_0.005_dt_0.000125/blossey_ref_cfl_0.3.pdf"
    
    
    with nc.Dataset(data_ref, "r") as ds_0:
               
        xcr = ds_0["xcr"][...]
        ycr = ds_0["ycr"][...]
        
        u = ds_0["u"][...]
        v = ds_0["v"][...]
        
        nx, ny, nz = u.shape
        
        tracer_ref = ds_0["tracer"][...]
        
    with nc.Dataset(data_4, "r") as ds_4:
        
        tracer = ds_4["tracer"][...]
        
    e_inf = np.max(np.abs(tracer[:, :, 0] - tracer_ref[:, :, 0]))
    e_2 = np.sqrt((1 / (nx * ny)) * np.sum((tracer[:, :, 0] - tracer_ref[:, :, 0]) ** 2))

        
        
    plot_tracer_against_reference(
        xcr,
        ycr, 
        tracer[:, :, 0],
        tracer_ref[:, :, 0],
        e_2,
        e_inf,
        output
    )
    
    
    

    

    
    
    