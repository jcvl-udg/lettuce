# https://arxiv.org/pdf/2106.12929
# Lettuce: PyTorch-based Lattice Boltzmann Framework

####################### Code Example ########################
# Fig. 1. Q-criterion isosurfaces of a three-dimensional Taylor-Green Vortex at time step
# t=5000, 7000 and 10000 colored by streamwise velocity. The Reynolds number and
# grid resolution were 1600 and 2563, respectively.


# https://github.com/lettucecfd/lettuce/blob/c98365f4ecc91a1dd4d87ac066b5406a59ead6d0/native_cuda_synopsis.md?plain=1#L14
# 1. `lt.Lattice` is deprecated.
# The `device` and `dtype` fields moved to `lt.Context`. The `Stencil` object or class (either works) is passed to `Flow.__init__`.
# 2. `lt.Streaming` is deprecated.
# Streaming is now always handled as defined in `lt.StandardStreaming`.
# 3. `lt.Stencil` is stored in a field of `lt.Flow` and `D` is now lower case.
# So, the Dimensions can be accessed via `flow.stencil.d` instead of `flow.units.lattice.D`.

import lettuce as lt
import torch
import numpy as np
import matplotlib.pyplot as plt

# (stencil=lt.D3Q27, device='cuda')
lattice = lt.Context(torch.device('cpu'), dtype=torch.float32) 

flow = lt.TaylorGreenVortex(context= lattice,
    resolution=100,
    reynolds_number=1600,
    mach_number=0.075,
    stencil=lt.D3Q27
)

# streaming = lt.StandardStreaming(lattice)
collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)
simulation = lt.Simulation(flow, collision, [])

energyreporter = lt.ObservableReporter(lt.IncompressibleKineticEnergy(flow), interval=5, out=None)
# simulation.reporter.append(VTKreport)
VtkRep = lt.VTKReporter(interval=25, filename_base='./data/tgv')

simulation.reporter.append(energyreporter)
simulation.reporter.append(VtkRep)

# simulation = lt.Simulation(flow=flow, collision= collision)
# ---------- Simulate until time = tend (PU) -------------
tend = 10  # [PU]
nend = int(simulation.flow.units.convert_time_to_lu(30))  # [LU]
print(f"Simulating {nend} steps! Maybe drink some water in the meantime.")
# runs simulation, but also returns overall performance in MLUPS (million
# lattice units per second)
print("MLUPS: ", simulation(nend))
#simulation.step(num_steps=100)


#Lettuce provides various observables that can be reported during the simulation (e.g. kinetic energy, enstrophy, energy spectrum). These observables can
#be added easily to the Simulation class and exported for further analysis as
#follows:

# ---------- Plot kinetic energy over time (PU) -------------
# grab output of kinetic energy reporter
E = np.asarray(energyreporter.out)
# normalize to size of grid, not always necessary
E[:, 1] = E[:, 1] / (2 * np.pi) ** 3
# save kinetic energy values for later use
np.save("data/TGV3DoutRes" + str(256) + "E", E)
fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)
plt.xlabel('Time in physical units')
plt.ylabel('Kinetic energy in physical units')
ax1.plot(simulation.flow.units.convert_time_to_pu(range(0, E.shape[0])),
         E[:, 1])

# ---------- Plot magnitude of speed in slice of 3D volume -------------
# grab u in PU
u = flow.u_pu # Shape: [3, nx, ny, nz] - 3D velocity field
rho = flow.rho_pu  # Shape: [nx, ny, nz] - 3D density field
#---------------------------------2D slice----------------------------------------------------#
# # [direction of u: Y, X, Z] (due to ij indexing)
# uMagnitude = torch.pow(torch.pow(u[0, :, :, :], 2)
#                        + torch.pow(u[1, :, :, :], 2)
#                        + torch.pow(u[2, :, :, :], 2), 0.5)
#----------------------------------2D slice---------------------------------------------------#
# # select slice to plot
# uMagnitude = uMagnitude[:, :, round(0.1 * 256)]
# # send selected slice to CPU und numpy, to be able to plot it via matplotlib
# uMagnitude = uMagnitude.cpu().numpy()
# ax2 = plt.subplot(1, 2, 2)
# ax2.matshow(uMagnitude)
# plt.tight_layout()

import plotly.graph_objects as go  # For interactive 3D

# Create iso-surface of velocity magnitude
u_mag = torch.sqrt(u[0]**2 + u[1]**2 + u[2]**2).cpu().numpy()
# Get the grid dimensions
nx, ny, nz = u_mag.shape

# Create coordinate arrays - CORRECT WAY
X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]

# Use plotly for interactive 3D
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(), 
    y=Y.flatten(), 
    z=Z.flatten(),
    value=u_mag.flatten(),
    isomin=0.5 * u_mag.max(),
    isomax=0.9 * u_mag.max(),
    caps=dict(x_show=False, y_show=False, z_show=False),
    surface_count=3,  # Number of iso-surfaces
    colorscale='Viridis'
))

fig.update_layout(
    title='3D Velocity Magnitude Iso-surface',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y', 
        zaxis_title='Z',
        aspectmode='data'  # Maintain aspect ratio
    )
)

# For plotly, use write_html instead of savefig
fig.write_html('data/tgv3d-3d-isosurface_cqnd.html')
fig.show()

# plt.savefig('data/tgv3d-output.pdf')