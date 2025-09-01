"""
This file showcases the simplicity of the lettuce code.
The following code will run a two-dimensional Taylor-Green vortex on GPU.
"""

#set KMP_DUPLICATE_LIB_OK=TRUE {?????}

import torch
import lettuce as lt

flow = lt.TaylorGreenVortex(
    lt.Context(dtype=torch.float64),  # for running on cpu: device='cpu'
    resolution=128,
    reynolds_number=100,
    mach_number=0.05,
    stencil=lt.D2Q9
)

energyreporter = lt.ObservableReporter(lt.IncompressibleKineticEnergy(flow), interval=1000, out=None)

simulation = lt.Simulation(
    flow=flow,
    collision=lt.BGKCollision(tau=flow.units.relaxation_parameter_lu),
    reporter=[energyreporter])
mlups = simulation(num_steps=1000)

print("Performance in MLUPS:", mlups)

import matplotlib.pyplot as plt
import numpy as np

# Tomamos de la simulacion la Energia Kynetic desde el Reporter
energy = np.array(simulation.reporter[0].out)
print(energy.shape)
plt.figure(1)
plt.plot(energy[:,1],energy[:,2])
plt.title('Kinetic energy')
plt.xlabel('Time')
plt.ylabel('Energy in physical units')

# Velocity
# We calculate the speed in Lettuce units depending on the last 'f'. 
# Then we convert this velocity into physical units. 
# For further investigations the tensor must be converted into a Numpy-Array. 
# The norm of the fractions in x and y direction is plotted afterwards.

u = flow.u_pu.cpu().numpy()
u_norm = np.linalg.norm(u,axis=0)
plt.figure(2)
plt.imshow(u_norm)
#plt.show()

plt.show()