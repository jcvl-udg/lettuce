# Se pretende una simulacion de Vortice Taylor-Green con 10,000 pasos
# mediante el codigo siguiente se importan las librerias necesarias
# se inicializa entonces el contexto,flujo y colisiones.
import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch


# Se elige el device correspondiente a la simulacion [cpu] o CUDA si esta disponible
context = lt.Context(device=torch.device('cuda:0') if torch.cuda
                     .is_available() else torch.device('cpu'),
                     dtype=torch.float32)
flow = lt.TaylorGreenVortex(resolution=256, reynolds_number=100, 
                            mach_number=0.05, stencil=lt.D2Q9,
                            context=context)
collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)
energyreporter = lt.ObservableReporter(lt.IncompressibleKineticEnergy(flow), interval=1000, out=None)
simulation = lt.Simulation(flow=flow, collision=collision, reporter=[energyreporter])

# Reporters will grab the results in between simulation steps (see reporters.py and simulation.py)
# Output: Column 1: simulation steps, Column 2: time in LU, Column 3: kinetic energy in PU
# Output: separate VTK-file with ux,uy,(uz) and p for every 100. time step in ./output

mlups = simulation(10000)
print("Performance in MLUPS:", mlups)

#Aqui termina el trabajo de simulacion, pasaremos al Post Processing

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