# imports
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import random
from numba import jit, cuda

@jit(target_backend='cuda')                         

# defining functions
def Positives(bar,x,y,z):
    Pbars = [bar, x, y, z]
    return Pbars
def Negatives(bar,x,y,z):
    Nbars = [-1*bar, x, y, z]
    return Nbars
def Force(q1,q2,r):
    F = abs((q1*q2)/(r**2))
    return F
def Distance(x1,x2):
    return x1-x2

# creating bars
PositivesState = []
NegativesState = []
Pnumber = 30
Nnumber = 30
scale = 100
power = 100

for i in range(Pnumber):
    PositivesState.append(Positives(np.random.rand()*power,
                                    random.choice([-1, 1]) * np.random.rand()*scale,
                                    random.choice([-1, 1]) * np.random.rand()*scale,
                                    random.choice([-1, 1]) * np.random.rand()*scale))
for i in range(Nnumber):
    NegativesState.append(Negatives(np.random.rand()*power,
                                    random.choice([-1, 1]) * np.random.rand()*scale,
                                    random.choice([-1, 1]) * np.random.rand()*scale,
                                    random.choice([-1, 1]) * np.random.rand()*scale))

# finding forces between negatives bars
FNNXbars = np.zeros((Nnumber,Nnumber))
FNNYbars = np.zeros((Nnumber,Nnumber))
FNNZbars = np.zeros((Nnumber,Nnumber))
FNPXbars = np.zeros((Nnumber,Pnumber))
FNPYbars = np.zeros((Nnumber,Pnumber))
FNPZbars = np.zeros((Nnumber,Pnumber))
FPPXbars = np.zeros((Pnumber,Pnumber))
FPPYbars = np.zeros((Pnumber,Pnumber))
FPPZbars = np.zeros((Pnumber,Pnumber))
FPNXbars = np.zeros((Pnumber,Nnumber))
FPNYbars = np.zeros((Pnumber,Nnumber))
FPNZbars = np.zeros((Pnumber,Nnumber))

for i in range(Nnumber):
    for j in range(Nnumber):
        if i == j :
            continue
        else:
            FNNXbars[i][j] += Force(NegativesState[i][0],
                                    NegativesState[j][0],
                                    Distance(NegativesState[i][1],
                                    NegativesState[j][1]))
            FNNYbars[i][j] += Force(NegativesState[i][0],
                                    NegativesState[j][0],
                                    Distance(NegativesState[i][2],
                                    NegativesState[j][2]))
            FNNZbars[i][j] += Force(NegativesState[i][0],
                                    NegativesState[j][0],
                                    Distance(NegativesState[i][3],
                                    NegativesState[j][3]))                                
    for j in range(Pnumber):
        FNPXbars[i][j] += Force(NegativesState[i][0],
                                PositivesState[j][0],
                                Distance(NegativesState[i][1],
                                PositivesState[j][1]))
        FNPYbars[i][j] += Force(NegativesState[i][0],
                                PositivesState[j][0],
                                Distance(NegativesState[i][2],
                                PositivesState[j][2]))
        FNPZbars[i][j] += Force(NegativesState[i][0],
                                PositivesState[j][0],
                                Distance(NegativesState[i][3],
                                PositivesState[j][3],))
for i in range(Pnumber):
    for j in range(Pnumber):
        if i == j :
            continue
        else:
            FPPXbars[i][j] += Force(PositivesState[i][0],
                                    PositivesState[j][0],
                                    Distance(PositivesState[i][1],
                                    PositivesState[j][1]))
            FPPYbars[i][j] += Force(PositivesState[i][0],
                                    PositivesState[j][0],
                                    Distance(PositivesState[i][2],
                                    PositivesState[j][2]))
            FPPZbars[i][j] += Force(PositivesState[i][0],
                                    PositivesState[j][0],
                                    Distance(PositivesState[i][3],
                                    PositivesState[j][3]))       
    for j in range(Nnumber):
        FPNXbars[i][j] += Force(PositivesState[i][0],
                                NegativesState[j][0],
                                Distance(PositivesState[i][1],
                                NegativesState[j][1]))
        FPNYbars[i][j] += Force(PositivesState[i][0],
                                NegativesState[j][0],
                                Distance(PositivesState[i][2],
                                NegativesState[j][2]))
        FPNZbars[i][j] += Force(PositivesState[i][0],
                                NegativesState[j][0],
                                Distance(PositivesState[i][3],
                                NegativesState[j][3],))
            
# plotting steady state
Px, Py, Pz = [], [], []
Nx, Ny, Nz = [], [], []
for i in range(Pnumber):
    Px.append(PositivesState[i][1])
    Py.append(PositivesState[i][2])
    Pz.append(PositivesState[i][3])
for i in range(Nnumber):
    Nx.append(NegativesState[i][1])
    Ny.append(NegativesState[i][2])
    Nz.append(NegativesState[i][3])

#plt.ion()
fig = plt.figure()
# plt.set_cmap('binary')
ax = fig.add_subplot(111,projection='3d')
ax = Axes3D(fig)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-1.1* scale, 1.1 * scale)
ax.set_ylim(-1.1* scale, 1.1 * scale)
ax.set_zlim(-1.1* scale, 1.1 * scale)
ccc = []
asd = []
for i in range(Nnumber):
    ccc.append(1)
for i in range(Pnumber):
    ccc.append(2)
for i in range(Nnumber+Pnumber):
    temp = -1 if np.random.rand() <= 0.5 else 1
    asd.append(scale*temp)
sc = ax.scatter(asd,asd,asd,c=ccc)
'''
for i in range(10000):
    x = random.choice([-1, 1]) * np.random.rand()
    y = random.choice([-1, 1]) * np.random.rand()
    z = random.choice([-1, 1]) * np.random.rand()
    if 0.99 <= (x**2+y**2+z**2) <= 1.01:
        ax.scatter(x * scale, y * scale, z * scale, c=['k'])
'''
fig.show()
plt.axis("off")

t = 0
loopTime = time.time()
while True:
    #ax.view_init(elev = (time.time() - loopTime)*1e1/2, azim = (time.time() - loopTime)*1e1/2)
    plt.pause(0.01)
    delta_Nx = -(1/2)*np.sum(FNPXbars.T, axis=0)*(t**2) + (1/2)*np.sum(FNNXbars.T, axis=0)*(t**2)
    delta_Ny = -(1/2)*np.sum(FNPYbars.T, axis=0)*(t**2) + (1/2)*np.sum(FNNYbars.T, axis=0)*(t**2)
    delta_Nz = -(1/2)*np.sum(FNPZbars.T, axis=0)*(t**2) + (1/2)*np.sum(FNNZbars.T, axis=0)*(t**2)
    delta_Px = -(1/2)*np.sum(FPNXbars.T, axis=0)*(t**2) + (1/2)*np.sum(FPPXbars.T, axis=0)*(t**2)
    delta_Py = -(1/2)*np.sum(FPNYbars.T, axis=0)*(t**2) + (1/2)*np.sum(FPPYbars.T, axis=0)*(t**2)
    delta_Pz = -(1/2)*np.sum(FPNZbars.T, axis=0)*(t**2) + (1/2)*np.sum(FPPZbars.T, axis=0)*(t**2)
    Nx = Nx + delta_Nx
    Ny = Ny + delta_Ny
    Nz = Nz + delta_Nz
    Px = Px + delta_Px
    Py = Py + delta_Py
    Pz = Pz + delta_Pz
    
    for k in range(len(Nx)):
        if Nx[k]**2 + Ny[k]**2 + Nz[k]**2 >= scale:
            OP = [Nx[k]-delta_Nx[k], Ny[k]-delta_Ny[k], Nz[k]-delta_Nz[k]]
            OB = [Nx[k], Ny[k], Nz[k]]
            ON = list(np.dot(OP, OB)/np.dot(OB, OB) * np.array(OB))
            PN = list(np.array(ON) - np.array(OP))
            S = list(np.array(OP) + 2 * np.array(PN))
            Nx[k] = S[0]
            Ny[k] = S[1]
            Nz[k] = S[2]

    for k in range(len(Px)):
        if Px[k]**2 + Py[k]**2 + Pz[k]**2 >= scale:
            OP = [Px[k]-delta_Px[k], Py[k]-delta_Py[k], Pz[k]-delta_Pz[k]]
            OB = [Px[k], Py[k], Pz[k]]
            ON = list(np.dot(OP, OB)/np.dot(OB, OB) * np.array(OB))
            PN = list(np.array(ON) - np.array(OP))
            S = list(np.array(OP) + 2 * np.array(PN))
            Px[k] = S[0]
            Py[k] = S[1]
            Pz[k] = S[2]
            
    sc._offsets3d = (np.append(Nx,Px),np.append(Ny,Py),np.append(Nz,Pz))
    t += (time.time() - loopTime) * 1e-2
    plt.draw()
    for i in range(Nnumber):
        for j in range(Nnumber):
            if i == j :
                continue
            else:
                FNNXbars[i][j] = Force(NegativesState[i][0],
                                       NegativesState[j][0],
                                       Distance(Nx[i], Nx[j]))
                FNNYbars[i][j] = Force(NegativesState[i][0],
                                       NegativesState[j][0],
                                       Distance(Ny[i], Ny[j]))
                FNNZbars[i][j] = Force(NegativesState[i][0],
                                       NegativesState[j][0],
                                       Distance(Nz[i], Nz[j]))                                
        for j in range(Pnumber):
            FNPXbars[i][j] = Force(NegativesState[i][0],
                                   PositivesState[j][0],
                                   Distance(Nx[i], Px[j]))
            FNPYbars[i][j] = Force(NegativesState[i][0],
                                   PositivesState[j][0],
                                   Distance(Ny[i], Py[j]))
            FNPZbars[i][j] = Force(NegativesState[i][0],
                                   PositivesState[j][0],
                                   Distance(Nz[i], Pz[j]))
            
    for i in range(Pnumber):
        for j in range(Pnumber):
            if i == j :
                continue
            else:
                FPPXbars[i][j] = Force(PositivesState[i][0],
                                       PositivesState[j][0],
                                       Distance(Px[i], Px[j]))
                FPPYbars[i][j] = Force(PositivesState[i][0],
                                       PositivesState[j][0],
                                       Distance(Py[i], Py[j]))
                FPPZbars[i][j] = Force(PositivesState[i][0],
                                       PositivesState[j][0],
                                       Distance(Pz[i], Pz[j]))       
        for j in range(Nnumber):
            FPNXbars[i][j] = Force(PositivesState[i][0],
                                   NegativesState[j][0],
                                   Distance(Px[i], Nx[j]))
            FPNYbars[i][j] = Force(PositivesState[i][0],
                                   NegativesState[j][0],
                                   Distance(Py[i], Ny[j]))
            FPNZbars[i][j] = Force(PositivesState[i][0],
                                   NegativesState[j][0],
                                   Distance(Pz[i], Nz[j]))      