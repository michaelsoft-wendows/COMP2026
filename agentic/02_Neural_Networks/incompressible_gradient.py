import numpy as np
import matplotlib.pyplot as plt

N = 100

# Domain: NxN interior points plus a one-cell boundary ring on each side.
x = np.arange(0, N + 2)
y = np.arange(0, N + 2)
yc, xc = np.meshgrid(x, y)

# Velocity vector field on the domain, one component per direction.
Ux = np.zeros((N + 2, N + 2))
Uy = np.zeros((N + 2, N + 2))

def div(i,j):
    return Ux[i+1,j]-Ux[i-1,j]+Uy[i,j+1]-Uy[i,j-1]

def loss():
    L = 0
    for i in range(1,N+1):
        for j in range(1,N+1):
            L+=div(i,j)**2
    return L