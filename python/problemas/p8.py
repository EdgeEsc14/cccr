import numpy as np
from scipy.linalg import cho_factor, cho_solve

D, v = 1.0, 1.0
h = 1/20
n = 19
x = h*np.arange(1, n+1)
T = 0.1

A1 = (D/h**2)*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
I = np.eye(n)

mu = 2/h**2*(1 - np.cos(np.pi*h))
exacta = np.exp(-v*mu*T)*np.sin(np.pi*x)

errores = []
for dt in [0.02, 0.01, 0.005, 0.0025]:
    K = round(T/dt)
    Bmas = I + v*dt/2*A1
    Bmenos = I - v*dt/2*A1
    G = cho_factor(Bmas)
    phi = np.sin(np.pi*x)
    for k in range(K):
        phi = cho_solve(G, Bmenos @ phi)
    eps = np.max(np.abs(exacta - phi))
    errores.append(eps)
    print(f"dt = {dt:<7} K = {K:3d}  error = {eps:.6e}")

for m in range(3):
    R = errores[m]/errores[m+1]
    print(f"R = {R:.6f}   OCG = {np.log(R)/np.log(2):.6f}")
