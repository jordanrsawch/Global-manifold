from isingFuncts import *
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
import sys
from scipy.integrate import solve_ivp

"""
Takes protocol in terms of state energies.
"""
PIDX = int(sys.argv[1]) # takes integer index
folder = " " # import folder 
data = np.load(folder + " ") # import state energy trajectories
V = data[PIDX] # extract desired state energy trajectory


params = setSpinSystem(2, 2, 1, 1)
works = []
dts = []
taus = np.logspace(-2, 3, 100)
M = int(1E4)
for j, tau in enumerate(taus):
    tt = np.linspace(0, tau, M)
    dt = tt[1] - tt[0]
    Ws = np.array([rateMatrix(V[i], params) for i in range(len(V))])
    W = interp1d(np.linspace(0, tau, len(Ws)), Ws, axis = 0)
    def master_equation(t, p):
        return W(t) @ p
    p0 = eqbm(V[0])
    sol = solve_ivp(master_equation, [0, tau], p0, method='DOP853', atol=1e-12, rtol=1e-10) 
    p = interp1d(sol["t"], sol["y"], axis = 1)
    pEval = p(tt).T
    dp = np.zeros_like(pEval)
    Vt = interp1d(np.linspace(0, tau, len(V)), V, axis = 0)
    for i in range(M):
        dp[i] = pEval[i] - eqbm(Vt(tt[i]))
    work = integrate.simpson(np.einsum('ij, ij -> i', gradient(dt, Vt(tt)), dp), tt)
    works.append(work)

np.save(folder + f"exactWorks_{PIDX}.npy", works, p)
