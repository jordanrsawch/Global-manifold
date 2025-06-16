import numpy as np
from itertools import product, combinations
import os
import sys
import scipy.integrate as integrate
from scipy.interpolate import interp1d

CSETIDX = int(sys.argv[1]) # Integer b/w 0 and 199 (incl.)
folderName = "" # Folder to save data to

# Grid dimensions
Nx = 2; Ny = 2; Nz = 1  
dims = np.array([Nx, Ny, Nz])
N_spins = Nx*Ny*Nz

# States
N_states = 2**N_spins
spinvals = np.array([-1, 1])
states = np.array(list(product(spinvals, repeat=N_spins)))

# Parameters
beta = 1.0                            
duration = 1.0                          

# Calculate x, y, z coordinates from the spin index
def index_to_coords(n):
    x = n // (Ny * Nz)          
    y = (n // Nz) % Ny          
    z = n % Nz                 
    return np.array([x, y, z])

# State adjacency matrix 
differences = states[:, np.newaxis] != states[np.newaxis, :]
diff_count = differences.sum(axis=2)
stateAdj = (diff_count == 1).astype(int)

# Two-spin base couplings
Jcombs = list(combinations(range(N_spins), 2))
N_Jcombs = int((N_spins * (N_spins - 1)) / 2)

def getBaseCoupling(J):
    baseCoupling = np.zeros(N_Jcombs)
    # Spin adjacency matrix
    spinAdj = np.zeros((N_spins, N_spins))
    for i in range(N_spins):
        for j in range(i+1, N_spins):
            coords1 = index_to_coords(i)
            coords2 = index_to_coords(j)
            if np.sum(np.abs(coords1 - coords2)) == 1:
                spinAdj[i, j] = 1
                spinAdj[j, i] = 1
    for i, comb in enumerate(Jcombs):
        baseCoupling[i] = J*spinAdj[comb[0], comb[1]]
    return baseCoupling
baseCoupling = getBaseCoupling(1) 

# State energy
def couplingEnergy(state):
    return -state[0]*state[1] - state[0]*state[2] - state[1]*state[3] - state[2]*state[3]

def egy(state_idx, v):
    state = states[state_idx]
    if baseCouplingIsControlled:
        return np.dot(Jacobian[state_idx], v)
    else:
        return np.dot(Jacobian[state_idx], v) + couplingEnergy(state)

# Friction tensor and derivative
def get_friction_and_derivative(v):
    # Get the full energy vector
    energies = energies = np.array([egy(i, v) for i in range(N_states)])
    # Calculate the equilibrium distribution
    pi = np.exp(-beta*energies)
    pi /= np.sum(pi)
    # Get the eqbm projection matrix
    oV = np.ones(N_states)
    Proj = np.outer(pi, oV)
    # Get the derivatives of the equilibrium distribution w.r.t. the control parameters
    pex = pi[:, None]
    eqDerivs = - beta * (pex * (np.eye(N_states) - pex.T)) @ Jacobian
    # Get the rate matrix
    dE = energies[:, None] - energies
    W = np.where(stateAdj == 1, 1 / (1 + np.exp(beta * dE)), 0)
    W -= np.diag(np.sum(W, axis=0))
    # Get the friction tensor
    try:
        Wdi = np.linalg.pinv(W - Proj)
    except:
        Wdi = np.linalg.pinv(W - Proj + 1e-10*np.eye(N_states))
    fric = - Jacobian.T @ np.multiply(Proj + Wdi, pi) @ Jacobian 
    # Get the derivatives of the rate matrix w.r.t. the control parameters
    dW = -beta * np.multiply(W, W.T)[:, :, np.newaxis] * (Jacobian[:, np.newaxis, :] - Jacobian[np.newaxis, :, :])
    dW -= np.einsum('ijk,jk->ijk', np.eye(dW.shape[0])[:, :, None], np.sum(dW, axis=0))
    dW = np.array([dW[:,:,i] for i in range(numCPs)])
    outers = np.array([np.outer(eqDerivs[:,i], pi) + np.outer(pi, eqDerivs[:,i]) for i in range(numCPs)])
    Wdderiv = -np.array([-Wdi @ (np.outer(eqDerivs[:,i], np.ones(N_states)) - dW[i]) @ Wdi for i in range(numCPs)])
    # Get the friction tensor derivatives
    term1 = -outers
    term2 = np.array([np.multiply(Wdderiv[i], pi) + np.multiply(Wdi, eqDerivs[:,i]) for i in range(numCPs)])
    fricDeriv = Jacobian.T @ (term1 - term2) @ Jacobian
    return fric, fricDeriv

# Update step in relaxation method
def evolve_string(protocol, M, delta_r, delta_t):
    R = protocol.shape[0]
    new_protocol = np.zeros_like(protocol)
    velocity = np.gradient(protocol, delta_t, axis=0)
    frics = np.zeros((R, numCPs, numCPs))
    finvs = np.zeros((R, numCPs, numCPs))
    fdvs = np.zeros((R, numCPs, numCPs, numCPs))
    for j in range(numCPs):
        rhs = np.zeros((R, 1))
        for k in range(R):
            if j == 0:
                frics[k], fdvs[k] = get_friction_and_derivative(protocol[k])
                finvs[k] = np.linalg.pinv(frics[k]) 
            to_add = 0.0
            for l in range(numCPs):
                for m in range(numCPs):
                    for n in range(numCPs):
                        to_add += finvs[k][j,l] * velocity[k, n] * velocity[k, m] * (fdvs[k][m, l, n] - 0.5 * fdvs[k][l, m, n])
            rhs[k,0] = protocol[k, j] + delta_r*to_add
        rhs[0,0] = u0[j]
        rhs[R-1,0] = u1[j]
        new_protocol[:,j] = np.linalg.lstsq(M, rhs, rcond=None)[0].flatten()
    return new_protocol, frics, velocity 

# Compute power and work
def power_and_work(velocity, frics, delta_t):
    powers = np.einsum('ri,rij,rj->r', velocity, frics, velocity)
    return powers, integrate.trapezoid(powers, dx=delta_t)

# Convert b/w 15 param full-control vectors and partial-control vectors
def reducedVec(fullVector, numCPs, pIdxs):
    reducedVector = np.zeros(numCPs)
    for j in range(len(fullVector)):
        if pIdxs[j] != None:
            reducedVector[pIdxs[j]] = fullVector[j]
    return reducedVector
def fullVec(reducedVector):
    fullVector = np.zeros(15)
    fullVector[4:10] = baseCoupling
    for j in range(15):
        if pIdxs[j] != None:
            fullVector[j] = reducedVector[pIdxs[j]]
    return fullVector

# Get number of control parameters, parmeter indices, Jacobian matrix,
# controllability of base coupling, and boundary conditions in partial-control vectors
def setUpCset(cset):
    ncp = len(cset)
    def paramIdxs():
        params = np.empty(15, dtype=object)
        idx = 0
        for i, block in enumerate(cset):
            params[block] = idx 
            idx += 1
        return params
    pIdxs = paramIdxs()
    def getJacobian():
        Jacobian = np.zeros((N_states, ncp))
        for i in range(N_states):
            state = states[i]
            for j in range(ncp):
                cs = cset[j]
                sset = [pmap[c] for c in cs]
                for s in sset:
                    if len(s) == 1:
                        Jacobian[i, j] += state[s[0]]
                    elif len(s) == 2:
                        Jacobian[i, j] += state[s[0]]*state[s[1]]
                    elif len(s) == 3:
                        Jacobian[i, j] += state[s[0]]*state[s[1]]*state[s[2]]
                    elif len(s) == 4:
                        Jacobian[i, j] += state[s[0]]*state[s[1]]*state[s[2]]*state[s[3]]
        return -Jacobian
    Jacobian = getJacobian()
    baseCouplingIsControlled = cset.count([4,5,8,9]) > 0 or cset.count([4]) > 0
    # Initial and final points
    u0 = reducedVec(np.array([-1, -1, -1, -1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]), ncp, pIdxs)
    u1 = reducedVec(np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]), ncp, pIdxs)
    return ncp, pIdxs, Jacobian, baseCouplingIsControlled, u0, u1

"""
Indexing of control vectors:
- 0   <-->  h_0
- 1   <-->  h_1
- 2   <-->  h_2
- 3   <-->  h_3
- 4   <-->  J_{0,1}
- 5   <-->  J_{0,2}
- 6   <-->  J_{0,3}
- 7   <-->  J_{1,2}
- 8   <-->  J_{1,3}
- 9   <-->  J_{2,3}
- 10  <-->  K_{0,1,2}
- 11  <-->  K_{0,1,3}
- 12  <-->  K_{0,2,3}
- 13  <-->  K_{1,2,3}
- 14  <-->  L
"""
hSet = [0,1,2,3]
jSet = [4,5,6,7,8,9]
kSet = [10,11,12,13]
lSet = [14]

# Map between spin indices and couplings
pmap = [[0],[1],[2],[3],[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0,1,2],[0,1,3],[0,2,3],[1,2,3],[0,1,2,3]]

# Control subsets
hOpts = [[[0,1,2,3]], [[0,1], [2,3]], [[0,3], [1,2]], [[0],[1],[2],[3]]]
JOpts = [[], [[4,5,8,9]], [[6,7]], [[4,5,8,9], [6,7]] [[4],[5],[6],[7],[8],[9]]]
KOpts = [[], [[10,11], [12,13]], [[10,13],[11,12]], [[10],[11],[12],[13]]]
LOpts = [[], [[14]]]
controlSets = []
for h in hOpts:
    for J in JOpts:
        for K in KOpts:
            for L in LOpts:
                controlSets.append(h + J + K + L)

# Get control set
control_set = controlSets[CSETIDX]
numCPs, pIdxs, Jacobian, baseCouplingIsControlled, u0, u1 = setUpCset(control_set)

# Set base parameters 
beta = 1                                # Inverse temperature
duration = 1.0                          # Protocol duration (arb.)
dr_scaling = 0.75                       # Scaling of learning rate with time step. Lower values give more stability
threshold = 1E-7                        # Threshold for increasing the number of points
maxiter = int(1E6)                      # Maximum number of iterations
check_step = 1000                       # Check for convergence every check_step iterations
alpha = 0.1                             # Initial curvature of the protocol 
Rvals = [51, 101, 201, 401, 801]        # Number of points in the protocol

# Initial and final points
u0 = reducedVec(np.array([-1, -1, -1, -1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]))
u1 = reducedVec(np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]))


# Initialize protocol
def initialize(R, alpha): 
    initstring = np.zeros((R, numCPs))
    for i in range(numCPs):
        initstring[:,i] = np.linspace(u0[i],u1[i],R) + alpha * np.random.normal(loc=0,scale=2)*(np.ones((R))-(np.linspace(-1,1,R))**2)
    return initstring


# Relaxation method
def relaxation_method(Rvals, dr_scaling, duration, threshold, maxiter, check_step, alpha, initProt = None):
    # Initialize the protocol and create an array for the calulated works
    if initProt is None:
        prot = initialize(Rvals[0], alpha)
    else:
        Rinit = initProt.shape[0]
        RinitIdx = Rvals.index(Rinit)
        Rvals = Rvals[RinitIdx:]
        prot = initProt

    works = []

    # Evolve the string, interpolating the protocol to a larger number of points once the threshold is crossed
    for k, R in enumerate(Rvals):
        delta_t = duration / (R - 1)            # Time step
        delta_r = dr_scaling * (delta_t)**2     # Learning rate

        # Set up the implicit derivative matrix
        M = np.zeros((R, R))
        for i in range(1, R-1):
            M[i, i-1] = -delta_r/(delta_t**2)
            M[i, i] = 1 + 2*delta_r/(delta_t**2)
            M[i, i+1] = -delta_r/(delta_t**2)
        M[0, 0] = 1
        M[R-1, R-1] = 1

        # Begin iterations
        for i in range(maxiter):
            # Evolve the string
            prot, frics, velocity = evolve_string(prot, M, delta_r, delta_t)

            # Calculate the power and work
            power, work = power_and_work(velocity, frics, delta_t)
            works.append(work)

            # Save and check for convergence
            if i % check_step == 0:
                # Save the protocol and work to the main file
                np.save(os.path.join(folderName, f'protocol.npy'), prot)   
                np.save(os.path.join(folderName, f'works.npy'), works)   
                # Check for convergence 
                if i > 0 and np.abs(works[-1] - works[-1 -check_step]) < threshold and k < len(Rvals) - 1:
                    ttold = np.linspace(0, duration, Rvals[k])
                    ttnew = np.linspace(0, duration, Rvals[k+1])
                    # Save the final protocol for this section
                    np.save(os.path.join(folderName, f'protocol_{R}.npy'), prot)
                    # Interpolate the protocol to the new number of points
                    prot = interp1d(ttold, prot, axis=0)(ttnew)
                    break

        if i == maxiter - 1:
            # Save the final protocol for this section
            np.save(os.path.join(folderName, f'protocol_{R}.npy'), prot)
            if k < len(Rvals) - 1:
                # Interpolate the protocol to the new number of points
                ttold = np.linspace(0, duration, Rvals[k])
                ttnew = np.linspace(0, duration, Rvals[k+1])
                prot = interp1d(ttold, prot, axis=0)(ttnew)


relaxation_method(Rvals, dr_scaling, duration, threshold, maxiter, check_step, alpha)