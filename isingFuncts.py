# Base packages
import numpy as np
from itertools import product, combinations, chain

def setSpinSystem(Nx, Ny, Nz, J = 1):
    """
    Inputs:
        Nx (int): Number of spins in the x-direction.
        Ny (int): Number of spins in the y-direction.
        Nz (int): Number of spins in the z-direction.
        J (float): Coupling constant. Default is 1.
    Outputs:
        params (dict): Dictionary containing 
            N (int): Total number of spins.
            Ns (int): Number of spin configurations.
            states (numpy.ndarray): All possible spin configurations.
            baseCoupling (numpy.ndarray): Base coupling for the system.
            adj (numpy.ndarray): State adjacency matrix for single-spin flip dynamics.
            pmap (list): Map between spins and couplings from the cluster Hamiltonian.
    """
    # Basic parameters and state list
    dims = np.array([Nx, Ny, Nz])
    N = np.prod(dims)  # Total number of spins
    Ns = 2**N # Number of spin configurations
    states = np.array(list(product(np.array([-1, 1]), repeat=N)), dtype=int) # All possible spin configurations

    # Calculate x, y, z coordinates from the spin index
    def index_to_coords(n):
        x = n // (Ny * Nz)          
        y = (n // Nz) % Ny          
        z = n % Nz                 
        return np.array([x, y, z])
    
    # Spin adjacency matrix on grid
    spinAdj = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            coords1 = index_to_coords(i)
            coords2 = index_to_coords(j)
            if np.sum(np.abs(coords1 - coords2)) == 1:
                spinAdj[i, j] = 1
                spinAdj[j, i] = 1

    # State adjacency matrix for single-spin flip dynamics
    differences = states[:, np.newaxis] != states[np.newaxis, :]
    diff_count = differences.sum(axis=2)
    adj = (diff_count == 1).astype(int)

    # Set base coupling
    Jcombs = list(combinations(range(N), 2))
    N_Jcombs = int((N * (N - 1)) / 2)
    baseCoupling = np.zeros(N_Jcombs)
    for i, comb in enumerate(Jcombs):
        baseCoupling[i] = J*spinAdj[comb[0], comb[1]]

    # Get the map between spins and couplings from the cluster Hamiltonian
    pmap = [list(c) for c in chain.from_iterable(combinations(range(N), r) for r in range(1, N+1))]

    params = {
        'N': N,
        'Ns': Ns,
        'states': states,
        'baseCoupling': baseCoupling,
        'adj': adj,
        'pmap': pmap
    }
    return params


def Jacobian(cset, params):
    """
    Calculate the Jacobian matrix for a given control set.
    """
    numCPs = len(cset)
    params['Ns'] = params['Ns']
    states = params['states']
    Jacobian = np.zeros((params['Ns'], numCPs))
    for i in range(params['Ns']):
        state = states[i]
        for j in range(numCPs):
            sset = [params['pmap'][c] for c in cset[j]]
            for s in sset:
                Jacobian[i, j] -= np.prod(state[s])
    return Jacobian


def getEnergyFunction(cset, params):
    """
    Returns a function to calculate the energy of a given spin configuration with a given control
    parameter set, given values for the control parameters.

    Currently, this assumes that a control set controls either all NN couplings or none of them, but 
    it should be straightforward to generalize this to allow for partial control of the NN coupling set.
    """
    J = Jacobian(cset, params)

    # Check if the base couplings are controlled
    NNlocs = list(list(np.where(params['baseCoupling'] == 1) + params['N'])[0])
    csetFlat = list(chain.from_iterable(cset))
    baseCouplingIsControlled = (len(set(csetFlat) & set(NNlocs)) > 0)

    if baseCouplingIsControlled:
        def egy(state_index, u):
            return J[state_index] @ u
        def V(u):
            return np.array([egy(i, u) for i in range(params['Ns'])])
        return egy, V
    else:
        def egy(state_index, u):
            NNlocs = list(list(np.where(params['baseCoupling'] == 1) + params['N'])[0])
            state = params['states'][state_index]
            baseEnergy = 0
            sset = [params['pmap'][c] for c in NNlocs]
            for s in sset:
                baseEnergy -= np.prod(state[s])
            return J[state_index] @ u + baseEnergy
        def V(u):
            return np.array([egy(i, u) for i in range(params['Ns'])])
        return egy, V


def eqbm(V):
    """ 
    Takes a vector of state energies V and returns the Boltzmann distribution.
    Assumes that the energies are unitless (or that beta = 1)
    """
    p = np.exp(-V)
    return p / np.sum(p)


def rateMatrix(V, params):
    """
    Takes state energies and returns the rate matrix for the system.
    Assumes that the energies are unitless (or that beta = 1)
    """
    dE = V[:, np.newaxis] - V[np.newaxis, :]
    W = np.where(params['adj'] == 1, 1 / (1 + np.exp(dE)), 0)
    return W - np.diag(np.sum(W, axis = 0))


def frictionTensor(V, params, J):
    """
    Takes state energies and a Jacobian returns the friction tensor for the system
    """
    pi = eqbm(V)
    oV = np.ones(params['Ns'])
    Proj = np.outer(pi, oV)
    W = rateMatrix(V, params)
    try:
        Wdi = np.linalg.pinv(W - Proj)
    except:
        Wdi = np.linalg.pinv(W - Proj + 1E-10 * np.eye(params['Ns']))
    return - J.T @ np.multiply(Proj + Wdi, pi) @ J


def gradient(dt, f):
    """
    Get the derivative of a function f given equal time steps dt in the data.
    """
    df = np.zeros_like(f)
    
    if f.ndim == 1:
        df[0] = (-3 * f[0] + 4 * f[1] - f[2]) / 2
        df[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / 2
        df[1:-1] = (f[2:] - f[:-2]) / 2
    else:
        df[0, :] = (-3 * f[0, :] + 4 * f[1, :] - f[2, :]) / 2
        df[-1, :] = (3 * f[-1, :] - 4 * f[-2, :] + f[-3, :]) / 2
        df[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2
    
    return df / dt


def frictionAndDerivative(V, params, J, cset, beta = 1):
    """
    Takes state energies, Jacobian matrix, and control set and returns the friction tensor for the system,
    as well as the partial derivatives of the friction tensor along the control parameters.
    """
    num_CPs = len(cset)

    pi = eqbm(V)
    oV = np.ones(params['Ns'])
    Proj = np.outer(pi, oV)
    W = rateMatrix(V, params)
    try:
        Wdi = np.linalg.pinv(W - Proj)
    except:
        Wdi = np.linalg.pinv(W - Proj + 1E-10 * np.eye(params['Ns']))
    fric = - J.T @ np.multiply(Proj + Wdi, pi) @ J

    pex = pi[:, None]
    eqDerivs = - beta * (pex * (np.eye(params['Ns']) - pex.T)) @ J
    dW = -beta * np.multiply(W, W.T)[:, :, np.newaxis] * (J[:, np.newaxis, :] - J[np.newaxis, :, :])
    dW -= np.einsum('ijk,jk->ijk', np.eye(dW.shape[0])[:, :, None], np.sum(dW, axis=0))
    dW = np.array([dW[:,:,i] for i in range(num_CPs)])
    outers = np.array([np.outer(eqDerivs[:,i], pi) + np.outer(pi, eqDerivs[:,i]) for i in range(num_CPs)])
    Wdderiv = -np.array([-Wdi @ (np.outer(eqDerivs[:,i], np.ones(params['Ns'])) - dW[i]) @ Wdi for i in range(num_CPs)])

    term1 = -outers
    term2 = np.array([np.multiply(Wdderiv[i], pi) + np.multiply(Wdi, eqDerivs[:,i]) for i in range(num_CPs)])
    fricDeriv = J.T @ (term1 - term2) @ J

    return fric, fricDeriv
            
