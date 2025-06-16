import numpy as np
import numpy.linalg as la
from isingFuncts import *
from scipy.interpolate import interp1d

def rayleighQuotientIteration(psi0, A, params, threshold = 1E-8):
    v = psi0 / la.norm(psi0)
    mu = v.T @ A @ v
    while True:
        v = la.solve(A - (mu + 1E-9) * np.eye(params['Ns']), v)
        v = v / la.norm(v)
        mu = v.T @ A @ v
        if la.norm(A @ v - mu * v) < threshold:
            break
    return mu, v

# Iterate along protocol using Rayleigh quotient mathod
def iterPass(Vt, N, params, threshold = 1E-8):
    tt = np.linspace(0,1,N)
    V = interp1d(np.linspace(0,1,Vt.shape[0]), Vt, axis = 0)(tt)
    # Get the eigenvectors and eigenvalues at time step 0, 
    # dropping the null eigenvalue and vector
    eigvals = np.zeros((len(tt), params['Ns']-1))
    eigvecs = np.zeros((len(tt), params['Ns'], params['Ns']-1))
    val, vec = la.eig(rateMatrix(V[0], params))
    idxs = np.argsort(val)[::-1]
    eigvals[0] = np.real(val[idxs[1:]])
    vecs = vec[:, idxs[1:]]
    # Normalize the vectors
    for i in range(params['Ns']-1):
        eigvecs[0, :, i] = vecs[:, i] / la.norm(vecs[:, i])
    # Iterate for each time step from the previous eigenvectors    
    for i in range(1, len(tt)):
        for j in range(params['Ns'] - 1):
            W = rateMatrix(V[i], params)
            psi0 = eigvecs[i-1, :, j]
            mu, v = rayleighQuotientIteration(psi0, W, params, threshold)
            eigvals[i, j] = mu
            eigvecs[i, :, j] = v
    # Normalize the eigenvectors
    left_eigenvectors = np.zeros_like(eigvecs)
    right_eigenvectors = np.zeros_like(eigvecs)
    for i in range(len(tt)):
        pi = eqbm(V[i])
        vec = eigvecs[i]
        lvecs = np.zeros_like(vec)
        for j in range(params['Ns']-1):
            lvec = vec[:,j] / pi
            norm = np.sqrt(vec[:,j] @ lvec)
            vec[:,j] = vec[:,j] / norm
            lvecs[:,j] = lvec / norm
        right_eigenvectors[i] = vec
        left_eigenvectors[i] = lvecs

    return eigvals, right_eigenvectors, left_eigenvectors

# Computed vectors may disagree in sign - standardize to produce smooth curves in the eigenvector components
def standardizeSpectrum(vecs):
    R = vecs.shape[0]
    Ns = vecs.shape[2]
    for i in range(R):
        for j in range(1,Ns):
            last_signs = np.sign(vecs[i-1,:,j])
            signs = np.sign(vecs[i,:,j])
            sum_same = np.sum(last_signs == signs)
            sum_diff = np.sum(last_signs != signs)
            if sum_diff > sum_same:
                vecs[i,:,j] *= -1
    return vecs