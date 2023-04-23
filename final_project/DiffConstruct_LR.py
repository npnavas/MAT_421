import scipy.special as spp
import numpy as np

def construct_Ann(gamma, Nmax):
    '''
    PARAMETERS
    ----------
    gamma: Aspect ratio
    Nmax: Number of r-gridpoints

    RETRUNS
    -------
    Tridiagonal matrix A_(nxn), (see derivation).
    '''
    dr = 1/(Nmax) # spacing between r-girdpoints (easier to impliment here)
    ai = np.linspace(1, Nmax, Nmax, endpoint = True)**-2 # positional dependent factor of main diagonal
    ai_minus1 = 1/(2*np.linspace(1,Nmax-1, Nmax-1, endpoint = True)) # positional dependent factor of super and sub diagonal
    ann_main = -(2 + ai)/dr**2 # making the main diagonal array with rule -(2+1/i^2)/dr^2
    ann_sub = (1-ai_minus1)/dr**2 # making the subdiagonal array with rule (1-1/i^2)/dr^2
    ann_super = (1+ai_minus1)/dr**2 # making the superdiagonal array with rule (1+1/i^2)/dr^2

    # use np.diag(array) to make diagonal matrices with diagonals of my 1D-arrays
    return np.diag(ann_main) + np.diag(ann_sub, -1) + np.diag(ann_super, 1)

# Constructs B_nn matrix (see derivation)
def construct_Bnn(gamma, Mmax):
    '''
    PARAMETERS
    ----------
    gamma: Aspect ratio
    Mmax: Number of z-gridpoints

    RETRUNS
    -------
    Tridiagonal matrix B_(mxm), (see derivation)
    '''
    dz = gamma / Mmax # spacing between z-girdpoints (easier to impliment here)

    # Following np.eye(N,N) routine is used since all values here are constant
    bnn_sub = np.eye(Mmax,Mmax, k = -1) * 1/(dz**2) # writes the sub diag
    bnn_main = np.eye(Mmax,Mmax) * (-2/(dz**2)) # writes the main diag
    bnn_super = np.eye(Mmax,Mmax, k = 1) * 1/(dz**2) # writes the super diag

    return bnn_sub + bnn_main + bnn_super

# Constructs eigenvalue/vectors
def construct_Znn(gamma, Nmax):
    '''
    PARAMETERS
    ----------
    gamma: Aspect ratio
    Nmax: Number of r-gridpoints

    RETRUNS
    -------
    Tuple of arrays in the form (1D-eigenvalue array, eigenvector matrix)
    '''
    return np.linalg.eig(construct_Ann(gamma, Nmax))[0], np.linalg.eig(construct_Ann(gamma, Nmax))[1]

# Constructs a diagonal eigenvalue matrix
def construct_Enn(gamma, Nmax):
    '''
    PARAMETERS
    ----------
    gamma: Aspect ratio
    Nmax: Number of r-gridpoints

    RETRUNS
    -------
    E_nn: diagonal matrix of eigenvalues
    '''
    return construct_Znn(gamma, Nmax)[1].dot(construct_Ann(gamma, Nmax).dot(np.inverse(construct_Znn(gamma, Nmax)[1])))
    # Didn't want to use np.diag(np.linalg.eig(construct_LR_Ann(gamma, Nmax))[0]) since idk how ordering would work

# Constructs the RHS matrix that accounts for rotating bottom
def construct_Fnm(gamma, Nmax, Mmax):
    '''
    PARAMETERS
    ----------
    gamma: Aspect ratio
    Nmax: Number of r-gridpoints
    Mmax: Number of z-gridpoints

    RETRUNS
    -------
    Fnm: Zero array except for the first column which accounts for the z=0 condition
    '''
    dr = 1/Mmax
    dz = 1/Nmax
    Fnm = np.zeros((Nmax,Mmax), dtype = np.float64)
    f_edge = -(dr/dz**2) * np.linspace(1, Nmax, Nmax, endpoint = True) # Positional dependence of this "correction facotor" of sorts
    Fnm[:,0] = f_edge # mapping said factor to the larger array

    return Fnm
