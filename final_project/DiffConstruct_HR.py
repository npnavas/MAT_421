import scipy.special as spp
import numpy as np

## HR DIFF MATRIX CONSTRUCTIONS##
# Constructs A_nn for HRN
def construct_Ann(gamma, Nmax): #Adapted by Tristin Might be wrong
    '''
    PARAMETERS
    ----------
    gamma: Aspect ratio
    Nmax: Number of r-gridpoints

    RETRUNS
    -------
    Tridiagonal matrix A_(nxn), (see derivation).
    '''
    # use np.diag(array) to make diagonal matrices with diagonals of my 1D-arrays
    return np.diag(-2/dr**2*np.ones((Nmax))) + np.diag((1+1/(2*np.linspace(2,Nmax, Nmax-1)))/dr**2, -1) + np.diag((1-1/(2*np.linspace(1,Nmax-1, Nmax-1)))/dr**2, 1)

# Constructs B_mm for HRN
def construct_Bnn(gamma, Mmax): #Adapted by Tristin Also might be wrong
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
    return np.eye(Mmax,Mmax, k = -1) * 1/(dz**2) + np.eye(Mmax,Mmax) * (-2/(dz**2)) + np.eye(Mmax,Mmax, k = 1) * 1/(dz**2)

# Constructs Z_nn for HRN
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


# Constructs E_nn for HRN
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
    return construct_Znn(gamma, Nmax)[1].dot(construct_Ann(gamma, Nmax).dot(np.linalg.inv(construct_Znn(gamma, Nmax)[1])))

# Constructs F_nm for HRN
def construct_Fnm(eta, dr, gamma, Nmax, Mmax): #Writen By Tristin --Very possibly wrong
    '''
    PARAMETERS
    ----------
    eta: Current vorticity array
    dr: r grid-spacing
    gamma: Aspect ratio
    Nmax: Number of r grid-points
    Mmax: Number of z grid-points

    RETRUNS
    -------
    Fnm: RHS matrix used to solve Apsi + psiB = F
    '''
    dr = 1/Mmax
    dz = 1/Nmax
    Fnm = np.zeros((Nmax,Mmax), dtype = np.float64)
    i = np.indices((Nmax,Mmax))[0]
    i = i + np.ones_like(i) #I did this because this is the indices of the inside of eta, the first of which should be e
    return -eta[1:-1,1:-1]*i*dr
