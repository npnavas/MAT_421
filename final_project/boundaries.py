import numpy as np
def set_v_boundaries(Nmax, Mmax): # This sets the rotating bottom. Same for Low and High Re
    '''
    PARAMETERS
    ----------
    Nmax: Number of r-gridpoints
    Mmax: Number of z-gridpoints

    RETURNS
    -------
    v: array with the noslip boundary conditions along with rotating bottom
    '''
    v = np.zeros((Nmax,Mmax), dtype = np.float64) # Initalizes the v matrix
    r_condition = np.linspace(0,1, Nmax, endpoint = True) # Rotating bottom condition stored to a 1D array
    v[:,0] = r_condition # Maps r BC to the v array

    return v

def set_eta_bounds(eta, psi, dr, dz, Nmax, Mmax): # Used for poisson-like equation
    eta[-1,:] = 1/(2*(Mmax)*np.ones((Mmax))*dr**3)*(psi[-3,:]-8*psi[-2,:]) # right
    eta[0,:] = 0 # left
    eta[:,-1] = 1/(2*np.arange(1,Nmax+1)*dr*dz**2)*(psi[:,-3]-8*psi[:,-2]) # top
    eta[:,0] =  1/(2*np.arange(1,Nmax+1)*dr*dz**2)*(psi[:,2]-8*psi[:,1]) # bottom
    return eta
