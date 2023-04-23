import numpy as np
import scipy.special as spp

## Linear soln ##
def exact_lowR_solution(gamma,N=100,epsilon=1e-5,max_Iter = 1000):
    '''
        This function generates an array containing the values for the exact low Re solution for the equation
        It takes three parameters:
            gamma:  The height of the cylinder
            N:  The resolution of the matrix in the r direction ie. the numbers of columns.
            epsilon: The maximum acceptable absolute error between each iteration of the sum

        It returns a matrix of size (int(N*gamma),N) containing the approximate analytical solution for low Re
    '''
    #Creates arrays containing each array position's coordinates
    r, z = np.indices((N,int(N*gamma)))
    z = z/(int(N*gamma)-1)*gamma
    r = r/(N-1)

    #Array for the result
    result = np.zeros(shape = (N,int(N*gamma)))

    #Adds the summation part of the solution until the changes are smaller than the tolerance
    n = 1
    while True:
        updated = result + spp.i1(n*np.pi*r/gamma)/(n*spp.i1(n*np.pi/gamma))*np.sin(n*np.pi*z/gamma)
        if np.linalg.norm(updated-result, ord = "fro")<epsilon:
            result = updated
            print('Analytical solution converged in',n,'iterations to within a tolerance of', epsilon)
            break
        else:
            result = updated
            n += 1
        if n > max_Iter:
            print("Failed to converge after" ,max_Iter, "iterations")
            break

    #Finishes up the rest of the result
    result = -2/np.pi*result
    result += r*(1-z/gamma)
    return result


## TDMA Solver used for some steady state/poisson eq ##
def TDMAsolver(a, b, c, d): # Thanks to the person I stole this from github gist
    '''
    PARAMETERS
    ----------
    a: One dimensional array for the sub diagonal
    b: One dimensional array for the main diagonal
    c: One dimensional array for the super diagonal
    d: Solution vector

    RETRUNS
    -------
    xc: Solution to the vector equation Ax = d for A being a tridiagonal matrix and d being the solution vector
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = (x.astype(float) for x in (a, b, c, d)) # copy arrays & cast to floats
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]

    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

## RK2 methods LRN ##
def RHS_LR_T(v, dr, dz, Nmax, Mmax, Re = 1):
    '''
    PARAMETERS
    ----------
    v: Current velocity of fluid
    dr: r-gridpoint spacing
    dz: z-gridpoint spacing
    Nmax: Number of r-gridpoints
    Mmax: Number of z-gridpoints
    Re: Reynolds number

    RETRUNS
    -------
    dvdt: Current time derivative array of the fluid velocity
    '''
    dvdt = np.zeros((Nmax,Mmax))
    row_nums, col_nums = np.indices((Nmax,Mmax))

    mid_rows = row_nums[1:-1,1:-1]

    mid_v = v[1:-1,1:-1]
    up_v = v[2:,1:-1]
    down_v = v[:-2,1:-1]
    left_v = v[1:-1,:-2]
    right_v = v[1:-1,2:]

    a = up_v-2*mid_v+down_v#dr**2 # v_rr I think
    b = (up_v-down_v)/(2*mid_rows)#*dr**2) # v_r I think
    c = mid_v/np.power(mid_rows,2)#*dr**2) # v I think
    d = left_v+right_v-2*mid_v#/dz**2 # v_zz
    dvdt[1:-1,1:-1] = (a+b-c+d)/(Re*dr**2)
    return dvdt

def RK2_LR(v, dr, dz, Nmax, Mmax, Re = 1, dt = 1e-5):
    '''
    PARAMETERS
    ----------
    v: Array for the current velocity of fluid
    Re: Reynolds number
    dr: r-gridpoint spacing
    dz: z-gridpoint spacing
    Nmax: Number of r-gridpoints
    Mmax: Number of z-gridpoints
    Re: Reynolds number
    dt: Time step

    RETRUNS
    -------
    corrected: The next timestep
    '''
    predict = v + RHS_LR_T(v, dr, dz, Nmax, Mmax, Re = 1) * dt

    corrected = v + 0.5 * dt * (RHS_LR_T(v, dr, dz, Nmax, Mmax, Re = 1) + RHS_LR_T(predict, dr, dz, Nmax, Mmax, Re = 1))
    return corrected
