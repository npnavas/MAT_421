import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from datetime import datetime
from importlib import reload
import DiffConstruct_LR as lr
import DiffConstruct_HR as lr
import rk2tdma as rk
import AdvPlotting as ap
import boundaries as bc
reload(lr)
reload(rk)
reload(ap)
reload(bc)

## Consts ##
# uncomment the next line for other cases
Re = 1 # int(input("Enter a Reynolds number between 1 and 10,000 (Note, around Re = 100 the code takes a bit longer to run): "))
gamma = 1
Nmax = 100
N_in = Nmax - 2
Mmax = int(gamma * Nmax)
M_in = Mmax - 2
dr = 1/Nmax
dz = 1/Mmax
# uncomment the next line for other cases
dt = 1e-5 # float(input("Enter a time step (1e-5 reccomended for smaller Re): ")) # timesteps
tolerance = 1e-5
# uncomment the next line for other cases
max_iter = # int(input("Enter the max number of iterations (100000 reccomended): "))
tic = datetime.now()

## Matrix setup ##
# Main three arrays for Non-linear flow #
v = mc.set_v_boundaries(Nmax, Mmax)
psi = np.zeros((Nmax,Mmax),dtype = np.float64)
eta = np.zeros((Nmax,Mmax),dtype = np.float64)
