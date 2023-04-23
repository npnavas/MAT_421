import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from importlib import reload
import DiffConstruct_LR as lr
import rk2tdma as rk
import AdvPlotting as ap
import boundaries as bc
reload(lr)
reload(rk)
reload(ap)
reload(bc)

tic = datetime.now() # start timer

## Consts ##
gamma = 1

## Grid nonsense ##
Nmax = 100
N_in = Nmax - 2 # interior points r
Mmax = int(gamma * Nmax)
M_in = Mmax - 2 # interior points z

## Matrix constructions ##
soln = bc.set_v_boundaries(Nmax, Mmax)

# solve interior gridpoints
B_mm = lr.construct_Bnn(gamma, M_in)
eig = lr.construct_Znn(gamma, N_in)[0]
Z_nn = lr.construct_Znn(gamma, N_in)[1]
Z_nn_inv = np.linalg.inv(lr.construct_Znn(gamma, N_in)[1]) #
I_mm = np.identity(M_in)
F_nm = lr.construct_Fnm(gamma, N_in, M_in)
H_mn = np.transpose(F_nm).dot(np.transpose(Z_nn_inv))
U_nm = np.zeros((N_in, M_in), dtype=np.float64)

for i in range(N_in):
    U_nm[i, :] = rk.TDMAsolver((B_mm + eig[i]*I_mm).diagonal(-1),(B_mm + eig[i]*I_mm).diagonal(0),(B_mm + eig[i]*I_mm).diagonal(1), H_mn[:,i])

soln[1:-1, 1:-1] = Z_nn.dot(U_nm)
ap.plot_contour(soln, gamma, rf"Steady State Solution for $\Gamma$ = {gamma}", "SS_test.png")
toc = datetime.now()
print(f"Calculation Runtime: {toc - tic}.")

## Error-analysis ##
exact = rk.exact_lowR_solution(gamma, N = Nmax, epsilon = 1e-7)
ap.plot_contour(np.abs(soln - exact), gamma, 'Absolute Error Betwen Solutions', 'mat_error.png')
rel_error = np.abs(np.linalg.norm(soln - exact, ord = "fro")/np.linalg.norm(exact, ord = "fro")) * 100
print(f"Relative error: {rel_error} %")
plt.show()
