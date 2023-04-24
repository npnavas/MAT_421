import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from datetime import datetime
from importlib import reload
import DiffConstruct_HR as hr
import rk2tdma as rk
import AdvPlotting as ap
import boundaries as bc
reload(hr)
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
tol = 1e-5
# uncomment the next line for other cases
max_iter = 100000 # int(input("Enter the max number of iterations (100000 reccomended): "))
tic = datetime.now()

## Matrix setup ##
# Main three arrays for Non-linear flow #
v = bc.set_v_boundaries(Nmax, Mmax)
psi = np.zeros_like(v)
eta = bc.set_eta_bounds(np.zeros_like(v), psi, dr, dz, Nmax, Mmax)
v_old = np.copy(v)
eta_old = np.copy(eta)


# Arrays for poisson like eq #
B_mm = hr.construct_Bnn(gamma, M_in)
eig = hr.construct_Znn(gamma, N_in)[0]
Z_nn = hr.construct_Znn(gamma, N_in)[1]
Z_nn_inv = np.linalg.inv(hr.construct_Znn(gamma, N_in)[1]) #
I_mm = np.identity(M_in)
F_nm = hr.construct_Fnm(eta, dr, gamma, N_in, M_in)
H_mn = np.transpose(F_nm).dot(np.transpose(Z_nn_inv))
U_nm = np.zeros((N_in, M_in), dtype=np.float64)

# Animation setup#
soln_anim = np.zeros((int(max_iter/50), Nmax, Mmax), dtype = np.float64)
soln_anim[:, :, :] = v

# Integrator #
for n in range(max_iter):
    #huen step
    v = rk.RK2_HR_V(v_old, psi, dr, dz, Re, dt)
    eta = rk.RK2_HR_eta(v, psi, eta_old, dr, dz, Re, dt)

    # poisson eq
    F_nm[:, :] = hr.construct_Fnm(eta, dr, gamma, N_in, M_in)
    H_mn[:, :] = np.transpose(F_nm).dot(np.transpose(Z_nn_inv))
    for i in range(Nmax-2):
        U_nm[i,:] = rk.TDMAsolver((B_mm + eig[i]*I_mm).diagonal(-1),(B_mm + eig[i]*I_mm).diagonal(0),(B_mm + eig[i]*I_mm).diagonal(1), H_mn[:,i])

    psi[1:-1, 1:-1] = Z_nn.dot(U_nm)
    eta = bc.set_eta_bounds(eta, psi, dr, dz, Nmax, Mmax)
    if n % 50 == 0:
        soln_anim[n//50, :, :] = v
        print("Iteration {0:5d}".format(n), end="\r")

    err = np.linalg.norm(np.abs(v-v_old), "fro")
    if err < tol:
        print(f"Converged in {n} iterations.")
        print(f'Final error using Frobienus norm: {err}')
        soln_anim = soln_anim[0:iteration//50+1, :, :]
        break

    v_old[:, :] = v
    eta_old[:, :] = eta
    if n == max_iter-1:
        print(f"Did not converge in {max_iter} iterations")
        print(f'Final error using Frobienus norm: {err}')

toc = datetime.now() # End timer
print(f"Calculation run-time: {toc-tic}")

## Error-analysis ##
exact = rk.exact_lowR_solution(gamma, N = Nmax, epsilon = 1e-7)
ap.plot_contour(np.abs(soln_anim[-1] - exact), gamma, 'Absolute Error', filename = "Transient_error_HNR.png", cmap=plt.cm.turbo)
rel_error = np.abs(np.linalg.norm(soln_anim[-1] - exact, ord = "fro")/np.linalg.norm(exact, ord = "fro")) * 100
print(f"Relative error: {rel_error} %")

## Animation ##
def animate(k):
    ap.plotheatmap(np.transpose(soln_anim[k]), k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames = np.size(soln_anim, 0), repeat=True, save_count=1500)

print("Saving animation...")
writergif = animation.PillowWriter(fps=30)
anim.save(f"Transient_Solution_HRN_Re_{Re}.gif", writer=writergif)
print("Animation Saved")
