import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
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

tic = datetime.now()
## Consts/Grid nonsense ##
gamma = 1
Nmax = 100
Mmax = int(gamma * Nmax)
dr = 1/Nmax
dz = 1/Mmax
dt = 1e-5
Re = 1
tol = 1e-5
iter_count = 0
max_iter = 100000

# animation set up #
soln = bc.set_v_boundaries(Nmax, Mmax)
soln_anim = np.zeros((int(max_iter/50), Nmax, Mmax), dtype = np.float64)
soln_anim[:,:,:] = soln

# Integrator #
v_n = np.copy(soln)
for n in range(max_iter):
    soln = rk.RK2_LR(v_n, dr, dz, Nmax, Mmax, Re, dt)
    if n % 50 == 0:
        soln_anim[n//50, :, :] = v_n
        print("Iteration {0:5d}".format(n), end="\r")

    if np.linalg.norm(soln-v_n, ord = "fro") < tol:
        print(f"Converged in {n} iterations.")
        print(f"Difference between timesteps using Frobienus norm: {5}")
        soln_anim = soln_anim[0:n//50+1, :, :]
        break

    v_n[:, :] = soln

toc = datetime.now() # End timer
print(f"Calculation run-time: {toc-tic}")

## Error-analysis ##
exact = rk.exact_lowR_solution(gamma, N = Nmax, epsilon = 1e-7)
ap.plot_contour(np.abs(soln_anim[-1] - exact), gamma, 'Absolute Error', filename = "Transient_error_LNR.png", cmap=plt.cm.turbo)
rel_error = np.abs(np.linalg.norm(soln_anim[-1] - exact, ord = "fro")/np.linalg.norm(exact, ord = "fro")) * 100
print(f"Relative error: {rel_error} %")

## Animation ##
def animate(k):
    ap.plotheatmap(np.transpose(soln_anim[k]), k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames = np.size(soln_anim, 0), repeat=True, save_count=1500)

print("Saving animation...")
writergif = animation.PillowWriter(fps=30)
anim.save("Transient_Solution_LRN.gif", writer=writergif)
print("Animation Saved")
