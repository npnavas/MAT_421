import numpy as np
import matplotlib.pyplot as plt # Used for plotting
from mpl_toolkits.mplot3d import Axes3D # Used for some 3D-plotting
import matplotlib.animation as animation # Used to make the animation


def plotheatmap(v_k, k):
    """
    PARAMETERS
    ----------
    v_k: Array at time-step k
    k: Current time-step
    """
    # Clear the current plot figure
    plt.clf()
    plt.title(f"Fluid Velocity at time step {int(k*50)}")
    plt.xlabel(r"r $\times$ 10")
    plt.ylabel(r"z $\times$ 10")

    # This is to plot v_k (v at time-step k)
    plt.contourf(v_k, levels = 20, cmap=plt.cm.turbo)
    plt.colorbar()

    return plt

def plot_contour(Phi, gamma, title, filename=None, cmap=plt.cm.turbo):
    """
    Credit to Dr. Oliver Beckstein here. I took this from the resources of my
    PHY 202/432 classes on solving Laplace's equation numerically but made the title
    modification. His documnetation remains the same

    Plot Phi as a contour plot.

    Arguments
    ---------
    Phi : 2D array
          potential on lattice
    gamma : gamma value
    title : String type. Creates the title of out contour plot.
    filename : string or None, optional (default: None)
          If `None` then show the figure and return the axes object.
          If a string is given (like "contour.png") it will only plot
          to the filename and close the figure but return the filename.
    cmap : colormap
          pick one from matplotlib.cm
    """
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    x = np.arange(Phi.shape[0])
    y = np.arange(Phi.shape[1])
    X, Y = np.meshgrid(x, y)
    Z = Phi[X, Y]
    cset = ax.contourf(X, Y, Z, 20, cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel(r'10$\times$r')
    ax.set_ylabel(r'10$\times$z/$\Gamma$')
    ax.set_aspect(1)

    cb = fig.colorbar(cset, shrink=0.5, aspect=5)

    if filename:
        fig.savefig(filename)
        plt.close(fig)
        return filename
    else:
        return ax
