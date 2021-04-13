"""
Example plots for spherical distributions

Author: Romain Fayat, April 2021
"""
import numpy as np
from SphereProba.distributions import VonMisesFisher, Kent
import matplotlib.pyplot as plt
from pathlib import Path
import angle_visualization.triangulation
import seaborn as sns


def plot_vMF(*args, **kwargs):
    "Plot example von Mises Fisher distributions"

    # Create  a triangulated sphere
    n_points = 3000
    d = angle_visualization.triangulation.Delaunay_Sphere(n_points)
    center_all = d.face_centroids

    # Range of parameters
    kappa_all = [0.001, 0.01, 1., 10, 100]
    mu_all = [np.array([0, 0, 1])] * len(kappa_all)

    fig = plt.figure(figsize=(4 * len(mu_all), 4))
    for i, (mu, kappa) in enumerate(zip(mu_all, kappa_all)):
        # Create distributions for given parameters
        vmf = VonMisesFisher(mu, kappa)
        vmf_density = vmf(center_all)

        # Generate a color map from the density value
        cmap = sns.mpl_palette("viridis", as_cmap=True)
        rescaled = (vmf_density - vmf_density.min()) /\
                   (vmf_density.max() - vmf_density.min())
        colors = cmap(rescaled)

        # Plot the density
        ax = fig.add_subplot(1, len(mu_all), i + 1, projection='3d')

        angle_visualization.plot_faces(d.faces, ax=ax, linewidths=1,
                                       face_colors=colors, edge_colors=colors)
        ax.scatter(*mu, c="w", s=30)
        angle_visualization.utils.set_3Dlim(*[[-.8, .8] for _ in range(3)],
                                            ax=ax)
        # Final tweaking and save the figure
        ax.axis('off')
        ax.set_title(f"κ={kappa}")
    fig.tight_layout()
    fig.savefig(*args, **kwargs)
    plt.close(fig)


def plot_kent(*args, **kwargs):
    "Plot example Kent distributions"
    # Create  a triangulated sphere
    n_points = 3000
    d = angle_visualization.triangulation.Delaunay_Sphere(n_points)
    center_all = d.face_centroids


    # Range of parameters
    kappa_all = [.01, 1, 10, 100]
    beta_over_kappa_all = [0, .25, .49]
    gamma = np.eye(3)[[2, 1, 0]]
    params_grid = np.meshgrid(beta_over_kappa_all, kappa_all)
    params_grid = [params_grid[0].flatten(), params_grid[1].flatten()]

    # create the figure
    figsize = (4 * len(beta_over_kappa_all), 4 * len(kappa_all))
    fig = plt.figure(figsize=figsize)


    for i, (beta_over_kappa, kappa) in enumerate(zip(*params_grid)):

        # Create distributions for given parameters
        beta = beta_over_kappa * kappa
        k = Kent(gamma, kappa, beta)
        k_density = k(center_all)


        # Generate a color map from the density value
        cmap = sns.mpl_palette("viridis", as_cmap=True)
        rescaled = (k_density - k_density.min()) /\
                   (k_density.max() - k_density.min())
        colors = cmap(rescaled)

        # Plot the density
        ax = fig.add_subplot(len(kappa_all), len(beta_over_kappa_all),
                             i + 1, projection='3d')
        ax.view_init(60, 0)
        angle_visualization.plot_faces(d.faces, ax=ax, linewidths=1,
                                       face_colors=colors, edge_colors=colors)
        angle_visualization.utils.set_3Dlim(*[[-.8, .8] for _ in range(3)],
                                            ax=ax)
        ax.set_title(f"κ={kappa}, β={beta}")

        # Final tweaking and save the figure
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(*args, **kwargs)
    plt.close(fig)



if __name__ == "__main__":
    OUTPUT_FOLDER = Path(__file__).parent
    p = OUTPUT_FOLDER / "vmf.png"
    plot_vMF(p, transparent=True)
    p = OUTPUT_FOLDER / "kent.png"
    plot_kent(p, transparent=True)
