import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from matplotlib import pyplot as plt
from PIL.Image import Image

# libraries
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

tick_size = 15
label_size = 20
title_size = 25
fig_size = (15, 8)


def save_fig(fig: Image | go.Figure | plt.Figure, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fig, Image):
        fig.save(path)
    elif isinstance(fig, go.Figure):
        fig.write_image(path)
    elif isinstance(fig, plt.Figure):
        fig.savefig(path)
    else:
        raise ValueError(f"Unknown figure type {type(fig)}.")


def plot_heatmap(x: torch.Tensor, y: torch.Tensor):
    """
    Plot the heatmap
    """
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 2
    assert y.shape[1] == 1

    if x.device != "cpu":
        x = x.cpu()
        y = y.cpu()
    # Convert tensors to numpy arrays for plotting
    x_np = x.numpy()
    y_np = y.numpy()

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # Create a 2D histogram plot
    h = ax.hist2d(x_np[:, 0], x_np[:, 1], bins=50, cmap="hot", weights=y_np.squeeze(-1))

    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.title("Heatmap", fontsize=title_size)
    plt.colorbar(h[3], ax=ax)

    return fig
    # check if we are at ablations folder
    # if os.getcwd().split('/')[-1] == 'ablation':
    #     name = '../plots'
    # else:
    #     name = 'plots'
    # plt.savefig(os.path.join(name, fig_name))

    # plt.close(fig)


def plot_wos_max_step_traj(
    max_step: List[int],
    n_traj: int,
    n_dim: int,
    l2_loss: List[float],
    relative_error: List[float],
    pde_name: str,
):
    """
    Plot WoS for different max_step and trajectories
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(121)
    plt.plot(max_step, l2_loss, marker="o")
    # make y axis log scale
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Number of maximum steps", fontsize=label_size)
    plt.ylabel("Loss", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.title(f"L2 Loss", fontsize=title_size)
    plt.grid()

    ax = fig.add_subplot(122)
    plt.plot(max_step, relative_error, marker="o")
    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel("Number of maximum steps", fontsize=20)
    plt.ylabel("Relative Error", fontsize=20)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title(f"Relative Error", fontsize=title_size)
    plt.grid()

    plt.suptitle(
        f"WoS on {pde_name} with {n_dim} Dimensions with {n_traj}-trajectories",
        fontsize=25,
    )
    path = "plots/sweep_max_step_traj"
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(os.path.join(path, f"{pde_name}_{n_dim}d_{n_traj}.png"))
    plt.close(fig)


def plot_wos_dim_traj(
    n_traj: List[int],
    n_dim: int,
    l2_loss: List[float],
    relative_error: List[float],
    pde_name: str,
):
    """
    Plot WoS for different dimensions and trajectories
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(121)
    plt.plot(n_traj, l2_loss, marker="o")
    # make y axis log scale
    plt.yscale("log")
    plt.xlabel("Number of Trajectories", fontsize=label_size)
    plt.ylabel("Loss", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.title(f"L2 Loss", fontsize=title_size)
    plt.grid()

    ax = fig.add_subplot(122)
    plt.plot(n_traj, relative_error, marker="o")
    plt.yscale("log")

    plt.xlabel("Number of Trajectories", fontsize=20)
    plt.ylabel("Relative Error", fontsize=20)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title(f"Relative Error", fontsize=title_size)
    plt.grid()

    plt.suptitle(f"WoS on {pde_name} with {n_dim} Dimensions", fontsize=25)
    path = "plots/sweep_dim_traj"
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(os.path.join(path, f"{pde_name}_{n_dim}d_n_traj.png"))


def plot_step_of_wos(n_steps: torch.Tensor, n_bins: int = 100) -> None:
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    plot = ax.hist(n_steps.cpu().squeeze(-1), bins=n_bins)
    ax.set_xlabel("Number of Steps", fontsize=label_size)
    ax.set_ylabel("Count", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.title(f"Number of Steps", fontsize=25)
    plt.grid()
    return fig


def plot_step_of_wos_with_bins(counts, bins, log=True):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    plt.stairs(counts, bins)
    ax.set_xlabel("Number of Steps", fontsize=label_size)
    ax.set_ylabel("Count", fontsize=label_size)
    if log:
        ax.set_xscale("log")
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.title(f"Number of Steps", fontsize=25)
    plt.grid()
    return fig


def plot_hist(
    n_bins: int, y_true: torch.Tensor, y_pred: torch.Tensor, save_dir: str
) -> None:
    """
    Plot the histogram
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(121)
    # plot 2 histograms side by side
    n, bins, patches = ax.hist(y_true.cpu().squeeze(-1), bins=n_bins, density=True)
    ax.set_xlabel("Bin value", fontsize=label_size)
    ax.set_ylabel("Count", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title("Histogram of True Solutions", fontsize=25)
    ax = fig.add_subplot(122)
    plot = ax.hist(y_pred.cpu().squeeze(-1), bins=bins, density=True)

    ax.set_xlabel("Bin value", fontsize=label_size)
    ax.set_ylabel("Count", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title("Histogram of Predicted Solutions", fontsize=25)

    plt.savefig(save_dir)
    plt.close()


def plot_true(x: torch.Tensor, y: torch.Tensor, title: str = "annulus_true"):
    """
    Plot the true solution
    """

    if x.device != "cpu":
        x = x.cpu()
        y = y.cpu()

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y, marker="o")

    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title("True Solution on {}".format(title), fontsize=25)
    plt.grid()
    plt.savefig(f"plots/{title}.png")
    plt.close()


def plot_wos(n_traj: int, x: torch.Tensor, y: torch.Tensor):

    x = x.cpu()
    y = y.cpu()

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x[:, 0], x[:, 1], y, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title(f"WoS on Annulus with {n_traj} Trajectories", fontsize=25)
    plt.grid()
    plt.savefig(f"plots/annulus_wos_{n_traj}.png")
    plt.close()


def plot_loss_nn(losses: List):
    fig = plt.figure(figsize=fig_size)
    plt.plot(losses, marker="o")
    plt.xlabel("Number of Iterations", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.title("Loss vs. Number of Iterations", fontsize=25)
    plt.grid()
    plt.savefig("plots/training_loss.png")
    plt.close()

def plot_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    save_path: str,
    title: str,
):
    fig = plt.figure(figsize=fig_size)
    # Make 2 plots side by side where left one is the true solution and right one is the solution from WOS
    ax = fig.add_subplot(111, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y, marker="o")
    # ax.set_xlabel("x", fontsize=label_size)
    # ax.set_ylabel("y", fontsize=label_size)
    # ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    # for t in ax.zaxis.get_major_ticks():
    #     t.label.set_fontsize(tick_size)
    if title == 'WoS':
        title = 'Ground Truth'
    plt.title(title, fontsize=25)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig

 
 
def plot_3d_sns(x: torch.Tensor,
    y: torch.Tensor,
    save_path: str,
    title: str,
):
    # to Add a color bar which maps values to colors.
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")
    ax.axis("off")

    surf=ax.scatter(x[:,0], x[:,1], y.squeeze(-1), c=y.squeeze(-1), cmap=plt.cm.viridis)#plot_trisurf(x[:,0], x[:,1], y.squeeze(-1), cmap=plt.cm.viridis, linewidth=0.2)
    if title == 'WoS':
        title = 'Ground Truth'
    ax.grid()
    plt.title(title, fontsize=25)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig


def plot_ys(
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_wos: torch.Tensor,
    y_pred: torch.Tensor,
    iter: int,
):
    fig = plt.figure(figsize=fig_size)
    # Make 2 plots side by side where left one is the true solution and right one is the solution from WOS
    ax = fig.add_subplot(131, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y_true, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    for t in ax.zaxis.get_major_ticks():
        t.label.set_fontsize(tick_size)

    plt.title("True", fontsize=25)
    plt.grid()

    ax = fig.add_subplot(132, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y_wos, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    for t in ax.zaxis.get_major_ticks():
        t.label.set_fontsize(tick_size)

    plt.title(f"WoS", fontsize=25)
    plt.grid()

    ax = fig.add_subplot(133, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y_pred, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    for t in ax.zaxis.get_major_ticks():
        t.label.set_fontsize(tick_size)

    plt.title(f"NN", fontsize=25)
    plt.grid()

    plt.savefig(f"plots/annulus_true_wos_pred_{iter}.png")
    plt.close()


def plot_all_solvers(
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_nn: torch.Tensor,
    y_wos: torch.Tensor,
    y_ritz: torch.Tensor,
    y_pinn: torch.Tensor,
    iter: torch.Tensor,
):

    y_true = y_true.cpu()
    y_nn = y_nn.cpu()
    y_wos = y_wos.cpu()
    y_ritz = y_ritz.cpu()
    y_pinn = y_pinn.cpu()

    fig = plt.figure(figsize=fig_size)
    # Make 2 plots side by side where left one is the true solution and right one is the solution from WOS
    ax = fig.add_subplot(151, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y_true, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title("True", fontsize=25)
    plt.grid()

    ax = fig.add_subplot(152, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y_wos, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title(f"WoS", fontsize=25)
    plt.grid()

    ax = fig.add_subplot(153, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y_nn, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title(f"NWoS", fontsize=25)
    plt.grid()

    ax = fig.add_subplot(154, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y_ritz, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)


    plt.title(f"DeepRitz", fontsize=25)
    plt.grid()

    ax = fig.add_subplot(155, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y_pinn, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title(f"PINN", fontsize=25)
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"plots/all_solvers.png")
    plt.close()


def plot_loss_std(n_trajs: List, losses: List, stds: List):
    fig = plt.figure(figsize=fig_size)
    plt.plot(n_trajs, losses, marker="o")
    plt.xlabel("Number of Trajectories", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.title("Loss vs. Number of Trajectories", fontsize=25)
    plt.grid()
    plt.savefig("plots/annulus_loss.png")
    plt.close()

    fig = plt.figure(figsize=fig_size)
    plt.plot(n_trajs, stds, marker="o")
    plt.xlabel("Number of Trajectories", fontsize=20)
    plt.ylabel("Std", fontsize=20)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.title("Std vs. Number of Trajectories", fontsize=25)
    plt.grid()
    plt.savefig("plots/annulus_std.png")
    plt.close()


def plot_wos_true(
    n_traj: int,
    x: torch.Tensor,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    title: str = "annulus_true",
):

    x = x.cpu()
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()

    fig = plt.figure(figsize=fig_size)
    # Make 2 plots side by side where left one is the true solution and right one is the solution from WOS
    ax = fig.add_subplot(121, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y_true, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title(f"True Solution on {title}", fontsize=25)
    plt.grid()

    ax = fig.add_subplot(122, projection="3d")
    plot = ax.scatter(x[:, 0], x[:, 1], y_pred, marker="o")
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.set_zlabel("u", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.title(f"{title} with {n_traj} Trajectories", fontsize=25)
    plt.grid()

    plt.savefig(f"plots/{title}_{n_traj}.png")
    plt.close()
