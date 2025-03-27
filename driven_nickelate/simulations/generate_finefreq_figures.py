import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

from driven_nickelate.config import paths as PROJECT_PATHS

SIMULATIONS_ROOT = PROJECT_PATHS / "simulations"

data_dir = SIMULATIONS_ROOT / "data/19.06.23/finefreq"
fig_dir = SIMULATIONS_ROOT / "figures/"

data_files = [
    os.path.join(data_dir, fn)
    for fn in os.listdir(data_dir)
    if fn.startswith("surface")
]
cond_list = [float(re.findall(r"(\d+\.\d+E[+-]\d+)", fn)[0]) for fn in data_files]
data_files, cond_list = list(
    zip(*sorted(zip(data_files, cond_list), key=lambda x: x[1]))
)

for i in range(len(cond_list)):
    data = pd.read_csv(data_files[i])

    x = data["x"].unique()
    y = data["y"].unique()

    Ey = np.abs(
        data.set_index("freq").query('variable == "Ey (V/m)"')["real_value"]
        + 1j * data.set_index("freq").query('variable == "Ey (V/m)"')["real_value"]
    )
    Jx = data.set_index("freq").query('variable == "Jsupx (A/m)"')["real_value"]
    Jy = data.set_index("freq").query('variable == "Jsupy (A/m)"')["real_value"]
    J = np.hypot(Jx, Jy)

    J_max = J.max()
    Ey_max = Ey.max()

    norm_Ey = Normalize(vmin=0, vmax=Ey_max)
    norm_J = Normalize(vmin=0, vmax=J_max)

    for f in sorted(data["freq"].unique()):
        Ey_data = Ey.loc[f].values.reshape((len(x), len(y)))
        Jx_data = Jx.loc[f].values.reshape((len(x), len(y)))
        Jy_data = Jy.loc[f].values.reshape((len(x), len(y)))
        J_data = np.hypot(Jx_data, Jy_data)

        fig = plt.figure(figsize=(6.8 * (1 + 0.4), 3.4), layout="constrained")
        gs = fig.add_gridspec(1, 4, width_ratios=(1, 0.1, 1, 0.1), wspace=0.1)
        ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
        cax = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3])]

        cmap = sns.light_palette("#CC5803", input="html", as_cmap=True)
        ax[0].pcolormesh(x, y, Ey_data, cmap=cmap, norm=norm_Ey)
        ax[0].set(
            xlim=(-15, 15), ylim=(-15, 15), xlabel=r"x ($\mu$m)", ylabel=r"y ($\mu$m)"
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_Ey)
        fig.colorbar(sm, cax=cax[0], label=r"$|E_y|$ (MV/cm)")

        lw = 10 * J_data / J_max
        ax[1].streamplot(
            x,
            y,
            Jx_data,
            Jy_data,
            density=3,
            linewidth=lw,
            color="k",
            arrowstyle="->",
            arrowsize=0.8,
        )
        ax[1].pcolormesh(x, y, J_data, cmap=cmap, norm=norm_J)
        ax[1].set(
            xlim=(-15, 15), ylim=(-15, 15), xlabel=r"x ($\mu$m)", ylabel=r"y ($\mu$m)"
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_J)
        fig.colorbar(sm, cax=cax[1], label=r"$|J|$ (A/m)")

        dxf_file = (
            "/Users/mmatthiesen/University/Projects/Nickelate simulations/F21059.dxf"
        )
        im = image_from_dxf(dxf_file)
        x_dxf = np.linspace(data["x"].min(), data["x"].max(), im.shape[1])
        y_dxf = np.linspace(data["y"].min(), data["y"].max(), im.shape[0])
        ax[0].pcolormesh(x_dxf, y_dxf, 1 - im)
        ax[1].pcolormesh(x_dxf, y_dxf, 1 - im)

        fig.savefig(
            os.path.join(fig_dir, f"plot_f={f}THz_c={cond_list[i]:.2E}.png"),
            bbox_inches="tight",
            dpi=300,
        )

        plt.close()
