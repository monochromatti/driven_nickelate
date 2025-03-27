import logging

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import Normalize
from polars_complex import ccol
from polars_dataset import Datafile

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.simulations.plotting import plot_dxf

STORE_DIR = PROJECT_PATHS.processed_data / "simulations/surface/"
STORE_DIR.mkdir(parents=True, exist_ok=True)
DATAFILES = {
    "surface_data": Datafile(
        path=STORE_DIR / "surface_data.csv",
        index="freq",
        id_vars=["x", "y", "conductivity"],
    ),
}


def struct_amplitude(x: pl.Struct) -> pl.Series:
    amplitude = np.linalg.norm((x.struct[0], x.struct[1]), axis=0)
    return pl.Series(amplitude, dtype=pl.Float64)


def load_data(datafiles):
    surface_data = (
        pl.concat(
            [
                pl.scan_csv(x[1]).with_columns(pl.lit(x[0]).alias("conductivity"))
                for x in datafiles
            ]
        )
        .with_columns(ccol("real_value", "imag_value").modulus().alias("magnitude"))
        .sort("variable", "y", "x")
        .collect()
    )

    return surface_data


def ndarray_rectangle(df: pl.DataFrame, shape: tuple) -> np.ndarray:
    return df.to_numpy().reshape(shape)


def create_figures(plot_data, dxf_path, save_path):
    Ey_max = plot_data["Ey"].max()
    J_max = (plot_data["Jx"] ** 2 + plot_data["Jy"] ** 2).sqrt().max()

    for (conductivity,), data in plot_data.group_by(["conductivity"]):
        logging.info("Creating figures for conductivity %s ...", conductivity)

        x = data["x"].unique(maintain_order=True)
        y = data["y"].unique(maintain_order=True)

        nx, ny = len(x), len(y)

        norm_Ey = Normalize(vmin=-Ey_max, vmax=Ey_max)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        for f in sorted(data["freq"].unique()):
            logging.info("Creating figure for frequency %s ...", f)

            data_freqfilt = data.filter(pl.col("freq") == f)
            Ey_data = ndarray_rectangle(data_freqfilt["Ey"], (nx, ny))
            Jx_data = ndarray_rectangle(data_freqfilt["Jx"], (nx, ny))
            Jy_data = ndarray_rectangle(data_freqfilt["Jy"], (nx, ny))

            fig = plt.figure(figsize=(3.4, 2.6))
            ax = fig.subplot_mosaic(
                [["plot", "cbar"]],
                width_ratios=(1, 0.08),
                gridspec_kw={"wspace": 0.04},
            )
            ax["plot"].set_box_aspect(1)

            ax["plot"].pcolormesh(x, y, Ey_data, cmap=cmap, norm=norm_Ey)
            ax["plot"].set(
                xlim=(-15, 15),
                ylim=(-15, 15),
                xlabel=r"x ($\mu$m)",
                ylabel=r"y ($\mu$m)",
            )

            fig.colorbar(
                plt.cm.ScalarMappable(cmap=cmap, norm=norm_Ey),
                cax=ax["cbar"],
                label=r"$E_y$ (norm.)",
            )

            linewidth = np.sqrt(Jx_data**2 + Jy_data**2) / J_max * 2
            ax["plot"].streamplot(
                x,
                y,
                Jx_data,
                Jy_data,
                density=3,
                color="k",
                linewidth=linewidth,
                arrowstyle="-",
                arrowsize=0.8,
            )

            plot_dxf(dxf_path, ax=ax["plot"])

            fig.savefig(
                save_path / f"plot_f={f}THz_c={conductivity:.2E}.png",
                bbox_inches="tight",
                dpi=600,
            )

            plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading data ...")
    data_folder = PROJECT_PATHS.root / "simulations/data"
    surface_data = load_data(
        [
            (0.00e00, data_folder / "15.06.23/surface_0.00E+00.csv"),
            (4.50e07, data_folder / "15.06.23/surface_4.50E+07.csv"),
        ]
    )

    Ey = (
        surface_data.filter(pl.col("variable") == "Ey (V/m)")
        .select("x", "y", "freq", "conductivity", "real_value")
        .rename({"real_value": "Ey"})
    )
    Jx = (
        surface_data.filter(pl.col("variable") == "Jsupx (A/m)")
        .select("x", "y", "freq", "conductivity", "real_value")
        .rename({"real_value": "Jx"})
    )
    Jy = (
        surface_data.filter(pl.col("variable") == "Jsupy (A/m)")
        .select("x", "y", "freq", "conductivity", "real_value")
        .rename({"real_value": "Jy"})
    )
    surface_data = (
        Ey.join(Jx, on=("x", "y", "freq", "conductivity"))
        .join(Jy, on=("x", "y", "freq", "conductivity"))
        .with_columns(
            (pl.col("Ey") / pl.col("Ey").max()).alias("Ey"),
            ccol("Jx", "Jy").modulus().alias("J"),
        )
        .with_columns(
            (pl.col("Jx") / pl.col("J").max()).alias("Jx"),
            (pl.col("Jy") / pl.col("J").max()).alias("Jy"),
        )
    )
    logging.info("Saving data ...")
    DATAFILES["surface_data"].write(surface_data)

    logging.info("Creating figures ...")
    dxf_path = PROJECT_PATHS.root / "device_design/uc.dxf"
    figure_path = PROJECT_PATHS.figures / "simulations/surface_maps"
    figure_path.mkdir(parents=True, exist_ok=True)
    create_figures(surface_data, dxf_path=dxf_path, save_path=figure_path)
