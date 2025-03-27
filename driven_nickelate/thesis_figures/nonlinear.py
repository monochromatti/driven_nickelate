from pathlib import Path

import lmfit as lm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import svgutils.compose as sc
from cairosvg import svg2pdf
from mplstylize import mpl_tools
from polars import col
from polars_complex import ccol
from polars_dataset import Dataset
from scipy.integrate import simpson

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.scripts.pump_probe.field_from_hwp import field_from_hwp
from driven_nickelate.tools import create_dataset, pl_fft


def to_field(hwp):
    return 192 * field_from_hwp(hwp)


def time_window(t, t1, t2):
    left = np.min([np.ones(len(t)), np.exp((t - t1) / 0.5)], axis=0)
    right = np.min([np.ones(len(t)), np.exp(-(t - t2) / 0.5)], axis=0)
    return left + right - 1


def integrate_reference(s):
    t, y = s.struct[0], s.struct[1]
    return simpson(y, x=t)


def set_common_xlabel(fig, subplot_spec, label):
    big_ax = plt.Subplot(fig, subplot_spec, frameon=False)
    big_ax.set(xticks=[], yticks=[])
    big_ax.patch.set_facecolor("none")
    big_ax.set_xlabel(label, labelpad=20)
    fig.add_subplot(big_ax)


def import_waveforms():
    files = pl.read_csv(
        PROJECT_PATHS.file_lists / "nonlinear_probe" / "xHWPxT.csv",
        comment_prefix="#",
    ).select(
        col("HWP position")
        .map_elements(to_field, return_dtype=pl.Float64)
        .alias("field_strength"),
        col("Temperature").alias("temperature"),
        col("Path")
        .map_elements(lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8)
        .alias("path"),
    )
    waveforms = (
        create_dataset(
            files,
            column_names=["delay", "X", "Y"],
            index="delay",
            lockin_schema={"X": ("X", "Y")},
        )
        .with_columns(((2 / 0.29979) * (177.6 - col("delay"))).alias("time"))
        .set(index="time")
        .select(
            col("field_strength"),
            col("temperature"),
            col("time"),
            ((col("X") - col("X").mean()).over("field_strength")).alias("X"),
        )
    )

    files_ref = pl.read_csv(
        PROJECT_PATHS.file_lists / "nonlinear_probe" / "xHWPxT_REF.csv",
        comment_prefix="#",
    ).select(
        col("HWP position")
        .map_elements(to_field, return_dtype=pl.Float64)
        .alias("field_strength"),
        col("Path")
        .map_elements(lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8)
        .alias("path"),
    )

    waveforms_vac = (
        create_dataset(
            files_ref,
            column_names=["delay", "X", "Y"],
            index="delay",
            lockin_schema={"X": ("X", "Y")},
        )
        .with_columns(
            ((2 / 0.29979) * (177.6 - col("delay"))).alias("time"),
            (col("X") - col("X").first()).alias("X"),
        )
        .set(index="time")
        .drop("delay")
    )

    time = pl.Series("time", np.arange(-10, 30, 6e-2))
    waveforms = waveforms.regrid(time, method="cosine", fill_value=0)
    waveforms_vac = waveforms_vac.regrid(time, method="cosine", fill_value=0)

    waveforms = waveforms.join(
        waveforms_vac, on=["time", "field_strength"], suffix="_vac"
    ).sort("time")

    window = col("time").map_batches(lambda t: time_window(t, -5, 10))
    window_vac = col("time").map_batches(lambda t: time_window(t, -10, -3))
    waveforms = waveforms.with_columns(
        (col("X") * window).over(waveforms.id_vars).alias("X"),
        (col("X_vac") * window_vac).over(waveforms.id_vars).alias("X_vac"),
    )

    return waveforms, waveforms_vac


def normalize_waveforms(waveforms, waveforms_vac):
    field_norm = (
        waveforms_vac.group_by("field_strength")
        .agg(
            pl.struct(col("time"), col("X") ** 2)
            .map_batches(lambda s: integrate_reference(s), return_dtype=pl.Float64)
            .get(0)
            .alias("norm")
        )
        .sort("field_strength")
    )

    model = lm.Model(lambda fs, a, b, c: fs * (a + fs * (b + fs * c)))
    params = model.make_params(a=1e-7, b=1e-7, c=0)
    field_norm_fit = model.fit(
        field_norm["norm"], params, fs=field_norm["field_strength"]
    )

    waveforms = (
        waveforms.with_columns(
            col("field_strength")
            .map_batches(lambda fs: field_norm_fit.eval(fs=fs))
            .sqrt()
            .alias("field_norm")
        )
        .with_columns(
            (col("X") / col("field_norm")).over("field_strength").alias("X"),
            (col("X_vac") / col("field_norm")).over("field_strength").alias("X_vac"),
        )
        .set(id_vars=waveforms.id_vars + ["field_norm"])
    )

    waveforms_ref = waveforms.filter(
        (col("temperature") == col("temperature").min())
        & (col("field_strength") == col("field_strength").min())
    ).select("time", "X")

    waveforms = waveforms.join(waveforms_ref, suffix="_ref", on=["time"])

    return waveforms


if __name__ == "__main__":
    waveforms = normalize_waveforms(*import_waveforms())

    spectra = (
        Dataset(
            pl_fft(waveforms, waveforms.index, waveforms.id_vars),
            index="freq",
            id_vars=waveforms.id_vars,
        )
        .select_data(
            ccol("X.real", "X.imag").alias("X[c]", fields="X"),
            ccol("X_vac.real", "X_vac.imag").alias("X_vac[c]", fields="X_vac"),
            ccol("X_ref.real", "X_ref.imag").alias("X_ref[c]", fields="X_ref"),
        )
        .with_columns(
            (ccol("X[c]") / ccol("X_ref[c]")).alias("t[c]", fields="t"),
        )
    )

    fig = plt.figure(figsize=(6.8, 5.1), layout="none")

    gs_outer = gridspec.GridSpec(2, 2, width_ratios=[1, 3], wspace=0.25)
    gs_panels = gridspec.GridSpecFromSubplotSpec(
        2,
        4,
        subplot_spec=gs_outer[:, 1],
        width_ratios=[1, 1, 1, 0.05],
        wspace=0,
        hspace=0.4,
    )
    gs_amp = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[:, 0], hspace=0.4
    )

    axis_labels = [["wf_0", "wf_1", "wf_2", "cbar"], ["sp_0", "sp_1", "sp_2", "."]]
    ax = {
        axis_labels[i][j]: fig.add_subplot(gs_panels[i, j])
        for i in range(2)
        for j in range(4)
    }
    ax["."].remove()

    set_common_xlabel(fig, gs_panels[0, :-1], r"$t$ (ps)")
    set_common_xlabel(fig, gs_panels[1, :-1], r"$f$ (THz)")

    temperatures = waveforms.coord("temperature").sort()
    field_strength = waveforms.coord("field_strength")

    color_norm = plt.Normalize(0, 200)
    three_colors = ["#ACBEA3", "#AD5D4E", "#DDA448"]
    palettes = [sns.light_palette(color, as_cmap=True) for color in three_colors]
    for i in range(3):
        sns.lineplot(
            waveforms.filter(
                col("time").is_between(-2, 7) & col("temperature").eq(temperatures[i])
            ),
            x="time",
            y="X",
            hue="field_strength",
            hue_norm=color_norm,
            palette=palettes[i],
            legend=False,
            ax=ax[f"wf_{i}"],
        )
        sns.lineplot(
            spectra.select_data(ccol("t[c]").modulus().alias("t.mag")).filter(
                col("freq").is_between(0.3, 2.2)
                & col("temperature").eq(temperatures[i])
            ),
            x="freq",
            y="t.mag",
            hue="field_strength",
            hue_norm=color_norm,
            palette=palettes[i],
            ax=ax[f"sp_{i}"],
            legend=False,
        )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=color_norm, cmap=sns.color_palette("Grays", as_cmap=True)
        ),
        cax=ax["cbar"],
        label=r"$E_0\,({\rm kV/cm})$",
    )

    resonance_amplitude = (
        spectra.filter(col("freq").is_between(0.98, 1.02))
        .select_data(ccol("t[c]").modulus().alias("t.mag"))
        .group_by("temperature", "field_strength")
        .mean()
        .sort("temperature", "field_strength")
    )

    ax["amp"] = fig.add_subplot(gs_amp[1])
    ax["amp"].sharey(ax["sp_0"])
    g = sns.lineplot(
        resonance_amplitude.with_columns(col("field_strength") * 1e-3),
        x="field_strength",
        y="t.mag",
        hue="temperature",
        marker="o",
        palette=three_colors,
        ax=ax["amp"],
        legend=False,
    )
    g.set(xlim=(-0.01, 0.21), xticks=[0, 0.1, 0.2], xlabel="$E$ (MV/cm)", ylabel=None)

    ax["wf_0"].set(ylabel=r"$E$ (norm.)", ylim=(-0.5, 0.5))
    ax["sp_0"].set(ylabel=r"$|t|$", ylim=(0, 2))
    for i in range(3):
        ax["wf_%d" % i].set(ylim=(-0.35, 0.35), xlabel="", xlim=(-1, 5))
        ax["wf_%d" % i].xaxis.set_major_locator(plt.MaxNLocator(3))
        ax["wf_%d" % i].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1f}")
        )
        ax["wf_%d" % i].text(
            4, 0.3, f"{temperatures[i]} K", color=three_colors[i], ha="right"
        )
        ax["sp_%d" % i].set(ylim=(0, 2.0), xlabel=None, xticks=np.arange(0.5, 2.5, 0.5))
        if i > 0:
            ax["wf_%d" % i].set(ylabel="", yticks=[])
            ax["sp_%d" % i].set_yticks([])
        ax["sp_%d" % i].set_ylabel("")

    ax["amp"].set(ylabel=r"$|t|$")

    ax.pop("cbar")
    mpl_tools.enumerate_axes([ax["wf_0"], ax["sp_0"], ax["amp"]])
    mpl_tools.breathe_axes([_ax for key, _ax in ax.items() if "sp" in key], axis="y")
    mpl_tools.breathe_axes([ax["wf_%d" % i] for i in range(3)], axis="y")

    fig.tight_layout(pad=0)

    x, y = 72 * fig.get_size_inches()
    sc.Figure(
        f"{x}pt",
        f"{y}pt",
        sc.MplFigure(fig),
        sc.SVG(PROJECT_PATHS.figures / "illustrations/nonlinear_spectroscopy.svg")
        .scale(72 / 600 / 2)
        .move(10, 43),
    ).save("_tmp.svg")

    svg2pdf(
        url="_tmp.svg",
        write_to=str(
            PROJECT_PATHS.figures / "nonlinear_spectroscopy" / "three_temperatures.pdf"
        ),
    )

    Path("./_tmp.svg").unlink()
