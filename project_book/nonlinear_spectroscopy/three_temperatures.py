# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: /nix/store/ndw01kv30vabg62apaa655lyl0s4lys2-python3-3.12.9-env/share/jupyter/kernels/python3
# ---

# %% [markdown]
# ---
# title: "Comparing low and moderate temperature"
# ---

# %%
# | output: false

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from scipy.integrate import simpson
import lmfit as lm
import matplotlib.gridspec as gridspec

from polars import col
from polars_complex import ccol
from polars_dataset import Dataset
from driven_nickelate.scripts.pump_probe.field_from_hwp import field_from_hwp
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import create_dataset, pl_fft
from matplotlib.patches import FancyArrowPatch

from IPython.display import display
from mplstylize import mpl_tools, colors

from cairosvg import svg2pdf
import svgutils.compose as sc 

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)


# %%
def to_field(hwp):
    return 192 * field_from_hwp(hwp)


files = pl.read_csv(
    PROJECT_PATHS.file_lists / "xHWPxT.csv",
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
    PROJECT_PATHS.file_lists / "nonlinear_probe" / "xHWPxT_REF.csv", comment_prefix="#"
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


def time_window(t, t1, t2):
    left = np.min([np.ones(len(t)), np.exp((t - t1) / 0.5)], axis=0)
    right = np.min([np.ones(len(t)), np.exp(-(t - t2) / 0.5)], axis=0)
    return left + right - 1


window = col("time").map_batches(lambda t: time_window(t, -5, 10))
window_vac = col("time").map_batches(lambda t: time_window(t, -10, -3))
waveforms = waveforms.with_columns(
    (col("X") * window).over(waveforms.id_vars).alias("X"),
    (col("X_vac") * window_vac).over(waveforms.id_vars).alias("X_vac"),
)

# %%
# | label: fig-three-temperatures-field-norm
# | fig-cap: The normalization factor (integral of the square of the field) as a function of the field strength.


def integrate_reference(s):
    t, y = s.struct[0], s.struct[1]
    return simpson(y, x=t)


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
field_norm_fit = model.fit(field_norm["norm"], params, fs=field_norm["field_strength"])


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

# %%
# | label: fig-three-temperatures-field-norm-fit
# | fig-cap: The normalization factor (integral of the square of the field) as a function of the field strength, with a polynomial fit pinned to zero.

g = sns.lineplot(field_norm, x="field_strength", y="norm", marker="o")

fs = np.arange(0, 170, 1e-1)
plt.plot(fs, field_norm_fit.eval(fs=fs), "--")
plt.scatter(0, 0)
g.set(ylim=(-1e-4, None), xlim=(-1e-4, None))
plt.show()

# %%
# | label: fig-three-temperatures-waveforms
# | fig-cap: Time-domain waveforms at three temperatures, with color representing the relative field strength (1 is about 200 kV/cm incident).

g = sns.relplot(
    waveforms.filter(col("time").is_between(-2, 10)),
    x="time",
    y="X",
    hue="field_strength",
    col="temperature",
    col_wrap=3,
    palette="flare",
    kind="line",
    height=6.8 / 3,
    aspect=1.0,
    facet_kws={"sharey": True, "despine": False},
    estimator=None
)
g.figure.set_size_inches(6.8, 6.8 / 3)
g.legend.set(title="$T$ (K)")
g.set(xlabel="$t$ (ps)", ylabel=r"$E$ (norm.)")
g.set_titles("$T$ = {col_name} K")

plt.show()

# %%
spectra = (
    Dataset(
        pl_fft(waveforms, waveforms.index, waveforms.id_vars),
        index="freq",
        id_vars=waveforms.id_vars,
    )
    .select_data(
        col("X.real", "X.imag").complex.struct("X[c]"),
        col("X_vac.real", "X_vac.imag").complex.struct("X_vac[c]"),
        col("X_ref.real", "X_ref.imag").complex.struct("X_ref[c]"),
    )
    .with_columns(
        (ccol("X[c]") / ccol("X_ref[c]")).alias("t[c]"),
    )
)

# %%
# | label: fig-three-temperatures-raw-spectra
# | fig-cap: Raw spectra of the detected waveforms.
# | fig-subcap:
# |     - Spectra for waveforms transmitted through sample.
# |     - Spectra for waveforms transmitted through vacuum only.
# | layout-nrow: 2

g = sns.relplot(
    spectra.select_data(ccol("X[c]").modulus().alias("X.mag")).filter(
        col("freq").is_between(0, 5)
    ),
    x="freq",
    y="X.mag",
    hue="field_strength",
    col="temperature",
    col_wrap=3,
    palette="flare",
    kind="line",
    height=6.8 / 3,
    aspect=1.0,
    facet_kws={"sharey": True, "despine": False},
)
g.figure.set_size_inches(6.8, 6.8 / 3)
g.set(xlabel="$f$ (THz)", ylabel=r"$E$ (norm.)")
g.set_titles("$T$ = {col_name} K")
g.legend.set(title="$E$ (kV/cm)")
plt.show()

g = sns.relplot(
    spectra.select_data(ccol("X_vac[c]").modulus().alias("X_vac.mag")).filter(
        col("freq").is_between(0, 5)
    ),
    x="freq",
    y="X_vac.mag",
    hue="field_strength",
    col="temperature",
    col_wrap=3,
    palette="flare",
    kind="line",
    height=6.8 / 3,
    aspect=1.0,
    facet_kws={"sharey": True, "despine": False},
)
g.figure.set_size_inches(6.8, 6.8 / 3)
g.set(xlabel="$f$ (THz)", ylabel=r"$E$ (norm.)")
g.set_titles("$T$ = {col_name} K")
g.legend.set(title="$E$ (kV/cm)")
plt.show()

# %%
# | label: fig-three-temperatures-amplitude
# | fig-cap: Spectra at three temperatures, with color representing the relative field strength (1 is about 200 kV/cm incident).

norm = plt.Normalize(vmin=0, vmax=200)
g = sns.relplot(
    spectra.select_data(ccol("t[c]").modulus("t.mag"))
    .filter(col("freq").is_between(0, 2.3))
    .filter(col("field_strength") > col("field_strength").min()),
    x="freq",
    y="t.mag",
    hue="field_strength",
    hue_norm=norm,
    col="temperature",
    col_wrap=3,
    palette="flare",
    kind="line",
    height=6.8 / 3,
    aspect=1.0,
    facet_kws={"sharey": True, "despine": False},
    legend=False,
)
g.figure.set_size_inches(6.8, 6.8 / 3)
g.set(ylim=(0, 2), xlabel="$f$ (THz)", ylabel=r"$|t|$")
g.set_titles("$T$ = {col_name} K")

cbar = g.fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap="flare"),
    ax=g.axes,
    label="$E$ (kV/cm)",
    pad=0.02,
)

g.fig.savefig(
    PROJECT_PATHS.figures
    / "nonlinear_spectroscopy"
    / "three_temperatures_transmission_amplitude.pdf",
)
plt.show()

# %%
# | label: fig-three-temperatures-summary
# | fig-cap: Summary of time- and frequency-domain responses at three temperatures, with color representing the relative field strength (1 is about 200 kV/cm incident).

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
gs_amp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[:, 0], hspace=0.4)

axis_labels = [["wf_0", "wf_1", "wf_2", "cbar"], ["sp_0", "sp_1", "sp_2", "."]]
ax = {
    axis_labels[i][j]: fig.add_subplot(gs_panels[i, j])
    for i in range(2)
    for j in range(4)
}
ax["."].remove()

def set_common_xlabel(fig, subplot_spec, label):
    big_ax = plt.Subplot(fig, subplot_spec, frameon=False)
    big_ax.set(xticks=[], yticks=[])
    big_ax.patch.set_facecolor("none")
    big_ax.set_xlabel(label, labelpad=20)
    fig.add_subplot(big_ax)


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
        spectra.select_data(ccol("t[c]").modulus("t.mag")).filter(
            col("freq").is_between(0.3, 2.2) & col("temperature").eq(temperatures[i])
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
    label=r"$\alpha\,({\rm kV/cm})$",
)

resonance_amplitude = (
    spectra.filter(col("freq").is_between(0.98, 1.02))
    .select_data(ccol("t[c]").modulus("t.mag"))
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

ax[f"wf_0"].set(ylabel=r"$E$ (norm.)", ylim=(-0.5, 0.5))
ax[f"sp_0"].set(ylabel=r"$|t|$", ylim=(0, 2))
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

plt.show()
