# %% [markdown]
# # Field enhancement

# %%
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from IPython.display import display
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mplstylize import mpl_tools
from polars import col

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.simulations.field_enhancement import (
    DATAFILES as ENHANCEMENT_DATAFILES,
)
from driven_nickelate.simulations.gapfield import (
    DATAFILES as GAPFIELD_DATAFILES,
)
from driven_nickelate.simulations.plotting import plot_dxf
from driven_nickelate.simulations.surface import (
    DATAFILES as SURFACE_DATAFILES,
)

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %% [markdown]
# The fundamental mode of the split-ring resonator involves alternating circulating currents that accumulate charge at the resonator central gap. This charge accumulation produces an evanescent electric field which is larger than the incident electric field. The field enhancement is calculated as
#
# $$
#     \left|E/E_0\right| = \left|\frac{E_y (\mathbf{r}=\mathbf{0}) }{E_{y,0}(\mathbf{r}=\mathbf{0})}\right|
# $$
#
# where $\mathbf{r} = \mathbf{0}$ corresponds to the point at the center of the resonator gap, at the surface. The reference field $E_{y,0}$ is the field at the same point with the resonator removed.

# %%
field_enhancement = ENHANCEMENT_DATAFILES["field_enhancement"].load()
display(field_enhancement)

# %% [markdown]
# In @fig-field-enhancement, we show the field enhancement at 1 THz, both in the $xy$-plane and the profile extending away from the surface.
#

# %%
X, Y, Z = field_enhancement.get_columns()
nx, ny = len(X.unique()), len(Y.unique())
X = X.to_numpy().reshape((nx, ny))
Y = Y.to_numpy().reshape((nx, ny))
Z = Z.to_numpy().reshape((nx, ny))

fig, ax = plt.subplots(figsize=(6.8, 4.7))

ax.pcolormesh(X, Y, Z, cmap="sunlight_r", zorder=-1)
dxf_file = PROJECT_PATHS.root / "device_design/uc.dxf"
plot_dxf(dxf_file, ax=ax, layers=["structure", "unit_cell"])
cbar = ax.figure.colorbar(
    plt.cm.ScalarMappable(cmap="sunlight_r", norm=Normalize(Z.min(), Z.max())),
    ax=ax,
    label=r"$\left|E/E_0\right|$",
    pad=0.02,
)
ax.set_box_aspect((Y.max() - Y.min()) / (X.max() - X.min()))
ax.set(
    xlabel=r"x ($\mu$m)",
    ylabel=r"y ($\mu$m)",
    xlim=(-18.2, 18.2),
    ylim=(-14.3, 14.3),
)

plt.show()

# %% [markdown]
# We have also calculated the surface current distributions,

# %%
surface_data = (
    SURFACE_DATAFILES["surface_data"]
    .load()
    .filter((pl.col("freq") == 1.0) & (pl.col("conductivity") == 0))
).select_data("Jx", "Jy")
display(surface_data)

# %% [markdown]
# as well as the perpendicular field distribution. These give a full picture of the cavity mode, and the nature of the field enhancement, summarized in @fig-mode-summary.

# %%
norm = Normalize(
    *field_enhancement.select(
        pl.col("field_enhancement").min().alias("min"),
        pl.col("field_enhancement").max().alias("max"),
    ).row(0)
)

X, Y, Z = field_enhancement.get_columns()
nx, ny = len(X.unique()), len(Y.unique())
X = X.to_numpy().reshape((nx, ny))
Y = Y.to_numpy().reshape((nx, ny))
Z = Z.to_numpy().reshape((nx, ny))

fig = plt.figure(figsize=(6.8, 4.2))
ax = fig.subplot_mosaic(
    [["cbar", "."], ["surface", "profile"]],
    gridspec_kw={
        "height_ratios": [0.05, 1],
        "width_ratios": [1, 0.5],
        "wspace": 0.05,
    },
)

aspect = (Y.max() - Y.min()) / (X.max() - X.min())
ax["surface"].set_box_aspect(aspect)

ax["surface"].pcolormesh(X, Y, Z, norm=norm, cmap="sunlight_r", zorder=-1)

sm = plt.cm.ScalarMappable(cmap="sunlight_r", norm=norm)
fig.colorbar(
    sm,
    cax=ax["cbar"],
    label=r"$\left|E/E_0\right|$",
    orientation="horizontal",
    location="top",
)

dxf_file = PROJECT_PATHS.root / "device_design/uc.dxf"
plot_dxf(dxf_file, ax=ax["surface"], layers=["structure", "unit_cell"])

ax["surface"].set(
    xlabel=r"x ($\mu$m)",
    ylabel=r"y ($\mu$m)",
    xlim=(-18.2, 18.2),
    ylim=(-14.3, 14.3),
)

data = pl.read_csv(
    PROJECT_PATHS.root / "simulations/data/24.02.24/Ey_zcut.csv",
    comment_prefix="%",
    new_columns=["z", "Ey"],
).with_columns(pl.col("Ey") / pl.col("Ey").max() * Z.max())

g = sns.lineplot(
    data.sort("z"),
    x="Ey",
    y="z",
    sort=False,
    ax=ax["profile"],
    color="black",
)
g.set(xlabel=r"$\left|E/E_0\right|$", ylabel=r"z ($\mu$m)")


def decay_model(z, a, b, c):
    """Arbitrary rational decay"""
    return a / (b + z**c)


def decay_length(z, values, critical_value):
    """1/e decay length"""
    return z[int(np.abs(result.best_fit - result.eval(z=0) / np.exp(1)).argmin())]


model = lm.Model(decay_model)
params = model.make_params(a=100, b=1, c=1)

z, E = data.filter(pl.col("z") > 0).sort("z").get_columns()
result = model.fit(data=E, z=z, params=params)
delta_pos = decay_length(z, result.best_fit, result.eval(z=0) / np.exp(1))

z, E = data.filter(pl.col("z") < 0).sort("z").get_columns()
result = model.fit(data=E, z=-z, params=params)
delta_neg = decay_length(z, result.best_fit, result.eval(z=0) / np.exp(1))

ax["profile"].axhspan(
    delta_neg,
    delta_pos,
    alpha=1.0,
    ls="--",
    lw=0.6,
    color="k",
    fc=plt.get_cmap("sunlight_r")(0),
)
ax["profile"].set(ylim=(-5, 5))

ax["profile"].annotate(
    rf"$\delta = {(delta_pos - delta_neg) / 2: .2f}\,\mathrm{{\mu m}}$",
    xy=(25, 0),
    ha="center",
    va="center",
    # fontsize=8,
)
mpl_tools.breathe_axes(ax["profile"], "y", 0.03)
mpl_tools.enumerate_axes((ax["surface"], ax["profile"]))


surface_data = (
    SURFACE_DATAFILES["surface_data"]
    .load()
    .filter((pl.col("freq") == 1.0) & (pl.col("conductivity") == 0))
)

x, y, Jx, Jy = surface_data.fetch("x", "y", "Jx", "Jy")
nx, ny = len(x.unique()), len(y.unique())
x = np.reshape(x, (nx, ny))
y = np.reshape(y, (nx, ny))
Jx = np.reshape(Jx, (nx, ny))
Jy = np.reshape(Jy, (nx, ny))
linewidth = np.sqrt(Jx**2 + Jy**2)
linewidth *= 5 / linewidth.max()

import matplotlib as mpl

amin = 0.1
cmap = mpl.colors.ListedColormap(
    [(0, 0, 0, np.sqrt(amin + (a - amin) / (10 + 1))) for a in range(10)]
)

ax["surface"].streamplot(
    *(x, y),
    *(Jx, Jy),
    # linewidth=linewidth,
    arrowstyle="-",
    density=4,
    color=linewidth,
    cmap=sns.blend_palette(["#fff9f0", "#000000", "#000000"], as_cmap=True),
    zorder=-1,
)


fig.savefig(
    PROJECT_PATHS.figures / "simulations/field_enhancement.pdf", bbox_inches="tight"
)

plt.show()

# %% [markdown]
# The field enhancement is largest at the center of the resonator gap, and decays rapidly away from it. The decay length, defined as the distance at which the field enhancement decays to $1/e$ of its maximum value, is $\delta = 0.6\,\mu\mathrm{m}$.

# %%
gap_spectra = (
    GAPFIELD_DATAFILES["spectra"]
    .load()
    .select_data("Ey.mag", "E0y.mag")
    .filter(pl.col("freq") > 0.1)
    .regrid(pl.Series("freq", np.arange(0.1, 7.0, 1e-3)), method="catmullrom")
    .regrid(pl.Series("cond", np.geomspace(10.0, 5e6, 50)), method="catmullrom")
    .drop_nulls()
)
display(gap_spectra)

# %%
from matplotlib.colors import LogNorm

fig = plt.figure(figsize=(6.8, 4.2), constrained_layout=True)
ax = fig.subplot_mosaic("ab;AB", height_ratios=(0.05, 1.0))
ax["A"].set_box_aspect(1)
ax["B"].set_box_aspect(1)

freq_upper = 7
plot_data = gap_spectra.filter(pl.col("freq") < freq_upper).with_columns(
    pl.col("Ey.mag") * 1e-8, pl.col("E0y.mag") * 1e-8
)

condnorm = LogNorm(100, gap_spectra["cond"].max())
plot_kwargs = {
    "x": "freq",
    "hue": "cond",
    "hue_norm": condnorm,
    "legend": False,
    "errorbar": None,
    "palette": "Blues_r",
}
sns.lineplot(
    data=plot_data,
    y="Ey.mag",
    ax=ax["A"],
    **plot_kwargs,
)
sns.lineplot(
    data=plot_data.filter(pl.col("cond") == pl.col("cond").min()),
    x="freq",
    y="E0y.mag",
    ax=ax["A"],
    **dict(ls="--", lw=0.8, color="black"),
)
ax["A"].set(
    xlim=(0, 6),
    xlabel=r"$f$ (THz)",
    ylabel=r"$|E|$ (MV/cm)",
    ylim=(1e-5, 1e-1),
    yscale="log",
)
sm = plt.cm.ScalarMappable(cmap="Blues_r", norm=condnorm)
fig.colorbar(
    sm,
    cax=ax["a"],
    label=r"$\sigma$ (S/m)",
    orientation="horizontal",
    location="top",
)
fieldnorm = LogNorm(plot_data["Ey.mag"].min(), plot_data["Ey.mag"].max())

mesh_data = plot_data.select("freq", "cond", "Ey.mag").sort("freq", "cond")
freq = mesh_data["freq"].unique(maintain_order=True)
cond = mesh_data["cond"].unique(maintain_order=True)
magnitude = mesh_data.pivot(
    index="freq", columns="cond", values="Ey.mag", aggregate_function=None
).df.drop("freq")
ax["B"].pcolormesh(
    cond,
    freq,
    magnitude,
    norm=fieldnorm,
    shading="gouraud",
    cmap="roma_r",
)
ax["B"].set(
    xlabel=r"$\sigma$ (S/m)",
    ylabel=r"$f$ (THz)",
    xscale="log",
    ylim=(None, freq_upper - 0.2),
    xlim=(10, 3e6),
)
sm = plt.cm.ScalarMappable(cmap="roma_r", norm=fieldnorm)
fig.colorbar(
    sm,
    cax=ax["b"],
    label=r"$|E|$ (MV/cm)",
    orientation="horizontal",
    location="top",
)

ax_inset = ax["A"].inset_axes([0.6, 0.6, 0.35, 0.35])
sns.lineplot(
    plot_data.filter(col("freq").is_between(1.04, 1.15)).group_by("cond").max(),
    x="cond",
    y="Ey.mag",
    ax=ax_inset,
    color="black",
).set(xscale="log", ylabel="$|E| (f_0)$", xlabel=r"$\sigma$ (S/m)")

mpl_tools.enumerate_axes([ax["A"], ax["B"]])

fig.savefig(
    PROJECT_PATHS.figures / "simulations/gapfield_spectra.pdf", bbox_inches="tight"
)
plt.show()


# %%
time_data = GAPFIELD_DATAFILES["temporal"].load()
time_data = time_data.regrid(
    pl.Series("cond", np.geomspace(1e2, time_data["cond"].max(), 50)),
    method="catmullrom",
).drop_nulls()
display(time_data)

# %%
fig = plt.figure(figsize=(6.8, 4.2), layout="constrained")
ax = fig.subplot_mosaic("ab;AB", height_ratios=(0.05, 1))
ax["A"].set_box_aspect(1)
ax["B"].set_box_aspect(1)

iax = inset_axes(ax["A"], width="35%", height="30%", loc="upper right", borderpad=0.0)

condnorm = LogNorm(100, time_data["cond"].max())
sns.lineplot(
    time_data.filter(pl.col("cond").is_in(pl.col("cond").unique().gather_every(4))),
    x="time",
    y="Ey",
    ax=ax["A"],
    hue="cond",
    hue_norm=condnorm,
    palette="Spectral",
    legend=False,
).set(xlabel=r"$t$ (ps)", ylabel=r"$E$ (MV/cm)", ylim=(-6, 6))

reference_data = (
    time_data.group_by("time").agg(pl.col("E0y").first().alias("E0y")).sort("time")
)
iax.plot(reference_data["time"], reference_data["E0y"], color="black")
iax.set(xlim=(0, 4), xlabel=r"$t$ (ps)", ylabel=r"$E_0$ (MV/cm)", yticks=(-0.2, 0, 0.2))

sm = plt.cm.ScalarMappable(
    cmap="Spectral",
    norm=condnorm,
)
fig.colorbar(
    sm,
    cax=ax["a"],
    label=r"$\sigma$ (S/m)",
    orientation="horizontal",
    location="top",
)

mesh_data = time_data.select("time", "cond", "Ey").sort("time", "cond")
time = mesh_data["time"].unique(maintain_order=True)
cond = mesh_data["cond"].unique(maintain_order=True)
field = mesh_data.pivot(
    index="cond", columns="time", values="Ey", aggregate_function=None
).drop("cond")
ax["B"].pcolormesh(time, cond, field, shading="gouraud", cmap="RdBu")
ax["B"].set(
    xlabel=r"$t$ (ps)",
    ylabel=r"$\sigma$ (S/m)",
    yscale="log",
    ylim=(None, 3e6),
)
sm = plt.cm.ScalarMappable(
    cmap="RdBu",
    norm=Normalize(time_data["Ey"].min(), time_data["Ey"].max()),
)
fig.colorbar(
    sm,
    cax=ax["b"],
    label=r"$E$ (MV/cm)",
    orientation="horizontal",
    location="top",
)

mpl_tools.enumerate_axes([ax["A"], ax["B"]])
mpl_tools.breathe_axes(ax["A"], axis="y")
mpl_tools.breathe_axes(iax, "both", 0.2)

fig.savefig(
    PROJECT_PATHS.figures / "simulations/gapfield_temporal.pdf",
)

plt.show()

# %%
from scipy.special import erf


def oscillation_model(t, t0, a1, f1, tr1, td1, p1, a2, f2, tr2, td2, p2):
    t = t - t0
    osc1 = (
        a1 * np.sin(2 * np.pi * f1 * t + p1) * np.exp(-t / td1) * (1 + erf(t / tr1)) / 2
    )
    osc2 = (
        a2 * np.cos(2 * np.pi * f2 * t + p2) * np.exp(-t / td2) * (1 + erf(t / tr2)) / 2
    )
    return osc1 + osc2


model = lm.Model(oscillation_model)
osc1_dict = dict(a1=-5, f1=1.0, tr1=0.2, td1=3, p1=0)
osc2_dict = dict(a2=6.0, f2=1.6, tr2=0.2, td2=0.5, p2=0)
params = model.make_params(t0=1.14, **osc1_dict, **osc2_dict)

params["t0"].set(vary=True, min=1.1, max=1.2)

params["a1"].set(min=-6, max=0)
params["f1"].set(min=0, max=1.2, vary=1)
params["tr1"].set(min=0.1, max=0.4, vary=0)
params["td1"].set(min=0.3, max=5)
# params["p1"].set(min=0)

params["a2"].set(min=0, max=10)
params["f2"].set(min=1.2, max=2.0)
params["tr2"].set(expr="tr1")
params["td2"].set(min=0.1, max=5)
params["a2"].set(min=0, max=10)

results = {}
import logging

# i = 0
for (cond,), df in time_data.group_by(["cond"], maintain_order=True):
    try:
        t, E = df.select("time", "Ey").sort("time").get_columns()
        result = model.fit(data=E, t=t, params=params)
        results[cond] = result
        # plt.plot(t, i + E / E.max(), label=cond)
        # plt.plot(t, i + result.best_fit / E.max(), c="k")
        # i += 1
    except Exception:
        logging.warning("Failed to fit model for cond=%s", cond)
        # ax.plot(t, E, label=cond)
        continue


# %%
best_params = (
    pl.concat(
        pl.DataFrame(result.params.valuesdict()).with_columns(
            pl.lit(cond).alias("cond")
        )
        for cond, result in results.items()
    )
    # .with_columns(pl.col("phase") / np.pi)
    .unpivot(index="cond")
)

label_map = {
    "a1": r"$E_0$ (MV/cm)",
    "f1": r"$f_0$ (THz)",
    "td1": r"$\tau_\mathrm{d}$ (ps)",
    "p1": r"$\phi$ $(\pi)$",
    "a2": r"$E_0$ (MV/cm)",
    "f2": r"$f_0$ (THz)",
    "td2": r"$\tau_\mathrm{d}$ (ps)",
    "p2": r"$\phi$ $(\pi)$",
}

best_params = best_params.filter(pl.col("variable").is_in(label_map.keys()))

g = sns.relplot(
    best_params,
    x="cond",
    y="value",
    col="variable",
    col_wrap=2,
    height=1.7,
    facet_kws=dict(sharey=False, despine=False),
    kind="line",
    marker="o",
    color="black",
)
g.figure.set_size_inches((6.8, 5.1))
g.set_titles("{col_name}")
g.set_xlabels(r"$\sigma$ (S/m)")
g.set(xscale="log")

for ax in g.axes:
    param_name = ax.title.get_text()
    label = label_map.get(param_name)
    if label:
        ax.set(ylabel=label, title="")

mpl_tools.enumerate_axes(g.axes)
plt.show()

# %%
label_map = {
    "a": r"$E_0$ (MV/cm)",
    "f": r"$f_0$ (THz)",
    "td": r"$\tau_\mathrm{d}$ (ps)",
    "p": r"$\phi$ ($\pi$)",
}

best_params = pl.concat(
    pl.DataFrame(result.params.valuesdict()).with_columns(pl.lit(cond).alias("cond"))
    for cond, result in results.items()
).with_columns(pl.col(r"^a.*$").abs(), pl.col(r"^p.*$") / np.pi)

params1 = (
    best_params.select("cond", "a1", "f1", "td1", "p1")
    .unpivot(index=("cond", "a1"))
    .rename({"a1": "amplitude"})
    .with_columns(
        pl.lit("1").alias("resonance_id"),
        pl.col("variable").str.replace("1", ""),
    )
)
params2 = (
    best_params.select("cond", "a2", "f2", "td2", "p2")
    .unpivot(index=("cond", "a2"))
    .rename({"a2": "amplitude"})
    .with_columns(
        pl.lit("2").alias("resonance_id"),
        pl.col("variable").str.replace("2", ""),
    )
)

best_params = pl.concat([params1, params2]).sort("cond")

g = sns.relplot(
    best_params,
    x="cond",
    y="value",
    col="variable",
    col_wrap=3,
    height=1.7,
    hue="resonance_id",
    size="amplitude",
    facet_kws=dict(sharey=False, despine=False),
    legend=False,
)
g.figure.set_size_inches((6.8, 4.1))
g.set_titles("{col_name}")
g.set_xlabels(r"$\sigma$ (S/m)")
g.set(xscale="log")

for ax in g.axes:
    param_name = ax.title.get_text()
    label = label_map.get(param_name)
    if label:
        ax.set(ylabel=label, title="")

mpl_tools.enumerate_axes(g.axes)

g.figure.savefig(PROJECT_PATHS.figures / "simulations/gapfield_temporal_fits.pdf")
plt.show()

# %%
fig = plt.figure(figsize=(6.8, 5.1), layout="constrained")
ax = fig.subplot_mosaic(
    [["time", "f"], ["time", "td"], ["time", "p"]], width_ratios=(2, 1)
)

iax = inset_axes(
    ax["time"], width="35%", height="30%", loc="upper right", borderpad=0.0
)

condnorm = LogNorm(100, time_data["cond"].max())
sns.lineplot(
    time_data.filter(pl.col("cond").is_in(pl.col("cond").unique().gather_every(4))),
    x="time",
    y="Ey",
    ax=ax["time"],
    hue="cond",
    hue_norm=condnorm,
    palette="Spectral",
    legend=False,
)
ax["time"].set(xlabel=r"$t$ (ps)", ylabel=r"$E$ (MV/cm)", ylim=(-6, 6))
ax["time"].yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

reference_data = (
    time_data.group_by("time").agg(pl.col("E0y").first().alias("E0y")).sort("time")
)
iax.plot(reference_data["time"], reference_data["E0y"], color="black")
iax.set(
    xlim=(0, 4),
    xlabel=r"$t$ (ps)",
    ylabel=r"$E_0$ (MV/cm)",
    yticks=(-0.2, 0, 0.2),
    ylim=(-0.25, 0.25),
)

sm = plt.cm.ScalarMappable(
    cmap="Spectral",
    norm=condnorm,
)
fig.colorbar(
    sm,
    ax=ax["time"],
    label=r"$\sigma$ (S/m)",
    orientation="horizontal",
    location="top",
)

label_map = {
    "a": r"$E_0$ (MV/cm)",
    "f": r"$f_0$ (THz)",
    "td": r"$\tau_\mathrm{d}$ (ps)",
    "p": r"$\phi$ $(\pi)$",
}
for (variable,), group in best_params.group_by(["variable"]):
    x, y, size = (
        group.filter(pl.col("resonance_id") == "1")
        .select("cond", "value", "amplitude")
        .get_columns()
    )
    size *= 3
    ax[variable].scatter(x, y, s=size, ec="white", lw=0.6)

    x, y, size = (
        group.filter(pl.col("resonance_id") == "2")
        .select("cond", "value", "amplitude")
        .get_columns()
    )
    ax[variable].scatter(x, y, s=size, ec="white", lw=0.6)
    ax[variable].set(xscale="log", ylabel=label_map[variable])
    ax[variable].yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

ax["f"].sharex(ax["p"])
ax["td"].sharex(ax["p"])
ax["p"].set(xlabel=r"$\sigma$ (S/m)")
ax["f"].set(ylim=(0.5, 1.6))

mpl_tools.breathe_axes(ax.values(), "y")
mpl_tools.breathe_axes(iax)

mpl_tools.enumerate_axes(ax.values())

fig.savefig(PROJECT_PATHS.figures / "simulations/gapfield_temporal_summary.pdf")

plt.show()
