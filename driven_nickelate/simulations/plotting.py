import ezdxf as ez
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.properties import LayoutProperties
from matplotlib.colors import LogNorm


def plot_dxf(
    dxf_path: str, ax: plt.Axes, layers: list[str] = None, color="#000000"
) -> None:
    """
    Plot a dxf file on a matplotlib axis. If no axis is provided, a new figure is created.

    Parameters
    ----------
    dxf_path : str or Path
        Path to the dxf file.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created.
    extent : tuple, optional
        Extent of the plot (left, right, bottom, top). If None, the extent of the dxf file is used.
    kwargs : dict, optional
        Additional keyword arguments passed to ax.pcolormesh.
    """
    if isinstance(ax, (list, tuple)):
        for _ax in ax:
            plot_dxf(dxf_path, _ax)
        return None

    sdoc = ez.readfile(dxf_path)

    for layer in sdoc.layers:
        layer.color = ez.colors.BLACK

    msp = sdoc.modelspace()
    ctx = RenderContext(sdoc)
    msp_properties = LayoutProperties.from_layout(msp)
    msp_properties.set_colors("#eaeaeaff")

    if layers:
        for layer_name, layer in ctx.layers.items():
            layer.color = color
            if layer_name not in layers:
                layer.is_visible = False

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xvisible = ax.xaxis.get_visible()
    yvisible = ax.yaxis.get_visible()
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    spines_visible = list(map(lambda s: s.get_visible(), ax.spines.values()))
    aspect = ax.get_aspect()

    out = MatplotlibBackend(ax, adjust_figure=False)
    Frontend(ctx, out).draw_layout(
        msp,
        finalize=True,
        layout_properties=msp_properties,
    )

    ax.set(xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim)
    ax.xaxis.set_visible(xvisible)
    ax.yaxis.set_visible(yvisible)
    for spine, is_visible in zip(ax.spines.values(), spines_visible):
        spine.set_visible(is_visible)
    ax.set_aspect(aspect)


def plot_sij(smatrix_data):
    s21 = smatrix_data.filter(pl.col("param").eq("s21.mag"))
    s11 = smatrix_data.filter(pl.col("param").eq("s11.mag"))

    fig = plt.figure(
        figsize=(3.4, 5.1),
        layout="constrained",
    )
    ax = fig.subplot_mosaic([["s11", "cbar"], ["s21", "cbar"]], width_ratios=(1, 0.05))

    norm = LogNorm(vmin=1, vmax=smatrix_data["cond"].max())
    plot_kwargs = dict(
        x="freq",
        y="value",
        hue="cond",
        hue_norm=norm,
        palette="flare",
        legend=False,
    )
    sns.lineplot(data=s11, ax=ax["s11"], **plot_kwargs)
    sns.lineplot(data=s21, ax=ax["s21"], **plot_kwargs)

    ax["s11"].set(
        xlabel=r"$f$ (THz)", ylabel=r"$|s_{11}|$", xlim=(0.5, 2.5), ylim=(0, 1)
    )
    ax["s21"].set(
        xlabel=r"$f$ (THz)", ylabel=r"$|s_{21}|$", xlim=(0.5, 2.5), ylim=(0, 1)
    )

    sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
    fig.colorbar(sm, cax=ax["cbar"], label="Conductivity (S/m)")
    ax["cbar"].set(yscale="log")

    return fig, ax


def plot_s21(smatrix_data):
    s21 = smatrix_data.filter(pl.col("param").eq("s21.mag"))

    fig = plt.figure(
        figsize=(3.4, 2.6),
        layout="constrained",
    )
    ax = fig.subplot_mosaic([["s21", "cbar"]], width_ratios=(1, 0.05))

    norm = LogNorm(vmin=1, vmax=smatrix_data["cond"].max())
    plot_kwargs = dict(
        x="freq",
        y="value",
        hue="cond",
        hue_norm=norm,
        palette="flare",
        legend=False,
    )
    sns.lineplot(data=s21, ax=ax["s21"], **plot_kwargs)
    ax["s21"].set(
        xlabel=r"$f$ (THz)", ylabel=r"$|s_{21}|$", xlim=(0.5, 2.5), ylim=(0, None)
    )
    sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
    fig.colorbar(sm, cax=ax["cbar"], label="Conductivity (S/m)")
    ax["cbar"].set(yscale="log")

    return fig, ax


def plot_heatmap(df: pl.DataFrame, norm=None, **kwargs):
    x = df.columns[0]
    y = df.columns[1]
    z = df.columns[2]

    x_arr = df.get_column(x).unique(maintain_order=True).to_numpy()
    y_arr = df.get_column(y).unique(maintain_order=True).to_numpy()

    z_arr = (
        df.pivot(index=y, columns=x, values=z, aggregate_function=None)
        .drop(y)
        .to_numpy()
    )

    fig = plt.figure(figsize=(3.4, 3.4))
    cmap = kwargs.pop("cmap", "flare")
    ax = fig.subplot_mosaic(
        [["value", "colorbar"]],
        width_ratios=(1, 0.05),
    )
    ax["value"].pcolormesh(x_arr, y_arr, z_arr, shading="gouraud", cmap=cmap, **kwargs)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=(norm or plt.Normalize(np.min(z_arr), np.max(z_arr)))
    )
    fig.colorbar(sm, cax=ax["colorbar"])

    return fig, ax
