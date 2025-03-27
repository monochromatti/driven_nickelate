import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import rcParams
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from mplstylize import colors
from polars import col
from scipy.constants import c as c_const
from scipy.special import erf

from driven_nickelate.config import paths as PROJECT_PATHS


def decay_expr(t, t0, total_amp, amp1, amp2, sig0, tau1, tau2, offset):
    t = t - t0
    rise = total_amp * (1 + erf(t * sig0)) * 0.5
    return rise * (amp1 * np.exp(-t / tau1) + amp2 * np.exp(-t / tau2) + offset)


def annotate_timescales(ax_left, ax_right):
    double_bracket = ArrowStyle.BarAB(widthA=2, widthB=2)
    left_bracket = ArrowStyle.BarAB(widthA=2, widthB=0)
    right_bracket = ArrowStyle.BarAB(widthA=0, widthB=2)
    right_arrow = ArrowStyle.CurveFilledB(head_width=2, head_length=2)

    t0, tau1, tau2 = (
        result.params["t0"],
        result.params["tau1"],
        result.params["tau2"],
    )

    y0 = 3e-4
    ax_left.add_patch(
        FancyArrowPatch((t0, y0), (t0 + tau1, y0), arrowstyle=double_bracket)
    )
    ax_left.text(t0 + tau1, y0, r"$\tau_1$", ha="left", va="center")

    y0 -= 1e-4
    ax_left.add_patch(FancyArrowPatch((t0, y0), (5, y0), arrowstyle=left_bracket))
    ax_right.add_patch(
        FancyArrowPatch((t0, y0), (t0 + tau2, y0), arrowstyle=right_bracket)
    )
    ax_right.text(t0 + tau2, y0, r"$\tau_2$", ha="left", va="center")

    y0 -= 1e-4
    ax_right.add_patch(FancyArrowPatch((t0, y0), (5, y0), arrowstyle=left_bracket))
    ax_right.add_patch(FancyArrowPatch((t0, y0), (600, y0), arrowstyle=right_arrow))
    ax_right.text(600, 1e-4, r"$\tau_3 = \infty$", ha="left", va="center")


def create_breaklines(ax_left, ax_right):
    d = 0.02
    kwargs = dict(
        transform=ax_left.transAxes,
        color="k",
        clip_on=False,
        lw=rcParams["axes.linewidth"],
    )
    theta = np.deg2rad(45)
    dx, dy = d * np.array([np.cos(theta), np.sin(theta)])

    ax_left.plot([1.01 - dx, 1.01 + dx], [0 - dy, 0 + dy], **kwargs)
    ax_left.plot([1.01 - dx, 1.01 + dx], [1 - dy, 1 + dy], **kwargs)

    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot([0 - dx, 0 + dx], [0 - dy, 0 + dy], **kwargs)
    ax_right.plot([0 - dx, 0 + dx], [1 - dy, 1 + dy], **kwargs)


if __name__ == "__main__":
    model = lm.Model(decay_expr)
    params = model.make_params(
        t0=0, total_amp=1e-3, amp1=1, amp2=0.5, sig0=4, tau1=3, tau2=500, offset=0
    )
    params["sig0"].set(min=1e-2)
    params["tau1"].set(min=1e-2)
    params.add("tau0", expr="0.238/sig0")

    fname = (
        PROJECT_PATHS.raw_data
        / "05.03.23/13h39m17s_Pump chopped pump-probe scan nonlinear steps_3.01-4.37_Sam=9.14_HWP=30.00.txt"
    )
    data = (
        pl.read_csv(fname, separator="\t", comment_prefix="#", has_header=False)
        .select(
            ((2e9 / c_const) * (147 - col("column_1"))).alias("time"),
            col("column_2").alias("signal"),
        )
        .with_columns(col("signal") - col("signal").filter(col("time") < -0.2).mean())
    )

    result = model.fit(data["signal"], params, t=data["time"])

    fig = plt.figure(figsize=(6.8, 3.4), layout="none")
    axs = fig.subplot_mosaic([["short", "long"]])

    big_ax = plt.Subplot(fig, axs["short"].get_gridspec()[-1, :], frameon=False)
    big_ax.set(xticks=[], yticks=[])
    big_ax.patch.set_facecolor("none")
    big_ax.set_xlabel(r"$\tau\,({\rm ps})$", labelpad=20)

    fig.add_subplot(big_ax)

    time_stop = 5
    time_start = 20
    axs["short"].scatter(
        *data.filter(col("time") < time_stop), ec="white", c=colors("bice"), s=20
    )
    axs["long"].scatter(
        *data.filter(col("time") > time_start), ec="white", c=colors("bice"), s=20
    )

    time_short = np.linspace(-3, time_stop, 1000)
    time_long = np.linspace(time_start, 700, 1000)
    axs["short"].plot(time_short, result.eval(t=time_short), color=colors("deep_gray"))
    axs["long"].plot(time_long, result.eval(t=time_long), color=colors("deep_gray"))

    axs["inset"] = axs["long"].inset_axes([0.4, 0.5, 0.5, 0.4])
    axs["inset"].plot(
        data["time"].log10(),
        data["signal"],
        color=colors("deep_gray"),
        mew=0.6,
        ms=2,
    )
    axs["inset"].set(xlabel=r"$\log_{10}(\tau)$", ylabel=r"$\Delta E$ (a.u.)")

    create_breaklines(axs["short"], axs["long"])
    annotate_timescales(axs["short"], axs["long"])

    axs["long"].sharey(axs["short"])
    axs["long"].spines["left"].set_visible(False)
    axs["long"].yaxis.set_tick_params(labelleft=False, left=False)

    axs["short"].spines["right"].set_visible(False)
    axs["short"].yaxis.set_tick_params(labelright=False, right=False)
    axs["short"].set(ylabel=r"$\Delta E(t_0)$ (a.u.)")

    axs["long"].set(xlim=(time_start, None))
    axs["short"].set(xlim=(None, time_stop), ylim=(-1e-4, 1.5e-3))

    fig.tight_layout(pad=0.5)
    fig.savefig(PROJECT_PATHS.figures / "photodoping/decay_timescales.pdf")
