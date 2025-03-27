import re

import lmfit as lm
import matplotlib.pyplot as plt
import pandas as pd


def read_data(filename):
    df = pd.read_csv(root + filename, usecols=(0, 1))
    unit = re.search(r"\((.*?)\)", df.columns[0]).group(1)
    df.columns = ["z", "count"]
    if unit == "nm":
        df["z"] *= 1e-3
    return df


def get_params(df):
    gaussian = lm.models.GaussianModel()
    pars = gaussian.guess(df["count"], x=df["z"])
    out = gaussian.fit(df["count"], pars, x=df["z"])
    return out


root = "data/"
filenames = {
    "front": {
        "sample": "frontsurface_F21059_50X_0.55X_histogram.csv",
        "substrate": "frontsurface_50X_0.55X_histogram.csv",
    },
    "back": {
        "sample": "backsurface_F21059_50X_0.55X_histogram.csv",
        "substrate": "backsurface_50X_0.55X_histogram.csv",
    },
}

if __name__ == "__main__":
    fig = plt.figure(figsize=(6.8, 6.8), layout="constrained")
    gs = fig.add_gridspec(2, 2, hspace=0.1)
    ax_front = [fig.add_subplot(gs[0, i]) for i in range(2)]
    ax_back = [fig.add_subplot(gs[1, i]) for i in range(2)]

    for i, filename in enumerate(filenames["front"].values()):
        df = read_data(filename)
        out = get_params(df)
        params = out.params

        ax_front[i].bar(1e3 * (df["z"] - params["center"]), df["count"])
        ax_front[i].plot(1e3 * (df["z"] - params["center"]), out.best_fit, color="k")
        ax_front[i].set(
            xlim=(-30, 30),
            xlabel="Height (nm)",
            ylabel="Count",
            ylim=(0, df["count"].max() * 1.2),
            title=f"front, {list(filenames['back'].keys())[i]}",
        )
        sigma = params["sigma"]
        ax_front[i].annotate(
            f"sigma = {1e3 * sigma.value:.2f} ± {1e3 * sigma.stderr:.2f} nm",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
        )

    for i, filename in enumerate(filenames["back"].values()):
        df = read_data(filename)
        out = get_params(df)
        params = out.params

        ax_back[i].bar((df["z"] - params["center"]), df["count"], width=0.1)
        ax_back[i].plot((df["z"] - params["center"]), out.best_fit, color="k")
        ax_back[i].set(
            xlim=(-3, 3),
            xlabel="Height (um)",
            ylabel="Count",
            ylim=(0, df["count"].max() * 1.2),
            title=f"back, {list(filenames['back'].keys())[i]}",
        )
        sigma = params["sigma"]
        ax_back[i].annotate(
            f"sigma = {sigma.value:.2f} ± {sigma.stderr:.2f} um",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
        )
    fig.savefig("roughness.png", dpi=300)
