import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


def get_data(path):
    return pl.scan_csv(
        path,
        skip_rows=5,
        separator="\t",
        new_columns=["time", "temperature", "R1", "theta1", "X1", "R2", "theta2", "X2"],
    )


def plot_rxy(data):
    fig, ax = plt.subplots(figsize=(3.4, 3.4), layout="constrained")

    sns.lineplot(
        data=data.sample(fraction=0.05),
        x="temperature",
        y="R2",
        hue="direction",
        ax=ax,
    )
    ax.set(yscale="log", xlabel="Temperature (K)", ylabel=r"$R_{xy}$ ($\Omega$)")
    plt.show()


if __name__ == "__main__":
    data_warming = get_data(
        r"raw_data/Config 1/0_1_2002_F21062_RT_I_5_12_V1_5_12_V2_6-11_gains_1-100_1uA_cooldown.txt"
    ).with_columns(pl.lit(-1).alias("direction"))

    data_cooling = get_data(
        r"raw_data/Config 1/0_2_0754_F21062_RT_I_5_12_V1_5_12_V2_6-11_gains_1-100_1uA_warmup.txt"
    ).with_columns(pl.lit(+1).alias("direction"))

    data = (
        pl.concat([data_warming, data_cooling])
        .filter(pl.col("temperature") < 280)
        .filter(pl.col("R2") < 10921.7)
        .sort("direction", "temperature")
        .select("temperature", "direction", "R2")
        .rename({"R2": "R"})
        .collect()
    )

    fit_data = data.filter(pl.col("temperature") < 50).with_columns(
        pl.col("R").log().alias("logR")
    )

    model = lm.models.PolynomialModel(degree=3)
    params = model.make_params(c0=8.2, c1=0.03, c2=0.0, c3=0.0)

    temperature_fit = np.linspace(0, fit_data["temperature"].min(), 100)

    results = []
    for (direction,), data_subset in fit_data.group_by(["direction"]):
        temperatures = np.linspace(0, data_subset["temperature"].min(), 100)
        result = model.fit(data_subset["logR"], params, x=data_subset["temperature"])

        results.append(
            pl.DataFrame(
                {
                    "logR": result.eval(x=temperatures),
                }
            ).with_columns(
                pl.lit(temperatures).alias("temperature"),
                pl.lit(direction).alias("direction"),
                pl.col("logR").exp().alias("R"),
            )
        )

    data_extrapolated = pl.concat(results).select("temperature", "direction", "R")
    data = pl.concat(
        (
            data.with_columns(pl.lit("raw").alias("data_type")),
            data_extrapolated.with_columns(pl.lit("fit").alias("data_type")),
        )
    )

    vdP_correction = np.pi / np.log(2)
    film_thickness = 11e-9
    data = data.with_columns(
        (pl.col("R") * vdP_correction * film_thickness).alias("rho")
    ).with_columns((1 / pl.col("rho")).alias("sigma"))

    fig, ax = plt.subplots(figsize=(3.4, 3.4), layout="constrained")

    sns.lineplot(
        data=data.sample(fraction=0.1).filter(pl.col("data_type") == "raw"),
        x="temperature",
        y="sigma",
        hue="direction",
        ax=ax,
    )
    sns.lineplot(
        data=data.filter(pl.col("data_type") == "fit"),
        x="temperature",
        y="sigma",
        hue="direction",
        ax=ax,
        style=True,
        dashes=[(2, 2), (2, 2)],
        lw=0.8,
        legend=False,
    )
    ax.set(yscale="log", xlabel="$T$ (K)", ylabel=r"$\sigma$ (S/m)")
    fig.savefig("sigma.png")

    data.write_csv("data.csv")
