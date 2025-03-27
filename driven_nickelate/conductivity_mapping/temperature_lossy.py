import logging

import lmfit as lm
import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.linear_spectroscopy.esrr import (
    DATAFILES as MEASUREMENT_DATAFILES,
)

STORE_DIR = PROJECT_PATHS.processed_data / "conductivity_mapping/temperature_lossy"
STORE_DIR.mkdir(exist_ok=True, parents=True)

DATAFILES = {
    "trel_meas": Datafile(
        path=STORE_DIR / "trel_meas.csv",
        index="freq",
        id_vars=["temperature", "direction"],
    ),
    "trel_calc": Datafile(
        path=STORE_DIR / "trel_calc.csv",
        index="freq",
        id_vars=["cond", "cond0"],
    ),
    "error": Datafile(
        path=STORE_DIR / "error.csv",
        index="cond",
        id_vars=["cond0", "temperature", "direction"],
    ),
    "solution": Datafile(
        path=STORE_DIR / "solution.csv",
        index="freq",
        id_vars=["cond", "direction", "temperature"],
    ),
    "error_cond0": Datafile(
        path=STORE_DIR / "error_cond0.csv",
    ),
}


def load_mapping():
    return Dataset(
        (
            DATAFILES["solution"]
            .load()
            .fetch("temperature", "direction", "cond")
            .sort("direction", "temperature")
            .unique()
        ),
        index="temperature",
        id_vars=["direction"],
    )


def relativize_meas(spectra_meas):
    spectra_ref = (
        spectra_meas.filter(
            col("temperature").eq(col("temperature").min().over("direction"))
        )
        .group_by("freq")
        .agg(
            col("temperature").mean(),
            ccol("t[c]").real().mean().alias("t_ref.real"),
            ccol("t[c]").imag().mean().alias("t_ref.imag"),
        )
        .complex.nest("t_ref")
        .drop("temperature")
        .sort("freq")
    )
    trel_meas = spectra_meas.join(spectra_ref, on="freq").select_data(
        ((ccol("t[c]") - ccol("t_ref[c]")) / ccol("t_ref[c]")).alias(
            "t.reldiff[c]", fields="t.reldiff"
        )
    )
    return trel_meas


def relativize_calc(spectra_calc):
    frames = []
    for param in tqdm(spectra_calc.coord("param")):
        cond0, tau0 = param["cond"], param["tau"]
        spectra0 = (
            spectra_calc.filter(
                col("param").struct["cond"].eq(cond0)
                & col("param").struct["tau"].eq(tau0)
            )
            .select(
                col("freq"),
                col("param").struct.rename_fields(["cond0", "tau0"]).alias("param0"),
                col("t[c]").alias("t0[c]", fields="t0"),
            )
            .set(id_vars=["param0"])
        )
        spectra = spectra_calc.filter(
            (col("param").struct["tau"] == tau0)
            & (col("param").struct["cond"] >= cond0)
        )
        frames.append(
            spectra.join(spectra0, on="freq").select_data(
                ((ccol("t[c]") - ccol("t0[c]")) / ccol("t0[c]")).alias(
                    "t.reldiff[c]", fields="t.reldiff"
                )
            )
        )
    trel_calc = Dataset(frames, index="freq")
    return trel_calc


def freq_samples():
    start, step, stop = 0.2, 0.08, 2.2
    freq_list = [start]
    a_lc, b_lc = 0.08, 0.65
    a_dp, b_dp = 0.08, 0.45
    while freq_list[-1] <= stop:
        adapt_lc = b_lc * a_lc / ((freq_list[-1] - 1.0) ** 2 + a_lc)
        adapt_dp = b_dp * a_dp / ((freq_list[-1] - 2.0) ** 2 + a_dp)
        freq_list.append(round(freq_list[-1] + step * (1 - adapt_lc - adapt_dp), 3))
    return pl.Series("freq", freq_list)


def align_frequencies(ds_meas: Dataset, ds_calc: Dataset) -> tuple[Dataset, Dataset]:
    frequencies = freq_samples()
    ds_meas = ds_meas.sort("freq").regrid(frequencies, method="cosine")
    ds_calc = ds_calc.sort("freq").regrid(frequencies, method="cosine")
    return ds_meas, ds_calc


def minimize_cond0(error):
    error0 = (
        error.group_by(pl.all().exclude("param", "error"))
        .agg(col("param", "error").sort_by(col("error")).first())
        .group_by("param0")
        .agg(col("error").pow(2).mean().sqrt())
    )

    error_diff = (
        error0.sort("param0")
        .select(col("param0"), col("error").diff().alias("error_diff"))
        .drop_nulls()
    )
    roots = PchipInterpolator(*error_diff.get_columns(), extrapolate=False).roots()
    if len(roots) != 1:
        raise ValueError(f"Could not find a unique root. The roots are {roots}")

    return error0.with_columns(pl.lit(roots[0]).alias("cond0_best"))


def cond_from_temperatures(temperatures: list[float] | float):
    if isinstance(temperatures, float):
        temperatures = [temperatures]

    temperature_mapping = (
        load_mapping()
        .sort("direction", "temperature")
        .with_columns(
            col("temperature").cum_count().over("direction").alias("index"),
        )
        .fetch(
            col("temperature", "cond").mean().over("index"),
        )
        .unique("temperature")
        .sort("temperature")
    )

    cond_pchip = PchipInterpolator(
        *temperature_mapping.select("temperature", "cond").get_columns(),
        extrapolate=False,
    )

    model = lm.models.PolynomialModel(2)
    params = model.make_params(c0=1e3, c1=1e2, c2=0)
    max_temp = 100
    fit_data = temperature_mapping.with_columns(
        col("cond").log().alias("log-cond")
    ).filter((col("temperature") <= max_temp))
    result = model.fit(
        data=fit_data.get_column("log-cond"),
        x=fit_data.get_column("temperature"),
        params=params,
    )

    cond = cond_pchip(temperatures)
    for i in range(len(cond)):
        if np.isnan(cond[i]):
            cond[i] = np.exp(result.eval(x=temperatures[i]))

    if isinstance(temperatures, float):
        return cond[0]
    return cond


def load_calc(path):
    spectra_calc = pl.read_csv(path).select(
        col("cond", "tau", "freq"),
        ccol("t.real", "t.imag").alias("t[c]", fields="t"),
    )
    spectra_calc = Dataset(spectra_calc, index="freq", id_vars=["cond", "tau"]).sort()
    return spectra_calc


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Loading measurement")
    spectra_meas = (
        MEASUREMENT_DATAFILES["spectra"]
        .load()
        .select_data(ccol("t.real", "t.imag").alias("t[c]", fields="t"))
    )

    logging.info("Loading calculation")
    spectra_calc = load_calc(
        PROJECT_PATHS.root / "simulations/data/02.04.24/spectral_data.csv"
    )
    spectra_calc = Dataset(
        spectra_calc,
        index="freq",
        id_vars=["cond", "tau"],
    )
    tau = pl.Series("tau", np.linspace(0, 0.1, 40))
    cond = pl.Series("cond", np.geomspace(1e3, 5e5, 100))
    spectra_calc = spectra_calc.regrid(tau, method="catmullrom").drop_nulls()
    spectra_calc = spectra_calc.regrid(cond, method="catmullrom").drop_nulls()

    spectra_calc = Dataset(
        spectra_calc.select(
            pl.struct("cond", "tau").alias("param"), col("freq"), col("t[c]")
        ),
        spectra_calc.index,
        id_vars=["param"],
    )

    logging.info("Aligning frequencies")
    spectra_meas, spectra_calc = align_frequencies(spectra_meas, spectra_calc)

    logging.info("Relativizing calculation")
    trel_calc = (
        relativize_calc(spectra_calc)
        .select_data(
            col("t.reldiff[c]").alias("t_calc.reldiff[c]", fields="t_calc.reldiff"),
        )
        .sort("freq")
    )

    logging.info("Relativizing measurement")
    trel_meas = (
        relativize_meas(spectra_meas)
        .select_data(
            col("t.reldiff[c]").alias("t_meas.reldiff[c]", fields="t_meas.reldiff"),
        )
        .sort("freq")
    )
    DATAFILES["trel_meas"].write(trel_meas.unnest(cs.contains("[c]")))

    logging.info("Computing errors for all parameters")
    errors = []
    with tqdm(total=trel_meas.n_unique(["temperature", "direction"])) as pbar:
        for (direction, temperature), group in tqdm(
            trel_meas.group_by(["direction", "temperature"])
        ):
            error = (
                trel_calc.join(group, on="freq")
                .select_data(
                    (ccol("t_calc.reldiff[c]") - ccol("t_meas.reldiff[c]"))
                    .modulus()
                    .pow(2)
                    .alias("error")
                )
                .group_by("param", "param0")
                .agg(
                    pl.lit(direction).alias("direction"),
                    pl.lit(temperature).alias("temperature"),
                    col("error").mean(),
                )
            )
            errors.append(error)
            pbar.update(1)
    errors = pl.concat(errors)
    DATAFILES["error"].write(errors.unnest("param", "param0"))

    logging.info("Computing optimal `cond0`")
    error0 = (
        errors.group_by("param0", "direction", "temperature")
        .agg(col("param", "error").sort_by(col("error")).first())
        .group_by("param0")
        .agg(col("error").pow(2).mean().sqrt())
    )
    DATAFILES["error_cond0"].write(error0.unnest("param0"))

    logging.info("Filtering down to optimal `cond0`")
    cond0, tau0 = error0.select(col("param0").sort_by("error").first()).unnest("param0")
    optimum_condition = col("param0").struct["cond0"].eq(cond0) & (
        col("param0").struct["tau0"].eq(tau0)
    )

    logging.info("Upsampling error along `cond` on optimal `(cond0, tau0)`")
    cond = pl.Series("cond", np.geomspace(1e2, 1e6, 500))
    trel_calc = (
        trel_calc.unnest("param")
        .set(id_vars=["cond", "tau", "param0"])
        .filter(optimum_condition & (col("tau") == tau0))
        .drop(["param0", "tau"])
        .regrid(cond, method="cosine")
        .drop_nulls()
        .sort("freq", "cond")
    )
    DATAFILES["trel_calc"].write(trel_calc.unnest(cs.contains("[c]")))

    logging.info("Filtering down calculation to optimum_condition `(cond0, tau0)`")
    trel_joint = trel_calc.join(trel_meas, on="freq")

    logging.info("Computing errors at optimum_condition `(cond0, tau0)`")
    error_expr = (
        (ccol("t_meas.reldiff[c]") - ccol("t_calc.reldiff[c]")).modulus().pow(2)
    )
    errors = (
        trel_joint.with_columns(error_expr.alias("error"))
        .group_by("cond", "direction", "temperature")
        .agg(col("error").sum().alias("error"))
    )

    logging.info("Computing optimal `cond`")
    optima = errors.group_by("direction", "temperature").agg(
        col("cond", "error").sort_by("error").first()
    )

    solution = (
        trel_joint.join(
            optima.select("direction", "temperature", "cond"),
            on=["cond", "temperature", "direction"],
            how="inner",
        )
        .with_columns(
            (ccol("t_meas.reldiff[c]") - ccol("t_calc.reldiff[c]")).alias(
                "residual[c]", fields="residual"
            )
        )
        .unnest(cs.contains("[c]"))
    )
    DATAFILES["solution"].write(solution)

    logging.info("Finished.")
