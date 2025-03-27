import logging

import lmfit as lm
import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset
from scipy.interpolate import PchipInterpolator

from driven_nickelate.conductivity_mapping.calculation import (
    DATAFILES as CALCULATION_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.linear_spectroscopy.esrr import (
    DATAFILES as MEASUREMENT_DATAFILES,
)

STORE_DIR = PROJECT_PATHS.processed_data / "conductivity_mapping/temperature"
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
        id_vars=["cond", "cond0", "temperature", "direction"],
    ),
    "error_cond0": Datafile(
        path=STORE_DIR / "error_cond0.csv",
    ),
}


def relativize_meas(spectra_meas):
    spectra_ref = (
        spectra_meas.filter(
            col("temperature").eq(col("temperature").min().over("direction"))
        )
        .group_by("freq")
        .agg(
            col("temperature").mean(),
            ccol("t[c]").real.mean().alias("t_ref.real"),
            ccol("t[c]").imag.mean().alias("t_ref.imag"),
        )
        .with_columns(
            ccol("t_ref.real", "t_ref.imag").alias("t_ref[c]", fields="t_ref")
        )
        .drop("temperature")
        .sort("freq")
    )
    trel_meas = spectra_meas.join(spectra_ref, on="freq").select_data(
        ((ccol("t[c]") - ccol("t_ref[c]")) / ccol("t_ref[c]")).alias(
            "t.reldiff[c]", fields="t"
        )
    )
    return trel_meas


def equalize_conductivities(spectra_calc):
    cond = pl.Series("cond", np.r_[np.geomspace(1e3, 1e6, 500)]).unique().sort()

    spectra_calc = spectra_calc.regrid(
        cond.rename("cond_gap"), method="catmullrom"
    ).drop_nans()
    spectra_calc = spectra_calc.regrid(
        cond.rename("cond_film"), method="catmullrom"
    ).drop_nans()
    spectra_calc = (
        spectra_calc.filter(col("cond_gap").eq(col("cond_film")))
        .rename({"cond_gap": "cond"})
        .drop("cond_film")
        .select_data(r"^t\..*$")
        .sort("cond", "freq")
    )
    spectra_calc = spectra_calc.drop_nulls()
    return spectra_calc


def relativize_calc(spectra_calc):
    frames = []
    spectra_calc = spectra_calc.select_data(
        ccol("t.real", "t.imag").alias("t[c]", fields="t")
    )
    for cond in spectra_calc.coord("cond"):
        spectra_ref = spectra_calc.filter(col("cond") == cond)
        spectra_ref = spectra_ref.rename({"cond": "cond0", "t[c]": "t0[c]"})
        frames.append(
            spectra_calc.join(spectra_ref, on="freq").select_data(
                ((ccol("t[c]") - ccol("t0[c]")) / ccol("t0[c]")).alias(
                    "t.reldiff[c]", fields="t"
                )
            )
        )
    trel_calc = Dataset(frames, index="freq", id_vars=["cond", "cond0"]).filter(
        col("cond") >= col("cond0")
    )
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
    frequencies = ds_calc.coord("freq")
    ds_meas = ds_meas.sort("freq").regrid(frequencies, method="catmullrom")
    ds_calc = ds_calc.sort("freq")
    return ds_meas, ds_calc


def join_datasets(df_meas, df_calc) -> Dataset:
    df_joined = (
        df_calc.select_data(
            col("t.reldiff[c]").alias("t_calc.reldiff[c]", fields="t_calc.reldiff"),
        )
        .sort("freq")
        .join(
            df_meas.select_data(
                col("t.reldiff[c]").alias("t_meas.reldiff[c]", fields="t_meas.reldiff"),
            ).sort("freq"),
            on="freq",
        )
    )
    return df_joined.select_data(r"^t_.*$")


def compute_error(trel_joint):
    freq_weights = (
        (1 - (col("freq") - 1.5).tanh()) / 2  # Tapers off at higher frequencies
        + (-0.5 * ((col("freq") - 1.0) / 0.3) ** 2).exp()  # Extra attention to the peak
    ) / 2

    error_expr = freq_weights * (
        ccol("t_meas.reldiff[c]") - ccol("t_calc.reldiff[c]").conj()
    ).modulus().pow(2)
    error = (
        trel_joint.select_data(
            error_expr.alias("error[c]", fields="error"),
        )
        .group_by(pl.all().exclude("freq", "error[c]", r"^t\..*$"))
        .agg(ccol("error[c]").mean().alias("error"))
        .sort("cond", "temperature", "direction")
    )
    return error


def minimize_cond0(error):
    error_cond0 = (
        error.group_by(pl.all().exclude("cond", "error"))
        .agg(col("cond", "error").sort_by(col("error")).first())
        .group_by("cond0")
        .agg(col("error").pow(2).mean().sqrt())
    )

    error_diff = (
        error_cond0.filter(col("cond0").is_between(1e2, 5e4))
        .sort("cond0")
        .select(col("cond0"), col("error").diff().alias("error_diff"))
        .drop_nulls()
    )
    roots = PchipInterpolator(*error_diff.get_columns(), extrapolate=False).roots()
    if len(roots) != 1:
        raise ValueError(f"Could not find a unique root. The roots are {roots}")

    return error_cond0.with_columns(pl.lit(roots[0]).alias("cond0_best"))


def cond_from_temperatures(temperatures: list[float] | float):
    if isinstance(temperatures, float):
        temperatures = [temperatures]

    mapping = (
        DATAFILES["solution"]
        .load()
        .select("temperature", "direction", "cond")
        .unique()
        .sort("temperature", "direction")
    )
    temperature_mapping = (
        mapping.with_columns(
            col("temperature").cum_count().over("direction").alias("index"),
        )
        .select(
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

    logging.info("Relativizing measurement")
    trel_meas = relativize_meas(spectra_meas)

    logging.info("Loading calculation")
    spectra_calc = CALCULATION_DATAFILES["spectra"].load()

    logging.info("Upsampling and equalizing conductivity")
    spectra_calc = equalize_conductivities(spectra_calc)

    logging.info("Relativizing calculation")
    trel_calc = relativize_calc(spectra_calc)

    logging.info("Aligning frequencies")
    trel_meas, trel_calc = align_frequencies(trel_meas, trel_calc)
    DATAFILES["trel_meas"].write(trel_meas.unnest(cs.contains("[c]")))
    DATAFILES["trel_calc"].write(trel_calc.unnest(cs.contains("[c]")))

    logging.info("Joining datasets, with all candidate `cond0`")
    trel_joint = join_datasets(trel_meas, trel_calc)

    logging.info("Computing error over all parameters")
    error = compute_error(trel_joint)

    logging.info("Computing optimal `cond0`")
    error_cond0 = minimize_cond0(error)
    root = error_cond0["cond0_best"][0]
    DATAFILES["error_cond0"].write(error_cond0)

    logging.info("Re-joining datasets with optimal `cond0`")
    trel_calc = (
        trel_calc.filter(col("cond0").is_between(root * (1 - 0.2), root * (1 + 0.2)))
        .regrid(pl.Series("cond0", [root]), method="linear")
        .drop_nulls()
    )

    cond = pl.Series("cond", np.geomspace(1e2, 1e6, 1000))
    trel_calc = trel_calc.regrid(cond, method="catmullrom").drop_nulls()

    trel_joint = join_datasets(trel_meas, trel_calc)

    logging.info("Computing error over `cond`")
    error = compute_error(trel_joint)
    DATAFILES["error"].write(error)

    mapping = error.group_by("temperature", "direction").agg(
        col("cond", "error").sort_by("error").first()
    )

    solution = trel_joint.join(
        mapping, on=("temperature", "direction", "cond")
    ).with_columns(
        (ccol("t_meas.reldiff[c]") - ccol("t_calc.reldiff[c]")).alias(
            "residual[c]", fields="residual"
        ),
    )
    DATAFILES["solution"].write(solution.unnest(cs.contains("[c]")))

    logging.info("Finished.")
