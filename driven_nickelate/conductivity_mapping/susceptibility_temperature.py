# %%
import logging

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.polynomial import Polynomial
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

import driven_nickelate.conductivity_mapping.calculation as calc
from driven_nickelate.conductivity_mapping.temperature import cond_from_temperatures
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.scripts.pump_probe.field_from_hwp import field_from_hwp
from driven_nickelate.tools import create_dataset, pl_fft

STORE_DIR = (
    PROJECT_PATHS.processed_data / "conductivity_mapping/susceptibility_temperature"
)
STORE_DIR.mkdir(exist_ok=True, parents=True)

DATAFILES = {
    "waveforms_meas": Datafile(
        path=STORE_DIR / "waveforms_meas.csv",
        index="time",
        id_vars=["field_strength", "temperature"],
    ),
    "spectra_meas": Datafile(
        path=STORE_DIR / "spectra_meas.csv",
        index="freq",
        id_vars=["field_strength", "temperature"],
    ),
    "spectra_calc": Datafile(
        path=STORE_DIR / "spectra_calc.csv",
        index="freq",
        id_vars=["cond_gap", "cond_film"],
    ),
    "trel_joint": Datafile(
        path=STORE_DIR / "trel_joint.csv",
        index="freq",
        id_vars=["cond_gap", "field_strength", "temperature", "cond_film"],
    ),
    "solution_free": Datafile(
        path=STORE_DIR / "solution_free.csv",
    ),
    "solution_rigid": Datafile(
        path=STORE_DIR / "solution_rigid.csv",
    ),
    "solution_fixed": Datafile(
        path=STORE_DIR / "solution_fixed.csv",
    ),
}


# %%
def process_waveforms() -> Dataset:
    time = pl.Series("time", np.arange(-10, 50, 3e-3))
    waveforms = (
        create_dataset(
            pl.read_csv(
                PROJECT_PATHS.file_lists / "pump_probe/240223_xHWPxT.csv",
                comment_prefix="#",
            ).select(
                col("HWP")
                .map_elements(field_from_hwp, return_dtype=pl.Float64)
                .alias("field_strength"),
                col("Get temperature").round(0).cast(pl.Int32).alias("temperature"),
                (str(PROJECT_PATHS.root) + "/" + col("Path")).alias("path"),
            ),
            column_names=["delay", "X", "X SEM", "Y", "Y SEM"],
            index="delay",
            lockin_schema={"X": ("X", "Y")},
            id_schema={"field_strength": pl.Float32, "temperature": pl.Int32},
        )
        .with_columns(
            (6.67 * (9.8 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).alias("X"),
        )
        .rename({"delay": "time"})
        .regrid(time, method="catmullrom", fill_value=0)
        .select_data("X")
        .sort(auto=True)
    )
    waveforms = (
        waveforms.join(
            waveforms.filter(
                col("field_strength").eq(
                    col("field_strength").min().over("temperature")
                )
            ),
            on=("time", "temperature"),
            suffix="_eq",
        )
        .with_columns((col("X") - col("X_eq")).alias("dX"))
        .select("field_strength", "temperature", "time", "X", "X_eq", "dX")
    )
    return waveforms


def process_fourier(waveforms: Dataset) -> Dataset:
    spectra_meas = pl_fft(waveforms, "time", waveforms.id_vars)
    spectra_meas = Dataset(spectra_meas, "freq", waveforms.id_vars).select_data(
        ccol("X.real", "X.imag").alias("X[c]", fields="X"),
        ccol("dX.real", "dX.imag").alias("dX[c]", fields="dX"),
        ccol("X_eq.real", "X_eq.imag").alias("X_eq[c]", fields="X_eq"),
    )
    spectra_meas = spectra_meas.sort(auto=True).select_data(
        (ccol("dX[c]") / ccol("X_eq[c]")).alias("t.reldiff[c]", fields="t.reldiff")
    )
    return spectra_meas


def upsample_conductivity(df: Dataset) -> Dataset:
    cond_film = np.geomspace(*df.extrema("cond_film"), 200)[:-1]
    cond_gap = np.geomspace(*df.extrema("cond_gap"), 200)[:-1]

    cond_film = pl.Series("cond_film", cond_film)
    cond_gap = pl.Series("cond_gap", cond_gap)

    df = df.regrid(cond_film, method="catmullrom").drop_nulls()
    df = df.regrid(cond_gap, method="catmullrom").drop_nulls()
    return df


def resample_temperature(df_calc, temperatures) -> Dataset:
    temperature_mapping = pl.DataFrame(
        {"temperature": temperatures, "cond_film": cond_from_temperatures(temperatures)}
    )
    cond_film = temperature_mapping["cond_film"]
    cond_gap = df_calc.coord("cond_gap").extend(cond_film).sort()
    df_calc = df_calc.regrid(cond_film, method="catmullrom").drop_nulls()
    df_calc = df_calc.regrid(cond_gap, method="catmullrom").drop_nulls()

    df_calc = df_calc.join(temperature_mapping, on=["cond_film"])
    df_calc.id_vars += ["temperature"]

    return df_calc


def relativize_calc(spectra_calc) -> Dataset:
    cond_gap = (
        spectra_calc.coord("cond_gap")
        .extend(spectra_calc.coord("cond_film"))
        .sort()
        .unique()
    )

    spectra_calc = (
        spectra_calc.regrid(cond_gap, method="catmullrom").sort(auto=True).drop_nulls()
    )

    spectra_reference = (
        spectra_calc.filter(col("cond_gap").eq(col("cond_film")))
        .drop("cond_gap")
        .select_data(ccol("t.real", "t.imag").alias("t_ref[c]", fields="t_ref"))
    )

    spectra_calc = spectra_calc.select_data(
        ccol("t.real", "t.imag").alias("t[c]", fields="t")
    )
    trel_calc = (
        spectra_calc.join(
            spectra_reference,
            on=spectra_reference.id_vars + [spectra_reference.index],
            suffix="_ref",
        )
        .with_columns(
            ((ccol("t[c]") - ccol("t_ref[c]")) / ccol("t_ref[c]")).alias(
                "t.reldiff[c]", fields="t.reldiff"
            )
        )
        .select_data(col("t.reldiff[c]"))
        .sort(auto=True)
    )

    return trel_calc


def align_datasets(ds_meas: Dataset, ds_calc: Dataset) -> tuple[Dataset, Dataset]:
    frequencies = ds_calc.coord("freq")
    ds_meas = ds_meas.sort("freq").regrid(frequencies).drop_nulls()
    ds_calc = ds_calc.sort("freq")
    return ds_meas, ds_calc


def join_datasets(ds_meas, ds_calc, on=["freq"]) -> Dataset:
    ds_meas, ds_calc = ds_meas.sort(), ds_calc.sort()
    ds_meas, ds_calc = align_datasets(ds_meas, ds_calc)

    trel_joint = ds_meas.rename({"t.reldiff[c]": "t_meas.reldiff[c]"}).join(
        ds_calc.rename({"t.reldiff[c]": "t_calc.reldiff[c]"}),
        on=on,
    )
    trel_joint = trel_joint.select_data(col(r"^t_.*$")).sort()
    return trel_joint


def freq_weights():
    freq_weights = 0.5 * (
        1 - np.tanh((col("freq") - 1.8))
    )  # Tapers off at higher frequencies

    freq_weights += np.exp(
        -0.5 * ((col("freq") - 1.0) / 0.5) ** 2
    )  # Extra attention to the peak
    freq_weights /= 2
    return freq_weights


def compute_error(trel_joint):
    """
    Compute error (residual sum of squares) for each {`temperature`, `field_strength`, `cond_film`}.
    """
    taper_weight = (1 - (col("freq") - 2.0).tanh()) / 2
    peak_weight = (-0.5 * ((col("freq") - 1.0) / 0.8) ** 2).exp()
    weights = (taper_weight + peak_weight) / 2
    error_expr = (
        (ccol("t_meas.reldiff[c]") - ccol("t_calc.reldiff[c]")).modulus().pow(2)
    )
    return (
        trel_joint.filter(col("freq").is_between(0.5, 1.5))
        .with_columns(error_expr.alias("error"), weights.alias("weight"))
        .group_by(col(trel_joint.id_vars))
        .agg((col("error") * col("weight")).sum().alias("error"))
    )


def optimize_gap(error):
    """
    Find optimum `cond_gap` for each {`temperature`, `field_strength`, `cond_film`}.
    """
    result_gap = error.group_by("temperature", "field_strength", "cond_film").agg(
        col("error", "cond_gap").sort_by("error").first()
    )

    return result_gap


def optimize_rigid_background(result_gap):
    """
    Find optimum `cond_film` for each `temperature`, assuming no field-dependence for the background.

    `result_gap` is the result of `optimize_gap`. It contains the best `cond_gap` for each
    {`temperature`, `field_strength`, `cond_film`}, and the corresponding error (RSS).
    """
    result_film = (
        result_gap.group_by("temperature", "cond_film")
        .agg(col("error").mean())
        .group_by("temperature")
        .agg(col("error", "cond_film").sort_by("error").first())
    )
    return result_film


def optimize_free_background(result_gap):
    """
    Find optimum `cond_film` for each {`temperature`, `field_strength`}, with polynomial
    approximation into low-field regime where no clear minimum exists.

    `result_gap` is the result of `optimize_gap`. It contains the best `cond_gap` for each
    {`temperature`, `field_strength`, `cond_film`}, and the corresponding error (RSS).
    """
    result_film = (
        result_gap.group_by("temperature", "field_strength")
        .agg(col("error", "cond_film").sort_by("error").first())
        .sort("temperature", "field_strength")
    )
    partial_results = []
    for temperature in result_film["temperature"].unique():
        partial_result = result_film.filter(col("temperature").eq(temperature))

        df_fit = partial_result.filter(col("field_strength") > 0.2)
        fit_result = Polynomial.fit(df_fit["field_strength"], df_fit["cond_film"], 1)
        partial_result = partial_result.with_columns(
            pl.lit(fit_result(partial_result["field_strength"])).alias("cond_film_fit")
        )
        partial_results.append(partial_result)
    result_film = pl.concat(partial_results)

    result_film = (
        result_film.select("temperature", "field_strength", "cond_film_fit")
        .join(
            result_gap.select("temperature", "field_strength", "cond_film"),
            on=["temperature", "field_strength"],
        )
        .with_columns(
            (col("cond_film") - col("cond_film_fit")).abs().alias("cond_diff")
        )
        .group_by("temperature", "field_strength")
        .agg(
            col("cond_diff", "cond_film_fit", "cond_film").sort_by("cond_diff").first()
        )
        .select("temperature", "field_strength", "cond_film")
        .sort("temperature", "field_strength")
    )

    return result_film


def solve_bilevel(spectra_calc, spectra_meas):
    logging.info("Relativizing calculated spectra ...")
    trel_calc = relativize_calc(spectra_calc)

    logging.info("Upsampling conductivity ...")
    trel_calc = upsample_conductivity(trel_calc)

    logging.info("Joining datasets ...")
    trel_joint = join_datasets(
        spectra_meas.select_data(r"^t.reldiff.*$").sort(auto=True),
        trel_calc.select_data(r"^t.reldiff.*$").sort(auto=True),
    )

    logging.info("Computing error ...")
    error = compute_error(trel_joint)

    logging.info("Optimizing gap ...")
    result_gap = optimize_gap(error)

    logging.info("Optimizing free background ...")
    result_film = optimize_free_background(result_gap)
    solution = trel_joint.join(
        result_gap.join(result_film, on=("temperature", "cond_film")),
        on=("temperature", "field_strength", "cond_film", "cond_gap"),
    )
    DATAFILES["solution_free"].write(solution.unnest(cs.contains("[c]")))

    logging.info("Optimizing rigid background ...")
    result_film = optimize_rigid_background(result_gap)
    solution = trel_joint.join(
        result_gap.join(result_film, on=("temperature", "cond_film")),
        on=("temperature", "field_strength", "cond_film", "cond_gap"),
    )
    DATAFILES["solution_rigid"].write(solution.unnest(cs.contains("[c]")))


def solve_fixed(spectra_calc, spectra_meas):
    logging.info("Resampling calculation in experimental temperature range ...")
    temperatures = spectra_meas["temperature"].unique().sort()

    spectra_calc = resample_temperature(spectra_calc, temperatures)

    logging.info("Relativizing calculated spectra ...")
    trel_calc = relativize_calc(spectra_calc)

    logging.info("Upsampling conductivity ...")
    trel_calc = trel_calc.regrid(
        pl.Series("cond_gap", np.geomspace(*trel_calc.extrema("cond_gap"), 1_000)),
        method="catmullrom",
    ).drop_nulls()

    trel_joint = join_datasets(
        spectra_meas.select_data(r"^t.reldiff.*$").sort(auto=True),
        trel_calc.select_data(r"^t.reldiff.*$"),
        on=("freq", "temperature"),
    )
    error = compute_error(trel_joint)
    result = optimize_gap(error)

    solution = trel_joint.join(result, on=trel_joint.id_vars)
    DATAFILES["solution_fixed"].write(solution.unnest(cs.contains("[c]")))


# %%
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Processing raw data ...")
    waveforms = process_waveforms()
    DATAFILES["waveforms_meas"].write(waveforms)

    logging.info("Computing Fourier transforms ...")
    spectra_meas = process_fourier(waveforms).filter(col("freq").is_between(0.2, 2.2))

    DATAFILES["spectra_meas"].write(spectra_meas.unnest(cs.contains("[c]")))

    logging.info("Loading calculated spectra ...")
    spectra_calc = calc.load()

    logging.info("Solving bilevel problems:\n")
    solve_bilevel(spectra_calc, spectra_meas)

    logging.info("Solving single-level problem:\n")
    solve_fixed(spectra_calc, spectra_meas)

    logging.info("Finished.")
