import logging

import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset
from scipy.interpolate import PchipInterpolator

from driven_nickelate.conductivity_mapping import calculation as calc
from driven_nickelate.conductivity_mapping.temperature import (
    DATAFILES as TEMPERATURE_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.scripts.pump_probe.field_from_hwp import field_from_hwp
from driven_nickelate.tools import create_dataset, pl_fft

STORE_DIR = PROJECT_PATHS.processed_data / "conductivity_mapping/susceptibility_long"
STORE_DIR.mkdir(exist_ok=True, parents=True)

DATAFILES = {
    "waveforms_meas": Datafile(
        path=STORE_DIR / "waveforms_meas.csv",
        index="time",
        id_vars=["field_strength", "pp_delay"],
    ),
    "spectra_meas": Datafile(
        path=STORE_DIR / "spectra_meas.csv",
        index="freq",
        id_vars=["field_strength", "pp_delay"],
    ),
    "trel_meas": Datafile(
        path=STORE_DIR / "trel_meas.csv",
        index="freq",
        id_vars=["field_strength", "pp_delay"],
    ),
    "spectra_calc": Datafile(
        path=STORE_DIR / "spectra_calc.csv",
        index="freq",
        id_vars=["cond_gap", "cond_film"],
    ),
    "trel_calc": Datafile(
        path=STORE_DIR / "trel_calc.csv",
        index="freq",
        id_vars=["cond_gap", "cond_film"],
    ),
    "error": Datafile(
        path=STORE_DIR / "error.csv",
    ),
    "solution": Datafile(
        path=STORE_DIR / "solution.csv",
        index="freq",
        id_vars=["cond_gap", "cond_film", "field_strength", "pp_delay"],
    ),
}


def process_waveforms() -> None:
    waveforms = (
        create_dataset(
            pl.read_csv(
                PROJECT_PATHS.file_lists / "pump_probe/050223_xPP_xHWP.csv",
                comment_prefix="#",
            ).with_columns(
                col("Path").map_elements(
                    lambda path: str(PROJECT_PATHS.root / path), return_dtype=pl.Utf8
                ),
            ),
            column_names=["delay", "X", "X SEM", "Y", "Y SEM"],
            index="delay",
            lockin_schema={"X": ("X", "Y")},
            id_schema={
                "hwp": pl.Float32,
                "pp_position": pl.Float32,
                "pp_delay": pl.Float32,
            },
        )
        .select(
            col("hwp")
            .map_elements(field_from_hwp, return_dtype=pl.Float64)
            .round(2)
            .alias("field_strength"),
            col("pp_delay"),
            (6.67 * (9.77 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).alias("X"),
        )
        .rename({"delay": "time"})
        .group_by("time", "field_strength", "pp_delay")
        .agg(col("X").mean())
        .sort("time")
    )
    waveforms = Dataset(waveforms, index="time", id_vars=["field_strength", "pp_delay"])
    waveforms = waveforms.regrid(
        pl.Series("time", np.arange(-10, 50, 3e-3)), method="catmullrom", fill_value=0
    )
    return waveforms


def combine_close_times(waveforms):
    keys = waveforms.fetch(col("pp_delay").unique().sort())
    keys = (
        keys.with_columns(
            col("pp_delay").diff().alias("diff"),
        )
        .join(
            keys.with_columns(
                col("pp_delay").shift(-1).diff().alias("diff"),
            ),
            on="diff",
            suffix="_match",
        )
        .filter((col("pp_delay") - col("pp_delay_match")).abs() <= 1)
        .drop_nulls()
        .drop("diff")
    )
    from functools import reduce

    agg_dfs = []
    for x in keys.iter_rows():
        df = reduce(
            lambda df1, df2: df1.join(df2, on=["time", "field_strength"]),
            waveforms.filter(col("pp_delay").is_in(x)).partition_by("pp_delay"),
        ).select(
            col("time"),
            col("field_strength"),
            pl.reduce(lambda a, b: (a + b) / 2, col("^pp_delay.*$")).alias("pp_delay"),
            pl.reduce(lambda a, b: (a + b) / 2, col("^X.*$")).alias("X"),
        )

        agg_dfs.append(df)

    waveforms = waveforms.filter(
        ~col("pp_delay").is_in(pl.concat(keys.select("pp_delay", "pp_delay_match")))
    ).vstack(pl.concat(agg_dfs))

    return waveforms


def process_fourier(waveforms: Dataset) -> Dataset:
    spectra_meas = (
        pl_fft(waveforms.df, waveforms.index, id_vars=waveforms.id_vars)
        .sort("field_strength", "pp_delay", "freq")
        .with_columns(col("X.imag") * -1)
    )
    return Dataset(spectra_meas, index="freq", id_vars=waveforms.id_vars)


def relativize_spectra_meas(spectra_meas) -> Dataset:
    spectra_ref = (
        spectra_meas.filter(col("pp_delay") < -10)
        .group_by("freq")
        .agg(
            col("X.real").mean().alias("X_ref.real"),
            col("X.imag").mean().alias("X_ref.imag"),
        )
        .select(
            col("freq"),
            ccol("X_ref.real", "X_ref.imag").alias("X_ref[c]", fields="X_ref"),
        )
    )
    trel_meas = (
        spectra_meas.select_data(ccol("X.real", "X.imag").alias("X[c]", fields="X"))
        .join(spectra_ref, on="freq")
        .with_columns(
            ((ccol("X[c]") - ccol("X_ref[c]")) / ccol("X_ref[c]")).alias(
                "t.reldiff[c]", fields="t.reldiff"
            )
        )
        .select_data(col("t.reldiff[c]"))
        .sort("field_strength", "pp_delay", "freq")
    )
    return trel_meas


def process_calculation() -> Dataset:
    spectra_calc = calc.load()

    temperature_mapping = (
        TEMPERATURE_DATAFILES["solution"]
        .load()
        .select("cond", "direction", "temperature")
        .unique()
        .sort("direction", "temperature")
        .with_columns(
            col("temperature").cum_count().over("direction").alias("index"),
        )
        .select(
            col("temperature", "cond").mean().over("index"),
        )
        .unique("temperature")
        .sort("temperature")
    )
    temp_base = 4.7  # [K]
    cond_base = PchipInterpolator(*temperature_mapping.get_columns())([temp_base])

    cond_film = pl.Series("cond_film", cond_base)
    cond_gap = spectra_calc.get_column("cond_gap").unique().extend(cond_film).sort()

    spectra_calc = (
        spectra_calc.regrid(cond_film, method="catmullrom")
        .regrid(cond_gap, method="catmullrom")
        .drop_nulls()
        .sort("cond_gap", "cond_film", "freq")
        .sort_columns()
    )
    return spectra_calc


def relativize_spectra_calc(spectra_calc) -> Dataset:
    spectra_calc = spectra_calc.select_data(
        ccol("t.real", "t.imag").alias("t[c]", fields="t")
    )

    trel_calc = (
        spectra_calc.join(
            spectra_calc.with_columns(
                (col("cond_gap") - col("cond_film")).abs().alias("diff")
            )
            .filter(col("diff").eq(col("diff").min()))
            .drop("diff"),
            on="freq",
            suffix="_ref",
        )
        .with_columns(
            ((ccol("t[c]") - ccol("t[c]_ref")) / ccol("t[c]_ref")).alias(
                "t.reldiff[c]", fields="t.reldiff"
            )
        )
        .select_data("t.reldiff[c]")
    ).sort("cond_gap", "cond_film", "freq")
    return trel_calc


def align_datasets(spectra_meas, spectra_calc):
    min_freq = max(spectra_calc["freq"].min(), spectra_meas["freq"].min())
    max_freq = min(spectra_calc["freq"].max(), spectra_meas["freq"].max())
    frequencies = pl.Series("freq", np.arange(min_freq, max_freq, 1e-2))

    spectra_meas = spectra_meas.sort("freq").regrid(frequencies)
    spectra_calc = spectra_calc.sort("freq").regrid(frequencies)

    return spectra_meas, spectra_calc


def join_datasets(ds_meas, ds_calc) -> Dataset:
    ds_meas, ds_calc = align_datasets(ds_meas, ds_calc)

    trel_joint = ds_meas.rename({"t.reldiff[c]": "t_meas.reldiff[c]"}).join(
        ds_calc.rename({"t.reldiff[c]": "t_calc.reldiff[c]"}), on="freq"
    )
    trel_joint = trel_joint.select_data(col("^t.*$")).sort(auto=True)
    return trel_joint


def upsample_conductivity(df: Dataset) -> Dataset:
    cond_new = pl.Series(
        "cond_gap",
        np.geomspace(*df.extrema("cond_gap"), 1_000)[:-1],
    )
    return (
        df.sort("cond_gap", "freq").regrid(cond_new, method="catmullrom").drop_nulls()
    )


def compute_error(trel_joint: Dataset) -> Dataset:
    freq_weights = 0.5 * (
        1 - (col("freq") - 1.8).tanh()
    )  # Tapers off at higher frequencies
    freq_weights += (
        -0.5 * ((col("freq") - 1.0) / 0.6) ** 2
    ).exp()  # Extra attention to the peak

    error_expr = (
        (ccol("t_meas.reldiff[c]") - ccol("t_calc.reldiff[c]")).modulus().pow(2)
    )
    error = (
        trel_joint.with_columns(error_expr.alias("error"), freq_weights.alias("weight"))
        .group_by("field_strength", "pp_delay", "cond_gap", "cond_film")
        .agg((col("error") * col("weight")).sum().alias("error"))
    )
    return error


def minimize(trel_joint: Dataset, error: pl.DataFrame) -> Dataset:
    solution = error.group_by("field_strength", "pp_delay").agg(
        col("error", "cond_gap", "cond_film").sort_by("error").first()
    )
    return trel_joint.join(solution, on=trel_joint.id_vars)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Processing raw data ...")
    waveforms = process_waveforms()
    waveforms = combine_close_times(waveforms)
    DATAFILES["waveforms_meas"].write(waveforms.unnest(cs.contains("[c]")))

    logging.info("Computing Fourier transforms ...")
    spectra_meas = process_fourier(waveforms)
    DATAFILES["spectra_meas"].write(spectra_meas.unnest(cs.contains("[c]")))

    logging.info("Relativizing measured spectra ...")
    trel_meas = relativize_spectra_meas(spectra_meas)
    DATAFILES["trel_meas"].write(trel_meas.unnest(cs.contains("[c]")))

    logging.info("Processing calculated spectra ...")
    spectra_calc = process_calculation()
    DATAFILES["spectra_calc"].write(spectra_calc.unnest(cs.contains("[c]")))

    logging.info("Relativizing calculated spectra ...")
    trel_calc = relativize_spectra_calc(spectra_calc)
    DATAFILES["trel_calc"].write(trel_calc.unnest(cs.contains("[c]")))

    logging.info("Upsampling conductivity ...")
    trel_calc = upsample_conductivity(trel_calc)

    logging.info("Joining datasets ...")
    trel_joint = join_datasets(trel_meas, trel_calc)

    error = compute_error(trel_joint)
    DATAFILES["error"].write(error)

    logging.info("Computing mapping ...")
    solution = minimize(trel_joint, error)
    DATAFILES["solution"].write(solution.unnest(cs.contains("[c]")))

    logging.info("Finished.")
