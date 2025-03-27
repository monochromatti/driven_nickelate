import logging
import re

import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

from driven_nickelate.conductivity_mapping import calculation as calc
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import create_dataset, pl_fft

STORE_DIR = PROJECT_PATHS.root / "processed_data/abovegap"
print(STORE_DIR)
STORE_DIR.mkdir(exist_ok=True)

DATAFILES = {
    "waveforms_meas": Datafile(
        path=STORE_DIR / "waveforms_meas,csv",
        index="time",
        id_vars=["field_strength", "temperature"],
    ),
    "spectra_meas": Datafile(
        path=STORE_DIR / "spectra_meas.csv",
        index="freq",
        id_vars=["field_strength", "temperature"],
    ),
    "trel_meas": Datafile(
        path=STORE_DIR / "trel_meas.csv",
        index="freq",
        id_vars=["field_strength", "temperature"],
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
        id_vars=["cond_gap", "cond_film", "field_strength", "temperature"],
    ),
}


def process_waveforms() -> Dataset:
    time_values = pl.Series("time", np.arange(-1, 12, 3e-3))
    file_longtimes = "nir_pump/050323_xPP_TransientProbeTraces.csv"
    waveforms = (
        create_dataset(
            pl.read_csv(
                PROJECT_PATHS.file_lists / file_longtimes,
                comment_prefix="#",
            ).select(
                (6.67 * (147.4 - col("Delay"))).alias("pp_delay"),
                col("Path")
                .map_elements(
                    lambda path: str(PROJECT_PATHS.root / path), return_dtype=pl.Utf8
                )
                .alias("path"),
            ),
            index="delay",
            column_names=["delay", "X", "X SEM", "Y", "Y SEM"],
            lockin_schema={"dX": ("X", "Y")},
            id_schema={"pp_delay": pl.Float32},
        )
        .with_columns(
            (6.67 * (9.77 - col("delay"))).alias("delay"),
            (col("dX") - col("dX").mean()).alias("dX"),
        )
        .rename({"delay": "time"})
        .sort("time")
        .regrid(time_values, method="catmullrom", fill_value=0)
    )

    waveform_ref = (
        create_dataset(
            pl.DataFrame(
                {
                    "path": PROJECT_PATHS.raw_data
                    / "04.03.23/16h14m00s_ZnTe chopped_2.93-4.28_Del=140.00_HWP=25.00.txt"
                }
            ),
            index="delay",
            column_names=["delay", "X", "X SEM", "Y", "Y SEM"],
            lockin_schema={"X": ("X", "Y")},
        )
        .with_columns(
            (6.67 * (9.77 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).alias("X"),
        )
        .rename({"delay": "time"})
        .sort("time")
        .regrid(time_values, method="catmullrom", fill_value=0)
    )

    return waveforms.join(waveform_ref, on="time").select(
        "pp_delay", "time", (col("X") + col("dX")).alias("X")
    )


def process_fourier(waveforms: Dataset) -> Dataset:
    spectra_meas = pl_fft(waveforms, "time", id_vars=waveforms.id_vars)
    spectra_meas = Dataset(spectra_meas, "freq", waveforms.id_vars)
    return spectra_meas


def relativize_meas(spectra_meas) -> Dataset:
    spectra_meas = spectra_meas.select_data(
        ccol("X.real", "X.imag").alias("X[c]", fields="X")
    )
    trel_meas = (
        spectra_meas.join(
            spectra_meas.filter(col("pp_delay") == col("pp_delay").min()),
            on="freq",
            suffix="_ref",
        )
        .select_data(
            ((ccol("X[c]") - ccol("X[c]_ref")) / ccol("X[c]_ref")).alias(
                "t.reldiff[c]", fields="t.reldiff"
            )
        )
        .filter(col("pp_delay") > col("pp_delay").min())
        .select_data(col("t.reldiff[c]"))
        .sort()
    )
    return trel_meas


def relativize_calc(spectra_calc) -> Dataset:
    cond = pl.Series("cond", np.geomspace(1e3, 5e4, 250))
    spectra_calc = (
        spectra_calc.regrid(cond.rename("cond_film"), method="catmullrom")
        .filter(col("cond_film").is_between(1e3, 5e4))
        .drop_nulls()
    )
    spectra_calc = (
        spectra_calc.regrid(cond.rename("cond_gap"), method="catmullrom")
        .filter(col("cond_gap").is_between(1e3, 5e4))
        .drop_nulls()
    )
    spectra_calc = spectra_calc.select_data(
        ccol("t.real", "t.imag").alias("t[c]", fields="t")
    )

    spectra_eq = spectra_calc.filter(col("cond_gap").eq(col("cond_film"))).drop(
        "cond_gap"
    )  # Equilibrium => cond_gap = cond_film
    trel_calc = (
        spectra_calc.join(
            spectra_eq, on=[*spectra_eq.id_vars, spectra_eq.index], suffix="_ref"
        )
        .select_data(
            ((ccol("t[c]") - ccol("t[c]_ref")) / ccol("t[c]_ref")).alias(
                "t.reldiff[c]", fields="t.reldiff"
            )
        )
        .sort()
    )

    return trel_calc


def align_datasets(spectra_meas, spectra_calc):
    spectra_calc = spectra_calc.sort("freq")
    spectra_meas = (
        spectra_meas.sort("freq")
        .regrid(spectra_calc.coord("freq"), method="catmullrom")
        .drop_nulls()
    )
    return spectra_meas, spectra_calc


def join_datasets(ds_meas, ds_calc) -> Dataset:
    ds_meas, ds_calc = align_datasets(ds_meas, ds_calc)

    trel_joint = ds_meas.select_data(
        col("^t.*$").name.map(lambda name: re.sub(r"t(?=\[|\.)", "t_meas", name)),
    ).join(
        ds_calc.select_data(
            col("^t.*$").name.map(lambda name: re.sub(r"t(?=\[|\.)", "t_calc", name)),
        ),
        on="freq",
    )
    trel_joint = trel_joint.select_data(col("^t.*$")).sort()
    return trel_joint


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
    signal_expr = ccol("t_meas.reldiff[c]").modulus().pow(2)
    error = (
        trel_joint.filter(col("freq").is_between(0.5, 2.0))
        .with_columns(
            freq_weights.alias("weights"),
            error_expr.alias("error"),
            signal_expr.alias("signal"),
        )
        .group_by(trel_joint.id_vars)
        .agg(
            ((col("error") * col("weights")).mean() / col("signal").sum()).alias(
                "error"
            )
        )
    )
    return error


def minimize(trel_joint: Dataset, error: pl.DataFrame) -> Dataset:
    solution = error.group_by("pp_delay").agg(
        col("error", "cond_gap", "cond_film").sort_by("error").first()
    )
    return trel_joint.join(solution, on=trel_joint.id_vars)


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

    logging.info("Relativizing measured spectra ...")
    trel_meas = relativize_meas(spectra_meas)
    DATAFILES["trel_meas"].write(trel_meas.unnest(cs.contains("[c]")))

    logging.info("Processing calculated spectra ...")
    spectra_calc = calc.load()
    DATAFILES["spectra_calc"].write(spectra_calc)

    logging.info("Relativizing calculated spectra ...")
    trel_calc = relativize_calc(spectra_calc)
    DATAFILES["trel_calc"].write(trel_calc.unnest(cs.contains("[c]")))

    logging.info("Joining datasets ...")
    trel_joint = join_datasets(trel_meas, trel_calc)

    logging.info("Computing error ...")
    error = compute_error(trel_joint)
    DATAFILES["error"].write(error)

    logging.info("Computing mapping ...")
    solution = minimize(trel_joint, error).with_columns(
        (ccol("t_meas.reldiff[c]") - ccol("t_calc.reldiff[c]")).alias(
            "residual[c]", fields="residual"
        )
    )
    DATAFILES["solution"].write(solution.unnest(cs.contains("[c]")))

    logging.info("Done.")
