import logging

import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

from driven_nickelate.conductivity_mapping import calculation as calc
from driven_nickelate.conductivity_mapping.temperature import cond_from_temperatures
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.scripts.pump_probe.field_from_hwp import field_from_hwp
from driven_nickelate.tools import create_dataset, pl_fft

STORE_DIR = (
    PROJECT_PATHS.processed_data / "conductivity_mapping/susceptibility_evolution"
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
    time = pl.Series("time", np.arange(-10, 50, 3e-3))
    id_schema = {
        "dataset": pl.Utf8,
        "field_strength": pl.Float64,
        "temperature": pl.Float64,
        "pp_delay": pl.Float64,
    }
    data_meas = []

    field_strength = col("HWP").map_elements(
        lambda x: 192 * field_from_hwp(x), return_dtype=pl.Float64
    )
    path = col("Path").map_elements(
        lambda path: str(PROJECT_PATHS.root / path), return_dtype=pl.Utf8
    )

    logging.info("Processing `260223_xHWPfine` (0 ps) ...")
    waveforms = (
        create_dataset(
            pl.read_csv(
                PROJECT_PATHS.file_lists / "pump_probe/260223_xHWPfine.csv"
            ).select(
                pl.lit("260223_xHWPfine").alias("dataset"),
                field_strength.alias("field_strength"),
                col("Temperature").alias("temperature"),
                pl.lit(0.0).alias("pp_delay"),
                path.alias("path"),
            ),
            column_names=["delay", "X", "X SEM", "Y", "Y SEM"],
            index="delay",
            lockin_schema={"X": ("X", "Y")},
            id_schema=id_schema,
        )
        .with_columns(
            (6.67 * (9.77 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).alias("X"),
        )
        .rename({"delay": "time"})
        .regrid(time, method="catmullrom", fill_value=0)
        .select_data("X")
        .sort()
    )
    data_meas.append(waveforms)

    logging.info("Processing `240223_xHWPxT` (20 ps) ...")
    waveforms = (
        create_dataset(
            pl.read_csv(
                PROJECT_PATHS.file_lists / "pump_probe/240223_xHWPxT.csv",
                comment_prefix="#",
            ).select(
                pl.lit("240223_xHWPxT").alias("dataset"),
                field_strength.alias("field_strength"),
                col("Get temperature").round(0).cast(pl.Int32).alias("temperature"),
                pl.lit(20.0).alias("pp_delay"),
                path.alias("path"),
            ),
            index="delay",
            column_names=["delay", "X", "X SEM", "Y", "Y SEM"],
            lockin_schema={"X": ("X", "Y")},
            id_schema=id_schema,
        )
        .with_columns(
            (6.67 * (9.8 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).alias("X"),
        )
        .rename({"delay": "time"})
        .regrid(time, method="catmullrom", fill_value=0)
        .select_data("X")
        .sort()
    )
    data_meas.append(waveforms)

    logging.info("Processing `230223_xHWP_100ps_ProbeScan` (100 ps) ...")

    waveforms = (
        create_dataset(
            pl.read_csv(
                PROJECT_PATHS.file_lists / "pump_probe/230223_xHWP_100ps_ProbeScan.csv"
            ).select(
                pl.lit("230223_xHWP_100ps_ProbeScan").alias("dataset"),
                field_strength.alias("field_strength"),
                pl.lit(5.0).alias("temperature"),
                pl.lit(100.0).alias("pp_delay"),
                path.alias("path"),
            ),
            index="delay",
            column_names=["delay", "X", "X SEM", "Y", "Y SEM"],
            lockin_schema={"X": ("X", "Y")},
            id_schema=id_schema,
        )
        .with_columns(
            (6.67 * (9.77 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).alias("X"),
        )
        .rename({"delay": "time"})
        .regrid(time, method="catmullrom", fill_value=0)
        .select_data("X")
        .sort()
    )
    data_meas.append(waveforms)

    data_meas = Dataset(data_meas, index="time").with_columns(
        col("temperature").round(0)
    )
    return data_meas


def relativize_spectra_meas(spectra_meas) -> Dataset:
    spectra_ref = spectra_meas.filter(
        col("field_strength").eq(
            col("field_strength").min().over("temperature", "pp_delay")
        )
    ).select_data(
        ccol("X.real", "X.imag").alias("X_ref[c]", fields="X_ref"),
    )

    trel_meas = spectra_meas.select_data(
        ccol("X.real", "X.imag").alias("X[c]", fields="X"),
    ).join(spectra_ref, on=("freq", "temperature", "pp_delay"))

    trel_meas = (
        trel_meas.with_columns(
            ((ccol("X[c]") - ccol("X_ref[c]")) / ccol("X_ref[c]")).alias(
                "t.reldiff[c]", fields="t.reldiff"
            )
        )
        .select_data(col("t.reldiff[c]"))
        .sort("temperature", "field_strength", "freq")
    )
    return trel_meas


def process_calculation(temperatures) -> Dataset:
    df_calc = calc.load()

    temperature_mapping = pl.DataFrame(
        {"temperature": temperatures, "cond_film": cond_from_temperatures(temperatures)}
    )
    cond_film = temperature_mapping["cond_film"]
    cond_gap = df_calc.coord("cond_gap").extend(cond_film).sort()

    df_calc = df_calc.regrid(cond_film, method="catmullrom").regrid(
        cond_gap, method="catmullrom"
    )
    df_calc = df_calc.join(temperature_mapping, on=["cond_film"])
    df_calc.id_vars += ["temperature"]
    return df_calc


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
            on=["freq", "temperature"],
            suffix="_ref",
        )
        .with_columns(
            ((ccol("t[c]") - ccol("t[c]_ref")) / ccol("t[c]_ref")).alias(
                "t.reldiff[c]", fields="t.reldiff"
            )
        )
        .select_data("t.reldiff[c]")
    ).sort("cond_gap", "cond_film", "temperature", "freq")
    return trel_calc


def align_datasets(ds_meas: Dataset, ds_calc: Dataset) -> tuple[Dataset, Dataset]:
    frequencies = ds_calc.coord("freq")
    ds_meas = ds_meas.sort("freq").regrid(frequencies, method="catmullrom")
    ds_calc = ds_calc.sort("freq")
    return ds_meas, ds_calc


def join_datasets(ds_meas, ds_calc) -> Dataset:
    ds_meas, ds_calc = align_datasets(ds_meas, ds_calc)
    trel_joint = ds_meas.rename({"t.reldiff[c]": "t_meas.reldiff[c]"}).join(
        ds_calc.rename({"t.reldiff[c]": "t_calc.reldiff[c]"}),
        on=("freq", "temperature"),
    )
    trel_joint = trel_joint.select_data(col("^t_.*$")).sort(
        *trel_joint.id_vars, trel_joint.index
    )
    return trel_joint


def upsample_conductivity(df: Dataset) -> Dataset:
    cond_new = pl.Series(
        "cond_gap",
        np.geomspace(*df.extrema("cond_gap"), 1_000)[:-1],
    )
    return df.sort("cond_gap", "freq").regrid(cond_new, method="catmullrom")


def compute_error(trel_joint: Dataset) -> Dataset:
    error_expr = (
        (ccol("t_meas.reldiff[c]") - ccol("t_calc.reldiff[c]")).modulus().pow(2)
    )
    error = (
        trel_joint.with_columns(error_expr.alias("error"))
        .group_by(trel_joint.id_vars)
        .agg(col("error").sum().alias("error"))
    )
    return error


def minimize(trel_joint: Dataset, error: pl.DataFrame) -> Dataset:
    solution = error.group_by(
        col(trel_joint.id_vars).exclude("cond_gap", "cond_film")
    ).agg(col("error", "cond_gap", "cond_film").sort_by("error").first())
    return trel_joint.join(solution, on=trel_joint.id_vars)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Processing raw data:")
    waveforms = process_waveforms()
    DATAFILES["waveforms_meas"].write(waveforms)

    logging.info("Computing Fourier transforms ...")
    spectra_meas = pl_fft(waveforms.df, "time", id_vars=waveforms.id_vars)
    spectra_meas = Dataset(spectra_meas, index="freq", id_vars=waveforms.id_vars)
    spectra_meas = spectra_meas.filter(col("freq").is_between(0.1, 2.3))
    DATAFILES["spectra_meas"].write(spectra_meas)

    logging.info("Relativizing measured spectra ...")
    trel_meas = relativize_spectra_meas(spectra_meas)
    DATAFILES["trel_meas"].write(trel_meas.unnest(cs.contains("[c]")))

    logging.info("Processing calculated spectra ...")
    spectra_calc = process_calculation(spectra_meas["temperature"].unique().sort())

    logging.info("Relativizing calculated spectra ...")
    trel_calc = relativize_spectra_calc(spectra_calc).drop_nulls()
    DATAFILES["trel_calc"].write(trel_calc.unnest(cs.contains("[c]")))

    logging.info("Upsampling conductivity ...")
    trel_calc = upsample_conductivity(trel_calc).drop_nulls()

    logging.info("Joining datasets ...")
    trel_joint = join_datasets(trel_meas, trel_calc)

    error = compute_error(trel_joint)
    DATAFILES["error"].write(error)

    logging.info("Computing mapping ...")
    solution = minimize(trel_joint, error)
    DATAFILES["solution"].write(solution.unnest(cs.contains("[c]")))

    logging.info("Finished.")

    import seaborn as sns

    sns.lineplot(
        trel_calc.unnest(cs.contains("[c]")),
        x="freq",
        y="t.reldiff.imag",
        estimator=None,
        hue="cond_gap",
        units="cond_film",
        palette="Spectral",
    )

    sns.lineplot(
        trel_meas.unnest(cs.contains("[c]")),
        x="freq",
        y="t.reldiff.imag",
        estimator=None,
        hue="field_strength",
        units="temperature",
    )
