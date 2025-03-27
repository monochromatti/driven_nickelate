import logging

import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

import driven_nickelate.conductivity_mapping.calculation as calc
from driven_nickelate.conductivity_mapping.temperature import cond_from_temperatures
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import create_dataset, pl_fft

STORE_DIR = PROJECT_PATHS.processed_data / "conductivity_mapping/dynamics_slow"
STORE_DIR.mkdir(exist_ok=True, parents=True)

DATAFILES = {
    "waveforms_meas": Datafile(
        path=STORE_DIR / "waveforms_meas.csv",
        index="time",
        id_vars=["pp_delay"],
    ),
    "spectra_meas": Datafile(
        path=STORE_DIR / "spectra_meas.csv",
        index="freq",
        id_vars=["pp_delay"],
    ),
    "trel_joint": Datafile(
        path=STORE_DIR / "trel_joint.csv",
        index="freq",
        id_vars=["pp_delay", "cond_gap"],
    ),
    "solution": Datafile(
        path=STORE_DIR / "solution.csv",
        index="freq",
        id_vars=["pp_delay", "cond_gap"],
    ),
}


def process_waveforms() -> Dataset:
    # - Data import and processing
    time_values = pl.Series("time", np.arange(-10, 50, 3e-3))
    x0 = 9.88

    # Reference data
    reference_specs = {
        "time": "20h06m37s",
        "description": "ZnTe-PNPA delay dependence (fine, chopped probe) PUMP BLOCKED",
        "temperature_range": "5.13-10.95",
        "parameters": "HWP=65.00_Del=300.00_Dum=90.00",
    }
    filename = PROJECT_PATHS.raw_data / (
        "21.01.23/" + "_".join(v for v in reference_specs.values()) + ".txt"
    )
    waveforms_ref = (
        create_dataset(
            pl.DataFrame({"path": filename}),
            column_names=["delay", "X", "X SEM", "Y", "Y SEM"],
            index="delay",
            lockin_schema={"X.avg": ("X", "Y")},
        )
        .with_columns(
            (6.67 * (x0 - col("delay"))).alias("delay"),
            (col("X.avg") - col("X.avg").mean()).alias("X.avg"),
        )
        .rename({"delay": "time"})
        .regrid(time_values, method="catmullrom", fill_value=0)
    )

    # Pump-probe data
    waveforms = (
        create_dataset(
            pl.read_csv(
                PROJECT_PATHS.root
                / "file_lists/pump_probe/220123_xPP_ChoppedProbe.csv",
                comment_prefix="#",
                separator=";",
            ).with_columns(
                col("Path").map_elements(
                    lambda s: f"{str(PROJECT_PATHS.root)}/{s}", return_dtype=pl.Utf8
                )
            ),
            column_names=["delay", "X", "X SEM", "Y", "Y SEM"],
            index="delay",
            lockin_schema={"X.avg": ("X", "Y")},
            id_schema={"pp_delay": pl.Float32},
        )
        .with_columns(
            (6.67 * (x0 - col("delay"))).alias("delay"),
            (col("X.avg") - col("X.avg").mean()).alias("X.avg"),
        )
        .rename({"delay": "time"})
        .regrid(time_values, method="catmullrom", fill_value=0)
    )

    waveforms = waveforms.join(
        waveforms_ref.rename({"X.avg": "Xeq.avg"}), on="time", how="inner"
    ).with_columns(
        (1 * (col("X.avg") - col("Xeq.avg"))).alias("dX.avg"),
    )
    return waveforms


def process_fourier(waveforms: Dataset) -> Dataset:
    spectra_meas = pl_fft(waveforms, waveforms.index, waveforms.id_vars)
    spectra_meas = Dataset(spectra_meas, "freq", waveforms.id_vars).sort()
    spectra_meas = spectra_meas.select_data(
        ccol("dX.avg.real", "dX.avg.imag").alias("dX.avg[c]", fields="dX.avg"),
        ccol("Xeq.avg.real", "Xeq.avg.imag").alias("Xeq.avg[c]", fields="Xeq.avg"),
    ).select_data(
        (ccol("dX.avg[c]") / ccol("Xeq.avg[c]")).alias(
            "t.reldiff[c]", fields="t.reldiff"
        )
    )
    return spectra_meas


def relativize_calc(spectra_calc, cond_base) -> Dataset:
    cond_gap = (
        spectra_calc["cond_gap"]
        .unique()
        .extend(pl.Series("cond_gap", [cond_base]))
        .unique()
        .sort()
    )

    spectra_calc = spectra_calc.regrid(
        pl.Series("cond_film", [cond_base]), method="catmullrom"
    ).drop_nulls()
    spectra_calc = spectra_calc.regrid(cond_gap, method="catmullrom").drop_nulls()

    spectra_calc = spectra_calc.select_data(
        ccol("t.real", "t.imag").alias("t[c]", fields="t")
    )
    spectra_calc = spectra_calc.join(
        spectra_calc.filter(col("cond_gap").eq(col("cond_film"))),
        on="freq",
        suffix="_ref",
    ).select_data(
        ((ccol("t[c]") - ccol("t[c]_ref")) / ccol("t[c]_ref")).alias(
            "t.reldiff[c]", fields="t.reldiff"
        )
    )

    return spectra_calc


def align_frequencies(spectra_meas, spectra_calc) -> tuple[Dataset, Dataset]:
    min_freq = max(spectra_calc["freq"].min(), spectra_meas["freq"].min())
    max_freq = min(spectra_calc["freq"].max(), spectra_meas["freq"].max())
    frequencies = pl.Series("freq", np.arange(min_freq, max_freq, 1e-2))

    # Measured
    spectra_meas = spectra_meas.regrid(frequencies, method="cosine")

    # Calculated
    spectra_calc = (
        spectra_calc.sort("freq")
        .regrid(frequencies, method="cosine")
        .filter(col("cond_gap").le(1e5))
    )

    return spectra_meas.drop_nulls(), spectra_calc.drop_nulls()


def upsample_cond(spectra_calc):
    cond_new = pl.Series(
        "cond_gap",
        np.linspace(*spectra_calc.extrema("cond_gap"), 1000)[:-1],
    )
    spectra_calc = (
        spectra_calc.sort("cond_gap").regrid(cond_new, method="catmullrom").drop_nulls()
    )
    return spectra_calc


def join_datasets(df_meas, df_calc):
    return df_meas.select(
        col("freq"),
        col("pp_delay"),
        col("t.reldiff[c]").alias("t_meas.reldiff[c]", fields="t_meas.reldiff"),
    ).join(
        df_calc.select(
            col("freq"),
            col("cond_gap"),
            col("t.reldiff[c]").alias("t_calc.reldiff[c]", fields="t_calc.reldiff"),
        ),
        on="freq",
    )


def compute_errors(trel_joint) -> pl.DataFrame:
    error_expr = (
        (ccol("t_meas.reldiff[c]") - ccol("t_calc.reldiff[c]").conj()).modulus().pow(2)
    )
    errors = (
        trel_joint.select(
            col("freq"),
            col("pp_delay"),
            col("cond_gap"),
            error_expr.alias("error[c]", fields="error"),
        )
        .group_by("cond_gap", "pp_delay")
        .agg(
            ccol("error[c]").sum().alias("error"),
        )
        .with_columns((-1 * col("error").log()).alias("log-likelihood"))
    )
    return errors


def minimize(errors) -> pl.DataFrame:
    mapping = (
        errors.group_by("pp_delay")
        .agg(col("cond_gap", "error").sort_by("error").first())
        .sort("pp_delay")
    )
    return mapping


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Processing waveforms ...")
    waveforms = process_waveforms()
    DATAFILES["waveforms_meas"].write(waveforms)

    logging.info("Computing Fourier transforms ...")
    spectra_meas = process_fourier(waveforms)
    DATAFILES["spectra_meas"].write(spectra_meas.unnest(cs.contains("[c]")))

    logging.info("Importing calculation ...")
    spectra_calc = calc.load().with_columns(col("t.imag") * -1)
    logging.info("Aligning frequencies ...")
    spectra_meas, spectra_calc = align_frequencies(spectra_meas, spectra_calc)

    logging.info("Loading temperature-conductivity mapping ...")
    cond_base, cond_metallic = cond_from_temperatures([10.95, 200.0])

    logging.info("Relativizing calculation ...")
    trel_calc = relativize_calc(spectra_calc, cond_base).filter(
        col("cond_gap") >= col("cond_film") / 2
    )

    logging.info("Upsampling conductivity ...")
    trel_calc = upsample_cond(trel_calc)

    logging.info("Joining datasets ...")
    trel_joint = join_datasets(spectra_meas, trel_calc)
    DATAFILES["trel_joint"].write(
        trel_joint.sort_columns().unnest(cs.contains("[c]")),
    )

    logging.info("Computing errors ...")
    errors = compute_errors(trel_joint)

    logging.info("Minimizing ...")
    mapping = minimize(errors)

    logging.info("Writing solution ...")
    solution = trel_joint.join(mapping, on=["pp_delay", "cond_gap"]).unnest(
        cs.contains("[c]")
    )()
    DATAFILES["solution"].write(solution)

    logging.info("Done.")
