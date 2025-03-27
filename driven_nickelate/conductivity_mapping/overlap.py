import logging
import re

import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

from driven_nickelate.conductivity_mapping.calculation import (
    DATAFILES as CALCULATION_DATAFILES,
)
from driven_nickelate.conductivity_mapping.temperature import (
    DATAFILES as TEMPERATURE_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import create_dataset, pl_fft

STORE_DIR = PROJECT_PATHS.processed_data / "conductivity_mapping/overlap"
STORE_DIR.mkdir(exist_ok=True, parents=True)

DATAFILES = {
    "waveforms_meas": Datafile(
        path=STORE_DIR / "waveforms_meas.csv", index="time", id_vars=["pp_delay"]
    ),
    "spectra_meas": Datafile(
        path=STORE_DIR / "spectra_meas.csv", index="freq", id_vars=["pp_delay"]
    ),
    "trel_meas": Datafile(
        path=STORE_DIR / "trel_meas.csv", index="freq", id_vars=["pp_delay"]
    ),
    "trel_joint": Datafile(
        path=STORE_DIR / "trel_joint.csv",
        index="freq",
        id_vars=["pp_delay", "cond_gap"],
    ),
    "error": Datafile(
        path=STORE_DIR / "error.csv",
    ),
    "mapping": Datafile(
        path=STORE_DIR / "mapping.csv",
    ),
}


def load_conductivity_limits() -> tuple[float, float]:
    cond_limits = (
        TEMPERATURE_DATAFILES["solution"]
        .load()
        .fetch("direction", "temperature", "cond")
        .group_by("direction")
        .agg(col("cond").max().alias("max"), col("cond").min().alias("min"))
        .mean()
    )

    cond_I = cond_limits["min"].item()
    cond_M = cond_limits["max"].item()

    return cond_I, cond_M


def process_waveforms() -> Dataset:
    zero_pump_probe = 163.75
    zero_probe_gate = 9.88
    time = pl.Series("time", np.arange(-10, 50, 3e-3))

    paths = []
    dates = ["14.02.23", "13.02.23", "12.02.23", "11.02.23"]
    for date in dates:
        datepath = PROJECT_PATHS.raw_data / date
        paths += [datepath / filename for filename in datepath.glob("*.txt")]

    paths = list(filter(lambda x: x.suffix == ".txt", paths))
    paths = list(
        filter(lambda x: re.split("_", x.name)[1] == "2D THz pump-induced", paths)
    )

    delays = []
    for path in paths:
        delays.append(float(re.search(r"Del=(\d+\.\d+)", path.name).group(1)))
    delays, paths = list(zip(*sorted(zip(delays, paths))))

    column_names = ["delay"] + [
        c
        for s in (" 166 Hz", " 333 Hz")
        for c in [f"X{s}", f"X SEM{s}", f"Y{s}", f"Y SEM{s}"]
    ]
    paths = pl.DataFrame({"pp_delay": delays, "path": [str(path) for path in paths]})
    waveforms_pp = (
        create_dataset(
            paths=paths,
            column_names=column_names,
            index="delay",
            lockin_schema={
                "dX": ("X 166 Hz", "Y 166 Hz"),
            },
            id_schema={"pp_delay": pl.Float64},
        )
        .with_columns(
            (6.67 * (zero_probe_gate - col("delay"))).alias("delay"),
            (6.67 * (zero_pump_probe - col("pp_delay"))).alias("pp_delay"),
            (col("dX") - col("dX").mean()).alias("dX"),
        )
        .rename({"delay": "time"})
        .regrid(time, method="catmullrom", fill_value=0)
    )
    column_names = ["delay"] + [
        c
        for x in ("", ".1")
        for s in ("", "p")
        for c in [f"X{s}{x}", f"X{s}{x} SEM", f"Y{s}{x}", f"Y{s}{x} SEM"]
    ]

    path_ref = (
        PROJECT_PATHS.raw_data
        / "14.02.23/13h57m29s_Probe 500 Hz_3.87-5.52_Del=163.52.txt"
    )
    waveform_eq = (
        create_dataset(
            pl.DataFrame({"path": path_ref}),
            column_names,
            index="delay",
            lockin_schema={"X": ("X", "Y")},
            id_schema={},
        )
        .with_columns(
            (6.67 * (zero_probe_gate - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).alias("X"),
        )
        .rename({"delay": "time"})
        .regrid(time, method="catmullrom", fill_value=0)
        .select(col("time"), (-1 * col("X")).alias("X"))
    )
    waveforms = waveforms_pp.join(waveform_eq, on="time")
    return waveforms


def process_fourier(waveforms: Dataset) -> Dataset:
    spectra_meas = pl_fft(waveforms, "time", id_vars=waveforms.id_vars)
    spectra_meas = Dataset(spectra_meas, "freq", waveforms.id_vars).sort()
    return spectra_meas


def relativize_meas(spectra_meas: Dataset) -> Dataset:
    spectra_meas = (
        spectra_meas.select_data(
            ccol("dX.real", "dX.imag").alias("dX[c]", fields="dX"),
            ccol("X.real", "X.imag").alias("X[c]", fields="X"),
        )
        .with_columns(
            (ccol("dX[c]") / ccol("X[c]"))
            .conj()
            .alias("t.reldiff[c]", fields="t.reldiff"),
        )
        .select_data("t.reldiff[c]")
    )
    return spectra_meas


def relativize_calc(spectra_calc) -> Dataset:
    cond_I, _ = load_conductivity_limits()

    cond_film = (
        spectra_calc.coord("cond_film").extend(pl.Series("cond_film", [cond_I])).sort()
    )
    cond_gap = (
        spectra_calc.coord("cond_gap").extend(pl.Series("cond_gap", [cond_I])).sort()
    )
    spectra_calc = spectra_calc.regrid(cond_film, method="catmullrom").drop_nulls()
    spectra_calc = spectra_calc.regrid(cond_gap, method="catmullrom").drop_nulls()

    spectra_calc = spectra_calc.select_data(
        ccol("t.real", "t.imag").alias("t[c]", fields="t"),
    ).filter(col("cond_film").eq(cond_I))

    trel_calc = (
        spectra_calc.join(
            spectra_calc.filter(col("cond_gap").eq(col("cond_film"))),
            on="freq",
            suffix="_ref",
        )
        .with_columns(
            ((ccol("t[c]") - ccol("t[c]_ref")) / ccol("t[c]_ref")).alias(
                "t.reldiff[c]", fields="t.reldiff"
            )
        )
        .select_data("t.reldiff[c]")
    ).sort("cond_gap", "freq")
    return Dataset(trel_calc, index="freq", id_vars=["cond_gap"])


def upsample_conductivity(df: Dataset) -> Dataset:
    cond_gap = pl.Series(
        "cond_gap",
        np.geomspace(*df.extrema("cond_gap"), 1_000)[:-1],
    )
    df = df.sort("cond_gap", "freq").regrid(cond_gap, method="catmullrom").drop_nulls()
    return df


def align_datasets(ds_meas: Dataset, ds_calc: Dataset) -> tuple[Dataset, Dataset]:
    frequencies = ds_calc["freq"].unique().sort()

    ds_meas = ds_meas.sort("freq").regrid(frequencies).drop_nulls()
    ds_calc = ds_calc.sort("freq")

    return ds_meas, ds_calc


def join_datasets(ds_meas, ds_calc) -> Dataset:
    ds_meas, ds_calc = align_datasets(ds_meas, ds_calc)
    ds_meas = ds_meas.rename({"t.reldiff[c]": "t_meas.reldiff[c]"})
    ds_calc = ds_calc.rename({"t.reldiff[c]": "t_calc.reldiff[c]"})
    trel_joint = ds_meas.join(ds_calc, on="freq").select_data(col("^t.*$")).sort()
    return trel_joint


def compute_error(trel_joint):
    freq_weights = (
        0.5 * (1 - (col("freq") - 1.8).tanh()) / 2
        + (-0.5 * ((col("freq") - 1.0) / 0.6) ** 2).exp()
    )
    error_expr = (
        (ccol("t_calc.reldiff[c]") - ccol("t_meas.reldiff[c]")).modulus().pow(2)
    )
    error = (
        trel_joint.with_columns(error_expr.alias("error"), freq_weights.alias("weight"))
        .group_by(trel_joint.id_vars)
        .agg((col("error") * col("weight")).mean().alias("error"))
    )
    error = Dataset(error, "pp_delay")
    return error


def compute_mapping(residuals) -> Dataset:
    mapping = Dataset(
        (
            residuals.group_by(residuals.index).agg(
                col("cond_gap", "error").sort_by("error").first()
            )
        ),
        index="pp_delay",
    )
    return mapping


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Processing raw data ...")
    waveforms = process_waveforms()
    DATAFILES["waveforms_meas"].write(waveforms)

    logging.info("Computing Fourier transforms ...")
    spectra_meas = process_fourier(waveforms)
    DATAFILES["spectra_meas"].write(spectra_meas)

    logging.info("Relativizing measured spectra ...")
    trel_meas = relativize_meas(spectra_meas)

    logging.info("Loading calculated spectra ...")
    spectra_calc = (
        CALCULATION_DATAFILES["spectra"].load().with_columns(col("t.imag") * -1)
    )

    logging.info("Relativizing calculated spectra ...")
    trel_calc = relativize_calc(spectra_calc)

    logging.info("Upsampling conductivity ...")
    trel_calc = upsample_conductivity(trel_calc).filter(
        pl.col("cond_gap") >= pl.col("cond_film") / 2
    )

    logging.info("Joining datasets ...")
    trel_joint = join_datasets(trel_meas, trel_calc)
    DATAFILES["trel_joint"].write(trel_joint.unnest(cs.contains("[c]")))

    logging.info("Computing residuals ...")
    error = compute_error(trel_joint)
    DATAFILES["error"].write(error)

    logging.info("Computing mapping ...")
    mapping = compute_mapping(error)
    DATAFILES["mapping"].write(mapping)

    logging.info("Finished.")
