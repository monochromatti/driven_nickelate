import logging

import lmfit as lm
import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset
from scipy.integrate import simpson

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.scripts.pump_probe.field_from_hwp import field_from_hwp
from driven_nickelate.tools import create_dataset, pl_fft

STORE_DIR = PROJECT_PATHS.processed_data / "conductivity_mapping/nonlinear"
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


def process_waveforms() -> Dataset:
    def to_field(hwp):
        return 192 * field_from_hwp(hwp)

    files = pl.read_csv(
        PROJECT_PATHS.file_lists / "xHWPxT.csv",
        comment_prefix="#",
    ).select(
        col("HWP position")
        .map_elements(to_field, return_dtype=pl.Float64)
        .alias("field_strength"),
        col("Temperature").alias("temperature"),
        col("Path")
        .map_elements(lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8)
        .alias("path"),
    )
    waveforms = (
        create_dataset(
            files,
            column_names=["delay", "X", "Y"],
            index="delay",
            lockin_schema={"X": ("X", "Y")},
        )
        .with_columns(((2 / 0.29979) * (177.6 - col("delay"))).alias("time"))
        .set(index="time")
        .select(
            col("field_strength"),
            col("temperature"),
            col("time"),
            ((col("X") - col("X").mean()).over("field_strength")).alias("X"),
        )
    )

    files_ref = pl.read_csv(
        PROJECT_PATHS.file_lists / "nonlinear_probe" / "xHWPxT_REF.csv",
        comment_prefix="#",
    ).select(
        col("HWP position")
        .map_elements(to_field, return_dtype=pl.Float64)
        .alias("field_strength"),
        col("Path")
        .map_elements(lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8)
        .alias("path"),
    )

    waveforms_vac = (
        create_dataset(
            files_ref,
            column_names=["delay", "X", "Y"],
            index="delay",
            lockin_schema={"X": ("X", "Y")},
        )
        .with_columns(
            ((2 / 0.29979) * (177.6 - col("delay"))).alias("time"),
            (col("X") - col("X").first()).alias("X"),
        )
        .set(index="time")
        .drop("delay")
    )

    time = pl.Series("time", np.arange(-10, 30, 6e-2))
    waveforms = waveforms.regrid(time, method="cosine", fill_value=0)
    waveforms_vac = waveforms_vac.regrid(time, method="cosine", fill_value=0)

    waveforms = waveforms.join(
        waveforms_vac, on=["time", "field_strength"], suffix="_vac"
    ).sort("time")

    def time_window(t, t1, t2):
        left = np.min([np.ones(len(t)), np.exp((t - t1) / 0.5)], axis=0)
        right = np.min([np.ones(len(t)), np.exp(-(t - t2) / 0.5)], axis=0)
        return left + right - 1

    window = col("time").map_batches(lambda t: time_window(t, -5, 10))
    window_vac = col("time").map_batches(lambda t: time_window(t, -10, -3))
    waveforms = waveforms.with_columns(
        (col("X") * window).over(waveforms.id_vars).alias("X"),
        (col("X_vac") * window_vac).over(waveforms.id_vars).alias("X_vac"),
    )

    return waveforms


def normalize_waveforms(waveforms):
    def integrate_reference(s):
        t, y = s.struct[0], s.struct[1]
        return simpson(y, x=t)

    waveforms_vac = (
        waveforms.select("field_strength", "time", "X_vac").unique().sort("time")
    )
    field_norm = (
        waveforms_vac.group_by("field_strength")
        .agg(
            pl.struct(col("time"), col("X_vac") ** 2)
            .map_batches(lambda s: integrate_reference(s), return_dtype=pl.Float64)
            .get(0)
            .alias("norm")
        )
        .sort("field_strength")
    )

    model = lm.Model(lambda fs, a, b, c: fs * (a + fs * (b + fs * c)))
    params = model.make_params(a=1e-7, b=1e-7, c=0)
    field_norm_fit = model.fit(
        field_norm["norm"], params, fs=field_norm["field_strength"]
    )

    norm_expr = (
        col("field_strength")
        .map_batches(lambda fs: field_norm_fit.eval(fs=fs))
        .sqrt()
        .alias("field_norm")
    )
    waveforms = waveforms.with_columns(
        (col("X") / norm_expr).over("field_strength").alias("x"),
        (col("X_vac") / norm_expr).over("field_strength").alias("x_vac"),
    )
    return waveforms


def relativize_meas(waveforms):
    waveforms_ref = waveforms.filter(
        (col("temperature") == col("temperature").min())
        & (col("field_strength") == col("field_strength").min())
    ).select("time", "x", "X")
    return waveforms.join(waveforms_ref, suffix="_ref", on=["time"])


def process_fourier(waveforms: Dataset) -> Dataset:
    return (
        Dataset(
            pl_fft(waveforms, waveforms.index, waveforms.id_vars),
            index="freq",
            id_vars=waveforms.id_vars,
        )
        .select_data(
            ccol("x").alias("x[c]", fields="x"),
            ccol("x_ref").alias("x_ref[c]", fields="x_ref"),
        )
        .select_data(
            (ccol("x[c]") / col("x_ref[c]")).alias("t[c]", fields="t"),
            ((ccol("x[c]") - ccol("x_ref[c]")) / ccol("x_ref[c]")).alias(
                "t.reldiff[c]"
            ),
        )
    )


def upsample_conductivity(df: Dataset) -> Dataset:
    cond_gap = pl.Series(
        "cond_gap",
        np.geomspace(*df.extrema("cond_gap"), 1_000)[:-1],
    )
    df = df.sort("cond_gap", "freq").regrid(cond_gap, method="catmullrom").drop_nulls()
    return df


def align_datasets(ds_meas: Dataset, ds_calc: Dataset) -> tuple[Dataset, Dataset]:
    frequencies = freq_samples()

    ds_meas = ds_meas.sort("freq").regrid(frequencies).drop_nulls()
    ds_calc = ds_calc.sort("freq").regrid(frequencies).drop_nulls()

    return ds_meas, ds_calc


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


def join_datasets(ds_meas, ds_calc) -> Dataset:
    ds_meas, ds_calc = align_datasets(ds_meas, ds_calc)
    ds_meas = ds_meas.rename({"t.reldiff[c]": "t_meas.reldiff[c]"})
    ds_calc = ds_calc.rename({"t.reldiff[c]": "t_calc.reldiff[c]"})
    ds_joint = ds_meas.join(ds_calc, on="freq").select_data(col("^t_.*$")).sort()
    return ds_joint


def compute_error(trel_joint):
    freq_weights = (
        0.5 * (1 - (col("freq") - 1.8).tanh()) / 2
        + (-0.5 * ((col("freq") - 1.0) / 0.6) ** 2).exp()
    )
    residual = freq_weights * (col("t_meas.reldiff[c]") - col("t_calc.reldiff[c]"))
    error = trel_joint.group_by(trel_joint.id_vars).agg(
        residual.complex.squared_modulus().sum().alias("error")
    )
    return error


def compute_mapping(residuals) -> Dataset:
    mapping = residuals.group_by(["field_strength", "temperature"]).agg(
        col("cond_gap").sort_by("error").first(), col("error").min()
    )
    return mapping


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Processing raw data")
    waveforms = process_waveforms()
    DATAFILES["waveforms_meas"].write(waveforms)

    logging.info("Normalizing waveforms")
    waveforms = normalize_waveforms(waveforms)

    logging.info("Relativizing waveforms")
    waveforms = relativize_meas(waveforms)

    logging.info("Computing Fourier transforms")
    spectra_meas = process_fourier(waveforms)
    DATAFILES["spectra_meas"].write(spectra_meas.unnest(cs.contains("[c]")))

    logging.info("Importing calculations")
    spectra_calc = pl.read_csv(
        PROJECT_PATHS.root / "simulations/data/04.04.24/nonlinear/spectral_data.csv"
    )
    spectra_calc = (
        Dataset(spectra_calc, index="freq", id_vars=["cond_film", "cond_gap"])
        .select_data(ccol("t").alias("t[c]", fields="t"))
        .sort()
    )

    cond_film = pl.Series(
        "cond_film", np.geomspace(*spectra_calc.extrema("cond_film"), 100)[1:-1]
    )
    spectra_calc = spectra_calc.regrid(cond_film, method="cosine").drop_nulls()

    cond_gap = (
        spectra_calc.coord("cond_film")
        .extend(
            spectra_calc.filter(col("cond_gap") > col("cond_film").max()).coord(
                "cond_gap"
            )
        )
        .sort()
        .alias("cond_gap")
    )
    spectra_calc = spectra_calc.regrid(cond_gap, method="catmullrom").drop_nulls()

    logging.info("Relativizing calculated spectra")
    spectra0 = spectra_calc.filter(col("cond_gap") == col("cond_film")).select(
        col("cond_film"), col("freq"), col("t[c]").alias("t0[c]", fields="t")
    )
    spectra_calc = (
        spectra_calc.join(spectra0, on=["freq", "cond_film"], how="outer")
        .select_data(
            ((ccol("t[c]") - ccol("t0[c]")) / ccol("t0[c]"))
            .over("cond_film")
            .alias("t.reldiff[c]", fields="t.reldiff"),
            (ccol("t[c]") / ccol("t0[c]")).over("cond_film").alias("t[c]", fields="t"),
        )
        .sort()
    )

    # sns.relplot(
    #     spectra_calc.unnest(cs.contains("[c]")).filter(
    #         col("cond_film").is_in(col("cond_film").sample(fraction=0.1))
    #     ),
    #     x="freq",
    #     y="t.reldiff.real",
    #     hue="cond_gap",
    #     col="cond_film",
    #     col_wrap=3,
    #     kind="line",
    #     legend=False,
    # )

    logging.info("Joining datasets")
    spectra_joint = join_datasets(spectra_meas, spectra_calc)
    # DATAFILES["trel_joint"].write(trel_joint.unnest(cs.contains("[c]")))

    logging.info("Computing residuals")
    error = compute_error(spectra_joint.filter(col("freq").is_between(0.5, 2.0)))
    # DATAFILES["error"].write(error)

    # g = sns.relplot(
    #     error.filter(col("error") > 0).with_columns(col("error").log()),
    #     x="cond_gap",
    #     y="cond_film",
    #     hue="error",
    #     col="temperature",
    #     row="field_strength",
    #     height=3.4,
    #     s=5,
    #     ec=None,
    #     palette="Spectral",
    #     facet_kws=dict(margin_titles=True),
    # )
    # g.set(xscale="log", yscale="log")
    # plt.show()

    logging.info("Computing mapping")
    error = error.filter(
        col("cond_film").eq(
            col("cond_film").slice((col("cond_film") - 7054.802311).abs().arg_min(), 1)
        )
    )

    mapping = compute_mapping(error)
    # DATAFILES["mapping"].write(mapping)

    # logging.info("Finished.")
