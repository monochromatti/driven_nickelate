import re
from datetime import datetime
from pathlib import Path

import polars as pl
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import zero_quadrature

STORE_DIR = PROJECT_PATHS.root / "processed_data/linear_spectroscopy/nosample"
STORE_DIR.mkdir(exist_ok=True)

DATAFILES = {
    "waveforms": Datafile(
        path=STORE_DIR / "nosample.csv",
        index="delay",
        id_vars=[
            "detector",
            "source",
            "date",
            "temperature",
            "description",
            "filename",
        ],
    )
}


def generate_column_names(num_averages: int = 0, has_sem=True) -> list[str]:
    suffixes = [""] + [f".{i}" for i in range(1, num_averages + 1)]
    if has_sem:
        return ["delay"] + [
            n for s in suffixes for n in [f"X{s}", f"X{s} SEM", f"Y{s}", f"Y{s} SEM"]
        ]
    return ["delay", "X", "Y"]


def find_date(filename: str):
    match = re.search(r"\d{2}\.\d{2}\.\d{2}", filename)
    if match:
        value = match.group()
        return datetime.strptime(value, "%d.%m.%y").date()
    return None


def find_description(path: str) -> str:
    return re.search(r"\d{2}h\d{2}m\d{2}s_(.*?)_\d{3}\.\d{2}-\d{3}\.\d{2}", path).group(
        1
    )


def find_source(s: str) -> str:
    if "PNPA" in s:
        return "PNPA"
    else:
        return "ZnTe"


def find_detector(s: str) -> str:
    if "ZnTe" in s:
        return "ZnTe"
    else:
        return "GaP"


def import_data(paths: pl.DataFrame):
    columns = ("detector", "source", "date", "temperature", "description", "path")
    if not (set(paths.columns) & set(columns)):
        raise ValueError("The `paths` DataFrame does not have correct columns.")

    queries = []
    column_specs = {
        3: {"num_averages": 0, "has_sem": False},
        5: {"num_averages": 0, "has_sem": True},
        9: {"num_averages": 1, "has_sem": True},
    }
    for vals in paths.iter_rows():
        width = (
            pl.scan_csv(vals[-1], comment_prefix="#", separator="\t", has_header=False)
            .collect_schema()
            .len()
        )
        num_avg, has_sem = column_specs.get(width, None).values()
        q = (
            pl.scan_csv(
                vals[-1],
                comment_prefix="#",
                separator="\t",
                has_header=False,
                new_columns=generate_column_names(num_avg, has_sem),
            )
            .select(generate_column_names(0, has_sem))
            .with_columns(ccol("X", "Y").map_batches(zero_quadrature).alias("X.avg"))
            .with_columns(col("X.avg") - col("X.avg").mean())
        )

        q = (
            q.with_columns(
                ccol("X SEM", "Y SEM").map_batches(zero_quadrature).alias("X.sem")
            )
            if has_sem
            else q.with_columns(pl.lit(None).alias("X.sem"))
        )

        q = (
            q.select(["delay", "X.avg", "X.sem"])
            .unpivot(index=["delay"])
            .select(
                [pl.lit(vals[i]).alias(n) for i, n in enumerate(columns)]
                + [pl.col("delay"), pl.col("variable"), pl.col("value")]
            )
        )
        queries.append(q)
    return Dataset(pl.concat(pl.collect_all(queries)), "delay", id_vars=columns)


if __name__ == "__main__":
    includes_phrases = ["nosample", "no sample", "REF", "reference"]
    exclude_phrases = ["-search", "-test"]
    paths = (
        pl.DataFrame(
            {"path": [str(p) for p in PROJECT_PATHS.raw_data.glob("**/*.txt")]}
        )
        .filter(
            pl.col("path").str.contains("|".join(includes_phrases))
            & ~pl.col("path").str.contains("|".join(exclude_phrases))
        )
        .unique()
        .select(
            pl.col("path")
            .map_elements(find_detector, return_dtype=pl.Utf8)
            .alias("detector"),
            pl.col("path")
            .map_elements(find_source, return_dtype=pl.Utf8)
            .alias("source"),
            pl.col("path").map_elements(find_date, return_dtype=pl.Date).alias("date"),
            pl.lit(293.0).alias("temperature"),
            pl.col("path")
            .map_elements(find_description, return_dtype=pl.Utf8)
            .alias("description"),
            pl.col("path"),
        )
    )

    data = (
        import_data(paths)
        .with_columns(
            col("path").map_elements(lambda s: Path(s).name, return_dtype=pl.Utf8)
        )
        .rename({"path": "filename"})
    )
    DATAFILES["waveforms"].write(data)
