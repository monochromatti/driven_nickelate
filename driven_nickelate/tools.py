import numpy as np
import polars as pl
from polars_complex import ccol
from polars_dataset import Dataset


def pl_fft(df, xname, id_vars=None, rfft=True):
    """
    Compute the Fast Fourier Transform (FFT) of the given DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        xname (str): The name of the column representing the x-axis values.
        id_vars (list, optional): List of column names to use as grouping variables. Defaults to None.
        real_valued (bool, optional): Whether the input data is real-valued. Defaults to True, and uses `rfft`.

    Returns:
        pl.DataFrame: The DataFrame containing the FFT results.
    """

    def fftfreq(df):
        if rfft:
            return np.fft.rfftfreq(
                len(df[xname]),
                abs(df[xname][1] - df[xname][0]),
            )
        else:
            return np.fft.fftfreq(
                len(df[xname]),
                abs(df[xname][1] - df[xname][0]),
            )

    def varname_iter(fft_dict, value_vars):
        for name in value_vars:
            for component, operator in zip(("real", "imag"), (np.real, np.imag)):
                yield pl.Series(
                    f"{name}.{component}",
                    operator(fft_dict[name]),
                    dtype=df.schema[name],
                )

    id_vars = id_vars or []
    value_vars = [var for var in df.columns if var not in id_vars and var != xname]

    frames = []
    fft_transform = np.fft.rfft if rfft else np.fft.fft
    if not id_vars:
        fft_dict = {name: fft_transform(df[name].to_numpy()) for name in value_vars}
        frames.append(
            pl.DataFrame(
                (
                    pl.Series("freq", fftfreq(df)),
                    *varname_iter(fft_dict, value_vars),
                )
            )
        )
    else:
        for id_vals, group in df.group_by(*id_vars, maintain_order=True):
            if isinstance(id_vals, (float, int, str)):
                id_vals = [id_vals]
            fft_dict = {
                name: fft_transform(group[name].to_numpy()) for name in value_vars
            }
            frames.append(
                pl.DataFrame(
                    (
                        pl.Series("freq", fftfreq(group)),
                        *varname_iter(fft_dict, value_vars),
                    )
                )
                .with_columns(
                    pl.lit(value).cast(df.schema[name]).alias(name)
                    for name, value in zip(id_vars, list(id_vals))
                )
                .select(
                    *(pl.col(name) for name in id_vars),
                    pl.col("freq"),
                    pl.all().exclude("freq", *id_vars),
                )
            )
    return pl.concat(frames)


def zero_quadrature(s: pl.Series):
    real, imag = s.struct[0], s.struct[1]
    pha = np.arange(-1.571, 1.571, 1e-3)
    imag_outer = np.outer(real, np.sin(pha)) + np.outer(imag, np.cos(pha))
    pha_opt = pha[np.sum(np.abs(imag_outer) ** 2, axis=0).argmin()]
    return real * np.cos(pha_opt) - imag * np.sin(pha_opt)


def create_dataset(
    paths: pl.DataFrame,
    column_names: list[str],
    index: str,
    lockin_schema: dict[str, tuple[str, str] | str],
    id_schema: dict | None = None,
    **kwargs,
) -> Dataset:
    """Create a Dataset from a list of data files.

    Parameters
    ----------
    paths : pl.DataFrame
        A DataFrame with at last one column, named "path", containing the paths to the files.
    column_names : list[str]
        A list of column names to use for the data. Order should match the order of columns in the data files.
    index : str
    lockin_schema : dict[tuple[str] | str, str]
        Data files may contain two channels (X and Y). If so, the relative phase
        will be adjusted to maximize the amplitude of the X channel, and only X will be
        retained. The dictionary entry of this data has the structure
            - key (str): new name for retained column
            - value (tuple[str]): 2-tuple of names of X and Y channel, respectively.
        If only one single channel, the dictionary entry has the structure
            - key (str): new name for column
            - value (str): name of column in data file
    id_schema : dict[pl.DataType], optional
        Polars data type of the columns of `path` representing id (not index) parameters.

    Returns
    -------
    Dataset
        A Dataset object.
    """

    kwargs = {
        "separator": kwargs.pop("separator", "\t"),
        "has_header": kwargs.pop("has_header", False),
        "comment_prefix": kwargs.pop("comment_prefix", "#"),
        **kwargs,
    }
    pair_dict = {k: v for k, v in lockin_schema.items() if isinstance(v, tuple)}
    lone_dict = {k: v for k, v in lockin_schema.items() if isinstance(v, str)}

    lockin_exprs = [
        ccol(x, y).alias(name).map_batches(zero_quadrature)
        for name, (x, y) in pair_dict.items()
    ]
    lockin_exprs += [pl.col(col).alias(name) for name, col in lone_dict.items()]

    frames = []
    if not id_schema:
        id_schema = paths.schema
        del id_schema["path"]

    for *idvals, path in paths.iter_rows():
        id_exprs = [
            pl.lit(val).cast(dtype).alias(name)
            for val, (name, dtype) in zip(idvals, id_schema.items())
        ]
        df = (
            pl.read_csv(path, new_columns=column_names, **kwargs)
            .with_columns(*id_exprs, index, *lockin_exprs)
            .select(
                *(pl.col(name) for _, (name, _) in zip(idvals, id_schema.items())),
                pl.col(index),
                *(pl.col(name) for name in lockin_schema.keys()),
            )
            .sort(index, *id_schema.keys())
        )
        frames.append(df)
    return Dataset(frames, index, list(id_schema))
