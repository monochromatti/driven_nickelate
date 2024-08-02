import numpy as np
import polars as pl


def complex_struct(real: pl.Expr, imag: pl.Expr):
    return pl.struct(real.alias("real"), imag.alias("imag"))


@pl.api.register_expr_namespace("complex")
class ComplexArithmetic:
    def __init__(self, expr: pl.Expr):
        self._expr = expr
        self._names = expr.meta.root_names()
        self._name = expr.meta.output_name()

    def cformat(self, name: str):
        return name if name.endswith("[c]") else name + "[c]"

    def struct_repr(self, name: str = None):
        match self._names:
            case [str()]:
                stem = self._names[0]
                if name:
                    new_stem = name.replace("[c]", "")
                else:
                    new_stem = stem
                return pl.struct(
                    pl.col(f"{stem}.real").alias("real"),
                    pl.col(f"{stem}.imag").alias("imag"),
                ).alias(f"{new_stem}[c]")
            case [str(), str()]:
                real_str = self._names[0].replace(".real", "")
                imag_str = self._names[1].replace(".imag", "")
                stem = real_str if (real_str == imag_str) else name
                return pl.struct(
                    pl.col(self._names[0]).alias("real"),
                    pl.col(self._names[1]).alias("imag"),
                ).alias(f"{stem}[c]")
            case _:
                raise ValueError(
                    "Invalid expression used in `struct_repr`. Must be struct of length 2."
                )

    def nest(self, name: str = None):
        if name:
            name = self.cformat(name)
        return self.struct_repr(name)

    def rename(self, name):
        if not name.endswith("[c]"):
            name = name + "[c]"
        return self._expr.alias(name)

    def explode(self, struct_expr):
        return struct_expr.struct[0], struct_expr.struct[1]

    def real(self):
        return self._expr.struct[0]

    def imag(self):
        return self._expr.struct[1]

    def phase(self):
        return pl.arctan2(self._expr.struct[0], self._expr.struct[1])

    def unwrap_phase(self):
        diff = self._expr - self._expr.shift()
        if_lower = pl.when(diff < -3.14159).then(diff + 2 * 3.14159)
        if_higher = pl.when(diff > 3.14159).then(diff - 2 * 3.14159)
        corrected_diff = (
            if_higher.otherwise(if_lower.otherwise(diff)).fill_null(0).cumsum()
        )
        return self._expr.head(1).cast(pl.Float64) + corrected_diff

    def zero_quadrature(self):
        name = self._expr.meta.name.replace("[c]", "")
        real, imag = self.explode(self._expr)
        pha = np.arange(-1.571, 1.571, 1e-3)
        imag_outer = np.outer(real, np.sin(pha)) + np.outer(imag, np.cos(pha))
        pha_opt = pha[np.linalg.norm(imag_outer, axis=0).argmin()]
        return (real * np.cos(pha_opt) - imag * np.sin(pha_opt)).alias(name)

    def modulus(self):
        real, imag = self.explode(self._expr)
        return (real.pow(2) + imag.pow(2)).sqrt()

    def conj(self):
        real, imag = self.explode(self._expr)
        return complex_struct(real, -imag).alias(self._name)

    def quotient(self, expr):
        real1, imag1 = self.explode(self._expr.name.suffix_fields(".1"))
        real2, imag2 = self.explode(expr.name.suffix_fields(".2"))
        real = (real1 * real2 + imag1 * imag2) / (real2.pow(2) + imag2.pow(2))
        imag = (real2 * imag1 - real1 * imag2) / (real2.pow(2) + imag2.pow(2))
        return complex_struct(real, imag).alias(self._name)

    def divide(self, expr):
        return self.quotient(expr)

    def ratio(self, expr):
        return self.quotient(expr)

    def multiply(self, expr):
        a, b = self.explode(self._expr)
        c, d = self.explode(expr)
        real = a * c - b * d
        imag = a * d + b * c
        return complex_struct(real, imag).alias(self._name)

    def subtract(self, expr):
        real, imag = self._explode(self._expr - expr)
        return complex_struct(real, imag).alias(self.name)

    def difference(self, expr):
        return self.subtract(expr)

    def relative_difference(self, expr):
        real, imag = self.explode(self.quotient(expr))
        return complex_struct(real - 1, imag)


@pl.api.register_dataframe_namespace("complex")
class ComplexFrame:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def unnest(self, names: str | list[str] = None):
        df = self._df
        if names is None:
            names = [name for name in df.columns if name.endswith("[c]")]
        elif not isinstance(names, list):
            names = [names]
        stems = [name.replace("[c]", "") for name in names]
        return df.select(
            pl.all().exclude(names),
            *[
                pl.col(name).struct.rename_fields([f"{stem}.real", f"{stem}.imag"])
                for (name, stem) in zip(names, stems)
            ],
        ).unnest(names)

    def nest(self, varnames: str | list[str] = None):
        if varnames is None:
            varnames = [
                name.replace(".real", "")
                for name in self._df.columns
                if name.endswith(".real")
                and name.replace(".real", ".imag") in self._df.columns
            ]
            varnames = list(set(varnames))
        elif isinstance(varnames, str):
            varnames = [varnames]

        complex_columns = self._df.select(
            pl.col(f"{varname}.{comp}")
            for varname in varnames
            for comp in ("real", "imag")
        ).columns

        for varname in varnames:
            if f"{varname}.real" not in complex_columns:
                raise ValueError(f"Column {varname}.real missing.")
            if f"{varname}.imag" not in complex_columns:
                raise ValueError(f"Column {varname}.imag missing.")

        return self._df.select(
            pl.all().exclude(complex_columns),
            *[
                pl.struct(
                    pl.col(f"{var}.real").alias("real"),
                    pl.col(f"{var}.imag").alias("imag"),
                ).alias(f"{var}[c]")
                for var in varnames
            ],
        )

    def struct(self, *args, **kwargs):
        return self.nest(*args, **kwargs)


def pl_fft(df, xname, id_vars=None):
    def fftfreq(df):
        return np.fft.rfftfreq(
            len(df[xname]),
            abs(df[xname][1] - df[xname][0]),
        )

    value_vars = [var for var in df.columns if var not in id_vars and var != xname]

    frames = []
    if not id_vars:

        def varname_iter(fft_dict, value_vars):
            for name in value_vars:
                for component, operator in zip(("real", "imag"), (np.real, np.imag)):
                    yield pl.Series(f"{name}.{component}", operator(fft_dict[name]))

        fft_dict = {name: np.fft.rfft(df[name].to_numpy()) for name in value_vars}
        frames.append(
            pl.DataFrame(
                (
                    pl.Series("freq", fftfreq(df)),
                    *varname_iter(fft_dict, value_vars),
                )
            )
        )
    else:
        for id_vals, group in df.group_by(*id_vars):
            if isinstance(id_vals, (float, int, str)):
                id_vals = [id_vals]
            fft_dict = {
                name: np.fft.rfft(group[name].to_numpy()) for name in value_vars
            }
            frames.append(
                pl.DataFrame(
                    (
                        pl.Series("freq", fftfreq(group)),
                        *varname_iter(fft_dict, value_vars),
                    )
                )
                .with_columns(
                    pl.lit(value).alias(name)
                    for name, value in zip(id_vars, list(id_vals))
                )
                .select(
                    *(pl.col(name) for name in id_vars),
                    pl.col("freq"),
                    pl.all().exclude("freq", *id_vars),
                )
            )
    return pl.concat(frames)
