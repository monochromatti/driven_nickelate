import matplotlib.pyplot as plt
import numpy as np
import xrayutilities.materials.elements as e
from xrayutilities import wavelength
from xrayutilities.materials import Crystal, SGLattice
from xrayutilities.simpack import DynamicalModel, FitModel, Layer, LayerStack

from driven_nickelate.config import paths as PROJECT_PATHS


def to_distance(x):
    lda = wavelength("CuKa1") / 10  # [nm], Cu K-alpha
    return lda / (2 * np.sin(np.deg2rad(x) / 2)) * 2


LSAT = Crystal(
    "LSAT",
    SGLattice(
        221,
        3.868,
        atoms=[e.La, e.Al, e.Sr, e.Ta, e.Al, e.O],
        pos=["1a", "1b", "1a", "1b", "1b", "3c"],
        occ=[0.3, 0.3, 0.7, 0.35, 0.35, 1],
    ),
)

NNO = Crystal(
    "NNO",
    SGLattice(
        123,
        3.868,
        3.815,
        atoms=[e.Nd, e.Ni, e.O, e.O],
        pos=["1a", "1d", "1c", "2e"],
        occ=[1, 1, 1, 1],
    ),
)

STO = Crystal(
    "STO",
    SGLattice(
        123,
        3.868,
        3.905,
        atoms=[e.Sr, e.Ti, e.O, e.O],
        pos=["1a", "1d", "1c", "2e"],
        occ=[1, 1, 1, 1],
    ),
)

sub = Layer(LSAT, float("inf"))
lay1 = Layer(NNO, 30)
lay2 = Layer(STO, 5)
ls = LayerStack("sample", sub + lay1 + lay2)
md = DynamicalModel(
    ls,
    energy="CuKa1",
    resolution_width=0.001,
    I0=1e8,
    background=0.25,
)
ai = np.linspace(20, 30, 10000)
fitmdyn = FitModel(md)
fitmdyn.lmodel.set_hkl((0, 0, 2))

p = fitmdyn.make_params()
p.add_many(
    ("I0", 4e7, True, 1e4, 2e8),
    ("background", 3, True, 0.1, 20),
    ("resolution_width", 0.003, False, 0.00001, 0.1),
    ("LSAT_a", 3.868, False, 3.7, 4.0),
    ("STO_a", 3.868, False, None, None, "LSAT_a"),
    ("STO_c", 3.905, True, 3.8, 4.0),
    ("STO_thickness", 19, True, 18, 21),
    ("NNO_a", 3.868, False, None, None, "LSAT_a"),
    ("NNO_c", 3.76, True, 3.6, 4.0),
    ("NNO_thickness", 120, True, 100, 130),
)

data = np.genfromtxt("F21062_XRD.txt", delimiter=",", skip_header=354)
theta = data[:, 0] / 2
intensity = data[:, 1]

res = fitmdyn.fit(intensity, p, theta)

fig: plt.Figure = plt.figure(figsize=(3.4, 3.4))
ax: plt.Axes = fig.add_subplot(111)

ax.text(
    0.05,
    0.95,
    f"$c = {res.params['NNO_c'].value / 10:.3f}$ nm\n$d = {res.params['NNO_thickness'].value / 10:.1f}$ nm",
    ha="left",
    va="top",
    transform=ax.transAxes,
    linespacing=2.0,
)

ax.semilogy(to_distance(2 * theta), intensity)
ax.semilogy(to_distance(2 * theta), res.best_fit, "k-", label="Fit", alpha=0.8)
ax.set(xlabel=r"$c$-axis distance (nm)", ylabel="Intensity (cps)", xlim=(0.34, 0.43))
fig.savefig(PROJECT_PATHS.figures / "characterization/xrd/film.pdf")
plt.show()
