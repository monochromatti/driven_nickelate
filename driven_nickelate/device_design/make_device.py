import argparse
import json
from itertools import product

import ezdxf
import numpy as np
from ezdxf import recover
from ezdxf.addons.drawing import matplotlib
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Creates .dxf of electric split-ring resonator array"
)
parser.add_argument(
    "params_file", type=str, help="JSON file with structural parameters"
)
parser.add_argument(
    "-png", "--store", action="store_true", help="Save a .png output of the design"
)
parser.add_argument(
    "-uc", "--unitcell", action="store_true", help="Output only one unit cell"
)
parser.add_argument("--filename", help="{filename}.dxf (and .png)")
args = parser.parse_args()


def resonator_polylines(params_dict):
    sample_width = params_dict["Width"]
    sample_height = params_dict["Height"]
    lw = params_dict["Line thickness"]
    rmaj = params_dict["Large fillet radius"]
    rmin = params_dict["Small fillet radius"]
    gs = params_dict["Gap size"]
    pw = params_dict["Plate width"]

    # Outer perimeter
    b = 1 - np.sqrt(2)  # Bulge
    xo, yo = sample_width / 2, sample_height / 2
    outer_polyline = [
        [-xo, (yo - rmaj), b],
        [-(xo - rmaj), yo, 0],
        [xo - rmaj, yo, b],
        [xo, yo - rmaj, 0],
        [xo, -(yo - rmaj), b],
        [xo - rmaj, -yo, 0],
        [-(xo - rmaj), -yo, b],
        [-xo, -(yo - rmaj), 0],
        [-xo, (yo - rmaj), 0],
    ]

    # Internal perimeter
    Ri = rmaj * (
        1
        - 2
        * np.sqrt(2)
        * lw
        / np.sqrt((sample_width / 2) ** 2 + (sample_height / 2) ** 2)
    )  # Reduced radius
    xi = sample_width / 2 - lw
    yi = sample_height / 2 - lw
    inner_polyline = [
        [-xi, yi - Ri, b],
        [-(xi - Ri), yi, 0],
        [-(Ri + lw / 2), yi, b],
        [-lw / 2, yi - Ri, 0],
        [-lw / 2, gs / 2 + lw + rmin, b],
        [-lw / 2 - rmin, gs / 2 + lw, 0],
        [-(pw / 2 - rmin), gs / 2 + lw, -b],
        [-pw / 2, gs / 2 + lw - rmin, 0],
        [-pw / 2, gs / 2, 0],
        [pw / 2, gs / 2, 0],
        [pw / 2, gs / 2 + lw - rmin, -b],
        [pw / 2 - rmin, gs / 2 + lw, 0],
        [lw / 2 + rmin, gs / 2 + lw, b],
        [lw / 2, gs / 2 + lw + rmin, 0],
        [lw / 2, yi - Ri, b],
        [lw / 2 + Ri, yi, 0],
        [xi - Ri, yi, b],
        [xi, yi - Ri, 0],
        [xi, -(yi - Ri), b],
        [xi - Ri, -yi, 0],
        [lw / 2 + Ri, -yi, b],
        [lw / 2, -(yi - Ri), 0],
        [lw / 2, -(gs / 2 + lw + rmin), b],
        [lw / 2 + rmin, -(gs / 2 + lw), 0],
        [(pw / 2 - rmin), -(gs / 2 + lw), -b],
        [pw / 2, -(gs / 2 + lw - rmin), 0],
        [pw / 2, -gs / 2, 0],
        [-pw / 2, -gs / 2, 0],
        [-pw / 2, -(gs / 2 + lw - rmin), -b],
        [-(pw / 2 - rmin), -(gs / 2 + lw), 0],
        [-(lw / 2 + rmin), -(gs / 2 + lw), b],
        [-lw / 2, -(gs / 2 + lw + rmin), 0],
        [-lw / 2, -(yi - Ri), b],
        [-(lw / 2 + Ri), -yi, 0],
        [-(xi - Ri), -yi, b],
        [-(xi - Ri), -yi, b],
        [-xi, -(yi - Ri), 0],
        [-xi, (yi - Ri), 0],
    ]
    return inner_polyline, outer_polyline


def rectangle(x, y=None):
    if y is None:
        y = x
    return [
        (-x / 2, -y / 2),
        (+x / 2, -y / 2),
        (+x / 2, +y / 2),
        (-x / 2, +y / 2),
        (-x / 2, -y / 2),
    ]


if __name__ == "__main__":
    # --- Setup ---
    with open(args.params_file) as json_file:
        params_dict = json.load(json_file)

    # Metasurface
    sample_width = params_dict["Width"]
    sample_height = params_dict["Height"]
    substrate_size = params_dict["Substrate size"]
    radius_metasurface = params_dict["Metasurface radius"]
    ang_ms = params_dict["Metasurface angle"]
    wspace = params_dict["Horizontal space"]
    hspace = params_dict["Vertical space"]

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    if args.unitcell:
        uc_layer = doc.layers.new("unit_cell")
        uc_layer.color = 7
        coords = rectangle(sample_width + wspace, sample_height + hspace)
        msp.add_lwpolyline(coords, dxfattribs={"layer": "unit_cell"})

        gap_layer = doc.layers.new("gap")
        gap_layer.color = 4
        coords = rectangle(params_dict["Plate width"], params_dict["Gap size"])
        msp.add_lwpolyline(coords, dxfattribs={"layer": "gap"})

        structure_layer = doc.layers.new("structure")
        structure_layer.color = 2
        inner_polyline, outer_polyline = resonator_polylines(params_dict)
        msp.add_lwpolyline(
            inner_polyline, format="xyb", dxfattribs={"layer": "structure"}
        )
        msp.add_lwpolyline(
            outer_polyline, format="xyb", dxfattribs={"layer": "structure"}
        )
    else:
        # Create block and layer
        struc = doc.blocks.new(name="structure")
        struc_layer = doc.layers.new("structure")
        struc_layer.color = 2

        inner_polyline, outer_polyline = resonator_polylines(params_dict)

        # Add polylines to 'structure' block on the 'structure' layer
        struc.add_lwpolyline(
            outer_polyline, format="xyb", dxfattribs={"layer": "structure"}
        )
        struc.add_lwpolyline(
            inner_polyline, format="xyb", dxfattribs={"layer": "structure"}
        )
        # Combine 'structure' blocks in to a larger block 'array'
        array = doc.blocks.new(name="array")
        x = np.arange(-substrate_size / 2, substrate_size / 2, sample_width + wspace)
        y = np.arange(-substrate_size / 2, substrate_size / 2, sample_height + hspace)
        coords = product(x, y)
        for p in tqdm(coords, total=len(x) ** 2):
            if p[0] ** 2 + p[1] ** 2 < radius_metasurface**2:
                array.add_blockref("structure", p, dxfattribs={"layer": "structure"})

        # Add 'array' as INSERT to modelspace
        msp.add_blockref(
            "array", (0, 0), dxfattribs={"rotation": ang_ms, "layer": "structure"}
        )

        label_layer = doc.layers.new("label")
        label_layer.color = 9
        txtpos = (
            -(1.414 * radius_metasurface + substrate_size) / 4,
            (1.414 * radius_metasurface + substrate_size) / 4,
        )
        msp.add_text(
            params_dict["Label"],
            dxfattribs={
                "layer": "label",
                "style": "OpenSansCondensed-Bold",
                "height": 100,
            },
        ).set_pos(txtpos, align="CENTER")

        # Substrate outline
        substrate_layer = doc.layers.new("substrate")
        substrate_layer.color = 7
        msp.add_lwpolyline(rectangle(substrate_size), dxfattribs={"layer": "substrate"})

    # Save to .dxf (and .png)
    if args.filename is None:
        filename = "output"
    else:
        filename = args.filename
    doc.saveas("{}.dxf".format(filename))

    if args.store:
        doc, auditor = recover.readfile("{}.dxf".format(filename))
        if not auditor.has_errors:
            matplotlib.qsave(doc.modelspace(), "{}.png".format(filename))
