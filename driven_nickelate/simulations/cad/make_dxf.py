import argparse
import json

import ezdxf
import numpy as np
from ezdxf import recover
from ezdxf.addons.drawing import matplotlib

parser = argparse.ArgumentParser(
    description="Creates .dxf of electric split-ring resonator array"
)
parser.add_argument(
    "params_file", type=str, help="JSON file with structural parameters"
)
parser.add_argument(
    "-png", "--store", action="store_true", help="Save a .png output of the design"
)
parser.add_argument("--filename", help="{filename}.dxf (and .png)")
args = parser.parse_args()

if __name__ == "__main__":
    # --- Setup ---
    with open(args.params_file) as json_file:
        params_dict = json.load(json_file)

    # Unit cell
    Lx = params_dict["Width"]
    Ly = params_dict["Height"]
    lw = params_dict["Line thickness"]
    R = params_dict["Large fillet radius"]
    r = params_dict["Small fillet radius"]
    g = params_dict["Gap size"]
    l = params_dict["Plate width"]

    # Metasurface
    dx = params_dict["Horizontal space"]
    dy = params_dict["Vertical space"]

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # --- Structure polylines ---

    # Outer perimeter
    b = 1 - np.sqrt(2)  # Bulge
    xo, yo = Lx / 2, Ly / 2
    outer_polyline = [
        [-xo, (yo - R), b],
        [-(xo - R), yo, 0],
        [xo - R, yo, b],
        [xo, yo - R, 0],
        [xo, -(yo - R), b],
        [xo - R, -yo, 0],
        [-(xo - R), -yo, b],
        [-xo, -(yo - R), 0],
        [-xo, (yo - R), 0],
    ]

    # Internal perimeter
    Ri = R * (
        1 - 2 * np.sqrt(2) * lw / np.sqrt((Lx / 2) ** 2 + (Ly / 2) ** 2)
    )  # Reduced radius
    xi = Lx / 2 - lw
    yi = Ly / 2 - lw
    inner_polyline = [
        [-xi, yi - Ri, b],
        [-(xi - Ri), yi, 0],
        [-(Ri + lw / 2), yi, b],
        [-lw / 2, yi - Ri, 0],
        [-lw / 2, g / 2 + lw + r, b],
        [-lw / 2 - r, g / 2 + lw, 0],
        [-(l / 2 - r), g / 2 + lw, -b],
        [-l / 2, g / 2 + lw - r, 0],
        [-l / 2, g / 2, 0],
        [l / 2, g / 2, 0],
        [l / 2, g / 2 + lw - r, -b],
        [l / 2 - r, g / 2 + lw, 0],
        [lw / 2 + r, g / 2 + lw, b],
        [lw / 2, g / 2 + lw + r, 0],
        [lw / 2, yi - Ri, b],
        [lw / 2 + Ri, yi, 0],
        [xi - Ri, yi, b],
        [xi, yi - Ri, 0],
        [xi, -(yi - Ri), b],
        [xi - Ri, -yi, 0],
        [lw / 2 + Ri, -yi, b],
        [lw / 2, -(yi - Ri), 0],
        [lw / 2, -(g / 2 + lw + r), b],
        [lw / 2 + r, -(g / 2 + lw), 0],
        [(l / 2 - r), -(g / 2 + lw), -b],
        [l / 2, -(g / 2 + lw - r), 0],
        [l / 2, -g / 2, 0],
        [-l / 2, -g / 2, 0],
        [-l / 2, -(g / 2 + lw - r), -b],
        [-(l / 2 - r), -(g / 2 + lw), 0],
        [-(lw / 2 + r), -(g / 2 + lw), b],
        [-lw / 2, -(g / 2 + lw + r), 0],
        [-lw / 2, -(yi - Ri), b],
        [-(lw / 2 + Ri), -yi, 0],
        [-(xi - Ri), -yi, b],
        [-(xi - Ri), -yi, b],
        [-xi, -(yi - Ri), 0],
        [-xi, (yi - Ri), 0],
    ]

    # Create block and layer
    struc_layer = doc.layers.new("structure")

    # Add polylines on the 'structure' layer
    msp.add_lwpolyline(outer_polyline, format="xyb", dxfattribs={"layer": "structure"})
    msp.add_lwpolyline(inner_polyline, format="xyb", dxfattribs={"layer": "structure"})

    gap_layer = doc.layers.new("gap")
    gap_coords = [
        (-l / 2, g / 2),
        (l / 2, g / 2),
        (l / 2, -g / 2),
        (-l / 2, -g / 2),
        (-l / 2, g / 2),
    ]
    msp.add_lwpolyline(gap_coords, format="xy", dxfattribs={"layer": "gap"})

    uc_layer = doc.layers.new("unit_cell")
    perimeter_coords = [
        (-(Lx + 2 * dx) / 2, -(Ly + 2 * dy) / 2),
        (+(Lx + 2 * dx) / 2, -(Ly + 2 * dy) / 2),
        (+(Lx + 2 * dx) / 2, +(Ly + 2 * dy) / 2),
        (-(Lx + 2 * dx) / 2, +(Ly + 2 * dy) / 2),
        (-(Lx + 2 * dx) / 2, -(Ly + 2 * dy) / 2),
    ]
    msp.add_lwpolyline(perimeter_coords, dxfattribs={"layer": "unit_cell"})
    msp.add_lwpolyline(perimeter_coords, dxfattribs={"layer": "structure"})

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
