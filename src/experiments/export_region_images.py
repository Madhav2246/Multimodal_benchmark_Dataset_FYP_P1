"""Export one set of viewable images per region (for figures and quick browsing).

Writes to paper/figures/regions/:
    <region>_optical_100m.png   native-resolution WAC crop (data/intermediate/<region>/optical/wac_pilot.npy)
    <region>_layers.png         optical | elevation | slope | hillshade | thermal on the common grid

Plain PNGs (no georeferencing). Layers use the masks (invalid pixels are left blank).

Usage
-----
    python -m src.experiments.export_region_images
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..pipeline.config import list_regions, load_region

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "regions"


def _stretch(a, lo=2, hi=98):
    a = a.astype(np.float64)
    p0, p1 = np.nanpercentile(a, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-9), 0, 1)


def export_region(rid):
    r = load_region(rid)
    i = r["intermediate_dir"]
    native = np.load(i / "optical" / "wac_pilot.npy")
    plt.imsave(OUT / f"{rid}_optical_100m.png", _stretch(native), cmap="gray")

    opt = np.load(i / "optical" / "wac_common_grid.npy")
    ocov = np.load(i / "optical" / "wac_common_coverage.npy") >= 0.99
    dem = np.load(i / "dem" / "lola_dem_common_grid.npy")
    dok = np.load(i / "dem" / "lola_dem_common_valid_mask.npy").astype(bool)
    slope = np.load(i / "dem" / "lola_common_slope_deg.npy")
    hill = np.load(i / "dem" / "lola_common_hillshade.npy")
    th = np.load(i / "thermal" / "diviner_tbol_sam_common_grid.npy")
    tok = np.load(i / "thermal" / "diviner_tbol_sam_common_valid_mask.npy").astype(bool)

    panels = [
        ("Optical (WAC)", np.where(ocov, opt, np.nan), "gray", None),
        ("Elevation (km)", np.where(dok, dem, np.nan), "terrain", "km"),
        ("Slope (deg)", np.where(dok, slope, np.nan), "magma", "deg"),
        ("Hillshade", np.where(dok, hill, np.nan), "gray", None),
        ("Thermal, midnight (K)", np.where(tok, th, np.nan), "inferno", "K"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    for ax, (title, arr, cmap, unit) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if unit:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{rid}   {r['lat_min']}..{r['lat_max']} N, {r['lon_min']}..{r['lon_max']} E", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{rid}_layers.png", dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for rid in list_regions():
        export_region(rid)
        print(rid, flush=True)


if __name__ == "__main__":
    main()
