#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive KMOS astrometry correction (manual fallback mode)
Adapted for concatenation folder structure.

Purpose
-------
This script fixes only those IFUs that were not corrected by the
automatic astrometric pipeline.

Workflow
--------
1. Loop over concatenations (con1, con2, …) and automatically discover
   all OB subdirectories inside them.
2. For each OB, identify IFU cubes that do not yet have a corresponding
   *_astrocorr.fits file.
3. For each failed IFU:
   - build a collapsed image
   - select the local catalogue around the IFU reference position
   - display the IFU image and local catalogue stars with Ks labels
   - left click once on the IFU star
   - left click once on the matching catalogue star
   - or right click to reject the IFU if there is no reliable coincidence
4. Save:
   - corrected cube into corrected_fits_new
   - corrected collapsed file into corrected_fits_new/collapsed
   - rejection marker if needed

Notes
-----
- Uses the same directory philosophy as the automatic pipeline.
- Only processes files that are still uncorrected.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import warnings
from pathlib import Path
from typing import Optional

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS

# ============================================================
# CONFIGURATION – Edit only these variables
# ============================================================
pointing = "PX"                     # pointing name
CONS = ["con3"]     # concatenations to process

INPUT_PATTERN = "COMBINE_SKY_TWEAK_*.fits"

# Catalogue path (adjust to your real path)
catalogue_file =  Path("/home/jmgavilan/Desktop/KMOS/GNS_VIRAC_cat/FINAL_EAST_CENTRAL_WEST_VIRAC.fits")

# Reduced search radius so that the IFU fills more of the field
CATALOGUE_RADIUS_ARCSEC = 5.0
REFERENCE_K_LIMIT = 14.5

# Template for the sky_tweak directory of each OB.
# Must match the automatic pipeline structure.
BASE_DIR_TEMPLATE = (
    # "/home/data/KMOS/PILOT/reduced/P113/{pointing}/{con_name}/{ob_name}/sky_tweak/" # Server Path
    "/home/jmgavilan/Desktop/PX/{pointing}/{con_name}/{ob_name}/sky_tweak/"  # Local test     # Local test
)

# "/home/data/KMOS/PILOT/reduced/P113/{pointing}/{con_name}/{ob_name}/sky_tweak/" # Server Path

BASE_DIR = None   # will be updated for each OB

# ============================================================
# GET ARM NUMBER
# ============================================================
def get_arm_label(hdr, fits_name: str) -> str:
    """
    Return a short ARM label like 'ARM1' for the plot title.

    Strategy
    --------
    1. Search the header KEYS for a token matching 'ARM<digits>'
       (this is the original user method, e.g. 'HIERARCH ESO INS ARM1 ...').
    2. Search the VALUES of the standard KMOS keywords
       ('ARM', 'ESO INS OPTI3 NAME', HIERARCH variant).
    3. Search ALL header VALUES for the same pattern.
    4. Search the file name for a token like 'ARM1'.
    5. Fall back to 'ARM?' so the title never breaks.
    """
    pattern = re.compile(r"ARM(\d+)", re.IGNORECASE)

    # 1) Header keys (original user method)
    for key in hdr:
        m = pattern.search(key)
        if m:
            return f"ARM{m.group(1)}"

    # 2) Standard KMOS keyword values
    for key in ("ARM", "ESO INS OPTI3 NAME", "HIERARCH ESO INS OPTI3 NAME"):
        if key in hdr:
            m = pattern.search(str(hdr[key]))
            if m:
                return f"ARM{m.group(1)}"

    # 3) Any header value
    for key in hdr:
        m = pattern.search(str(hdr[key]))
        if m:
            return f"ARM{m.group(1)}"

    # 4) File name
    m = pattern.search(Path(fits_name).stem)
    if m:
        return f"ARM{m.group(1)}"

    # 5) Fallback
    return "ARM?"


# ============================================================
# PATH HANDLING
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_ob_paths(ob_name: str) -> dict:
    """
    Same structure as the automatic pipeline.
    - IFU_ROOT points to BASE_DIR (where COMBINE_SKY_TWEAK_*.fits are)
    - Results are saved inside res_{ob_name}
    """
    ob_dir = BASE_DIR / f"res_{ob_name}"

    paths = {
        "NAME": ob_name,
        "OB_DIR": ob_dir,
        "IFU_ROOT": BASE_DIR,
        "COLLAPSED_OUTPUT": ob_dir / f"{ob_name}_collapsed_new",
        "CORRECTED_CUBE_OUTPUT": ob_dir / "corrected_fits_new",
        "CORRECTED_COLLAPSED_OUTPUT": ob_dir / "corrected_fits_new" / "collapsed",
        "CHECK_PLOT_OUTPUT": ob_dir / "corrected_fits_new" / "visual_checks",
        "DIAGNOSTIC_PLOT_OUTPUT": ob_dir / "corrected_fits_new" / "diagnostic_plots",
        "FINAL_CHECK_PLOT_OUTPUT": ob_dir / "corrected_fits_new" / "final_check_plots",
    }

    for key in [
        "COLLAPSED_OUTPUT",
        "CORRECTED_CUBE_OUTPUT",
        "CORRECTED_COLLAPSED_OUTPUT",
        "CHECK_PLOT_OUTPUT",
        "DIAGNOSTIC_PLOT_OUTPUT",
        "FINAL_CHECK_PLOT_OUTPUT",
    ]:
        ensure_dir(paths[key])

    return paths


def get_uncorrected_ifus(paths: dict) -> list[Path]:
    """
    Returns only those IFUs that do not yet have a correction (*_astrocorr.fits)
    """
    all_fits = sorted(paths["IFU_ROOT"].glob(INPUT_PATTERN))
    corrected_dir = paths["CORRECTED_CUBE_OUTPUT"]

    corrected_names = {
        f.name.replace("_astrocorr.fits", "")
        for f in corrected_dir.glob("*_astrocorr.fits")
    }

    to_process = []
    for f in all_fits:
        if f.stem not in corrected_names:
            to_process.append(f)

    print("\nAll IFU cubes found:")
    for f in all_fits:
        print(f"  {f.name}")
    return to_process


# ============================================================
# CATALOGUE LOADING
# ============================================================
def load_catalogue(path: Path) -> pd.DataFrame:
    """
    Read catalogue from a FITS binary table.
    Expected columns: RA, Dec, and a magnitude (preferentially K, but can be named Kmag, KMAG, etc.)
    """
    with fits.open(path) as hdul:
        # Find first table HDU
        table_hdu = None
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU) or isinstance(hdu, fits.TableHDU):
                table_hdu = hdu
                break
        if table_hdu is None:
            raise ValueError("No table HDU found in catalogue FITS file.")
        
        data = table_hdu.data
        columns = [col.name.lower() for col in table_hdu.columns]
        
        # Identify RA column
        ra_col = None
        for possible in ['ra', 'raj2000', 'ra_deg', 'alpha']:
            if possible in columns:
                ra_col = possible
                break
        if ra_col is None:
            raise ValueError(f"Cannot find RA column. Available columns: {columns}")
        
        # Identify Dec column
        dec_col = None
        for possible in ['dec', 'dej2000', 'dec_deg', 'delta']:
            if possible in columns:
                dec_col = possible
                break
        if dec_col is None:
            raise ValueError(f"Cannot find Dec column. Available columns: {columns}")
        
        # Identify magnitude column (prefer K)
        mag_col = None
        for possible in ['k', 'kmag', 'k_mag', 'mag_k', 'ks', 'ksmag']:
            if possible in columns:
                mag_col = possible
                break
        if mag_col is None:
            # fallback to any column containing 'mag'
            mag_candidates = [c for c in columns if 'mag' in c]
            if mag_candidates:
                mag_col = mag_candidates[0]  # take first
            else:
                raise ValueError(f"Cannot find magnitude column. Available columns: {columns}")
        
        # Build DataFrame with standard names
        df = pd.DataFrame({
            'ra': data[ra_col].astype(float),
            'dec': data[dec_col].astype(float),
            'K': data[mag_col].astype(float)
        })
        
        # Filter valid entries
        good = (
            np.isfinite(df['ra'].to_numpy()) &
            np.isfinite(df['dec'].to_numpy()) &
            np.isfinite(df['K'].to_numpy())
        )
        return df.loc[good].copy().reset_index(drop=True)


def select_local_catalogue(df_cat: pd.DataFrame, ref_coord: SkyCoord) -> pd.DataFrame:
    coords = SkyCoord(df_cat["ra"].values * u.deg, df_cat["dec"].values * u.deg)
    sep = coords.separation(ref_coord).arcsec
    mask = (sep <= CATALOGUE_RADIUS_ARCSEC) & (df_cat["K"] <= REFERENCE_K_LIMIT)
    df_local = df_cat.loc[mask].copy()
    df_local["sep"] = sep[mask]
    return df_local.sort_values(["K", "sep"]).reset_index(drop=True)


# ============================================================
# CUBE COLLAPSE
# ============================================================
def collapse_cube(
    cube: np.ndarray,
    wcs_spectral: WCS,
    edge_trim_fraction: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse the cube by trimming spectral edges and applying sigma-clip,
    exactly like the automatic pipeline. This makes the image scale
    (and therefore the colorbar) consistent between both scripts.
    """
    nz = cube.shape[0]
    trim = int(np.floor(edge_trim_fraction * nz))
    i0, i1 = trim, nz - trim
    if i1 <= i0 + 5:
        raise ValueError("Too many spectral channels removed by EDGE_TRIM_FRACTION.")

    wave_idx = np.arange(i0, i1)
    lambda_m = wcs_spectral.pixel_to_world(wave_idx)
    lambda_angstrom = lambda_m.to(u.AA)

    subcube = cube[i0:i1, :, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clipped = sigma_clip(subcube, sigma=2, axis=0)

    masked_cube = np.ma.masked_where(
        clipped.mask | (clipped.data < 0),
        clipped.data
    )
    image = np.trapezoid(
        masked_cube,
        x=lambda_angstrom.value,
        axis=0
    ).filled(0)

    return image, wave_idx


# ============================================================
# APPLY SHIFT
# ============================================================
def apply_shift(hdul: fits.HDUList, dra_deg: float, ddec_deg: float) -> fits.HDUList:
    hdul_new = fits.HDUList([h.copy() for h in hdul])
    hdr = hdul_new[1].header
    hdr["CRVAL1"] = float(hdr["CRVAL1"]) + dra_deg
    hdr["CRVAL2"] = float(hdr["CRVAL2"]) + ddec_deg
    hdr["HISTORY"] = "=================================================="
    hdr["HISTORY"] = "Manual astrometry correction applied"
    hdr["HISTORY"] = f"dRA  = {dra_deg * 3600.0:.3f} arcsec"
    hdr["HISTORY"] = f"dDec = {ddec_deg * 3600.0:.3f} arcsec"
    if len(hdul_new) > 1:
        hdul_new[1].header["EXTNAME"] = "DATA"
    if len(hdul_new) > 2:
        hdul_new[2].header["EXTNAME"] = "STAT"
    return hdul_new


def save_corrected_collapsed(
    image: np.ndarray,
    wcs: WCS,
    dra_deg: float,
    ddec_deg: float,
    output_path: Path,
) -> None:
    wcs_new = wcs.deepcopy()
    wcs_new.wcs.crval[0] += dra_deg
    wcs_new.wcs.crval[1] += ddec_deg
    hdu_primary = fits.PrimaryHDU()
    hdu_image = fits.ImageHDU(data=image, header=wcs_new.to_header())
    hdu_image.header["EXTNAME"] = "DATA"
    hdu_image.header["HISTORY"] = "=================================================="
    hdu_image.header["HISTORY"] = "Manual astrometry correction applied"
    hdu_image.header["HISTORY"] = f"dRA  = {dra_deg * 3600.0:.3f} arcsec"
    hdu_image.header["HISTORY"] = f"dDec = {ddec_deg * 3600.0:.3f} arcsec"
    hdul = fits.HDUList([hdu_primary, hdu_image])
    hdul.writeto(output_path, overwrite=True)


# ============================================================
# CHECK PLOT (saved, not interactive)
# ============================================================
def save_manual_check_plot(
    image: np.ndarray,
    wcs: WCS,
    df_cat: pd.DataFrame,
    clicked_ifu: tuple[float, float],
    clicked_cat: tuple[float, float],
    matched_index: int,
    output_path: Path,
    title_info: str = "",
) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(projection=wcs)
    finite = image[np.isfinite(image)]
    if finite.size > 0:
        vmin, vmax = np.nanpercentile(finite, [5, 99.5])
    else:
        vmin, vmax = None, None
    im = ax.imshow(image, origin="lower", cmap="hot", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
    cbar.set_label("Collapsed flux")
    
    coords = SkyCoord(df_cat["ra"].values * u.deg, df_cat["dec"].values * u.deg)
    x_cat, y_cat = wcs.world_to_pixel(coords)
    ax.scatter(x_cat, y_cat, color="cyan", marker="x", s=60)
    for i, (xx, yy, k) in enumerate(zip(x_cat, y_cat, df_cat["K"])):
        ax.text(
            xx + 0.5,
            yy + 0.5,
            f"{i}:{k:.1f}",
            color="cyan",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6)
        )
    ax.plot(clicked_ifu[0], clicked_ifu[1], "ro", ms=8, label="Selected IFU star")
    ax.plot(clicked_cat[0], clicked_cat[1], "go", ms=8, label="Catalogue click")
    ax.plot(x_cat[matched_index], y_cat[matched_index], "gs", ms=8, label="Matched catalogue star")
    base_title = "Manual astrometry check"
    if title_info:
        ax.set_title(f"{title_info}\n{base_title}")
    else:
        ax.set_title(base_title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# INTERACTIVE MATCHER (unchanged)
# ============================================================
class InteractiveMatcher:
    def __init__(self, image: np.ndarray, wcs: WCS, df_cat: pd.DataFrame, title_info: str = ""):
        self.image = image
        self.wcs = wcs
        self.df_cat = df_cat
        self.title_info = title_info
        self.clicked_ifu = None
        self.clicked_cat = None
        self.rejected = False
        self.fig = plt.figure(figsize=(20, 14))
        self.ax = plt.subplot(projection=wcs)
        self.plot_base()
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)

    def plot_base(self):
        finite = self.image[np.isfinite(self.image)]
        if finite.size > 0:
            vmin, vmax = np.nanpercentile(finite, [5, 99.5])
        else:
            vmin, vmax = None, None
        im = self.ax.imshow(self.image, origin="lower", cmap="hot", vmin=vmin, vmax=vmax)
        cbar = self.fig.colorbar(im, ax=self.ax,  pad=0.02, shrink=0.95)
        cbar.set_label("Collapsed flux")
        
        coords = SkyCoord(self.df_cat["ra"].values * u.deg, self.df_cat["dec"].values * u.deg)
        x, y = self.wcs.world_to_pixel(coords)
        self.ax.scatter(x, y, color="cyan", marker="x", s=60)
        for i, (xx, yy, k) in enumerate(zip(x, y, self.df_cat["K"])):
            self.ax.text(
                xx + 0.5,
                yy + 0.5,
                f"{i}:{k:.1f}",
                color="cyan",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6)
            )
        base_title = "Left click: IFU star, then catalogue star | Right click: reject IFU"
        if self.title_info:
            self.ax.set_title(f"{self.title_info}\n{base_title}", fontsize=11)
        else:
            self.ax.set_title(base_title, fontsize=11)

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 3:
            self.rejected = True
            print("This IFU was marked as rejected. No astrometric correction will be applied.")
            plt.close(self.fig)
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if self.clicked_ifu is None:
            self.clicked_ifu = (x, y)
            self.ax.plot(x, y, "ro", ms=8)
            self.ax.text(x + 0.5, y + 0.5, "IFU", color="red", fontsize=10)
            print("Selected IFU star")
            self.fig.canvas.draw_idle()
        elif self.clicked_cat is None:
            self.clicked_cat = (x, y)
            self.ax.plot(x, y, "go", ms=8)
            self.ax.text(x + 0.5, y + 0.5, "CAT", color="lime", fontsize=10)
            print("Selected catalogue star")
            self.fig.canvas.draw_idle()
            plt.close(self.fig)

    def run(self):
        print("\nInteractive astrometry correction")
        print("--------------------------------")
        print("Left click 1: select the IFU star")
        print("Left click 2: select the matching catalogue star")
        print("Right click : reject this IFU because there is no reliable coincidence")
        print("")
        print("After a valid pair of left clicks, the window will close automatically.")
        print("After a right click, the IFU will be skipped with no correction applied.\n")
        plt.show(block=True)


# ============================================================
# MAIN (adapted for concatenations)
# ============================================================
def main():
    global BASE_DIR

    df_cat_full = load_catalogue(catalogue_file)

    for con_name in CONS:
        print(f"\n{'='*50}")
        print(f"MANUAL MODE – CONCATENATION: {con_name}")
        print(f"{'='*50}")

        # Build the path to the directory that contains the OB folders
        dummy_sky = Path(
            BASE_DIR_TEMPLATE.format(
                pointing=pointing, con_name=con_name, ob_name="dummy"
            )
        )
        con_dir = dummy_sky.parent.parent   # dummy_sky -> OB dir -> con dir

        if not con_dir.exists():
            print(f"Concatenation directory not found: {con_dir}")
            continue

        ob_dirs = sorted([
            d for d in con_dir.iterdir()
            if d.is_dir() and d.name.startswith("OB")
        ])

        if not ob_dirs:
            print(f"No OB folders found inside {con_dir}")
            continue

        print(f"Found {len(ob_dirs)} OB(s): {[d.name for d in ob_dirs]}")

        for ob_dir in ob_dirs:
            ob_name = ob_dir.name

            # Set BASE_DIR to the exact sky_tweak folder for this OB
            BASE_DIR = Path(
                BASE_DIR_TEMPLATE.format(
                    pointing=pointing, con_name=con_name, ob_name=ob_name
                )
            )

            if not BASE_DIR.exists():
                print(f"WARNING: sky_tweak folder missing for {ob_name}, skipping")
                continue

            print(f"\n========== Processing OB: {ob_name} ==========")

            paths = get_ob_paths(ob_name)

            corrected_cube_output = paths["CORRECTED_CUBE_OUTPUT"]
            corrected_collapsed_output = paths["CORRECTED_COLLAPSED_OUTPUT"]
            check_plot_output = paths["CHECK_PLOT_OUTPUT"]

            fits_files = get_uncorrected_ifus(paths)

            print(f"Uncorrected IFUs: {len(fits_files)}")

            if len(fits_files) == 0:
                continue

            for fits_file in fits_files:
                print(f"\n--- Processing {fits_file.name} ---")

                with fits.open(fits_file) as hdul:
                    cube = hdul[1].data
                    hdr = hdul[1].header
                    wcs = WCS(hdr).celestial
                    
                    wcs_spectral = WCS(hdr).sub(["spectral"])
                    arm_label = get_arm_label(hdr, fits_file.name)
                    corrected_filename = f"{fits_file.stem}_astrocorr.fits"
                    title_info = f"{corrected_filename} | {arm_label}"                    
                    ref_coord = SkyCoord(hdr["CRVAL1"] * u.deg, hdr["CRVAL2"] * u.deg)

                    df_cat = select_local_catalogue(df_cat_full, ref_coord)

                    print(f"Local catalogue stars: {len(df_cat)}")

                    if len(df_cat) == 0:
                        print("No catalogue stars nearby -> skipping")
                        continue

                    image, wave_idx = collapse_cube(              # now returns two objects
                        cube=cube,
                        wcs_spectral=wcs_spectral,
                        edge_trim_fraction=0.08, # EDGE_TRIM_FRACTION in astrometry correction .py
                    )

                    matcher = InteractiveMatcher(image, wcs, df_cat, title_info=title_info)
                    matcher.run()

                    if matcher.rejected:
                        print("Rejected -> no correction")
                        out_txt = corrected_cube_output / f"{fits_file.stem}_manual_rejected.txt"
                        with open(out_txt, "w") as f:
                            f.write("Manual astrometry result: REJECTED\n")
                        continue

                    if matcher.clicked_ifu is None or matcher.clicked_cat is None:
                        print("Selection not completed -> skipping")
                        continue

                    ifu_coord = wcs.pixel_to_world(
                        matcher.clicked_ifu[0],
                        matcher.clicked_ifu[1]
                    )

                    coords_cat = SkyCoord(
                        df_cat["ra"].values * u.deg,
                        df_cat["dec"].values * u.deg
                    )

                    click_coord = wcs.pixel_to_world(
                        matcher.clicked_cat[0],
                        matcher.clicked_cat[1]
                    )

                    idx = coords_cat.separation(click_coord).argmin()

                    ra_cat = float(df_cat.iloc[idx]["ra"])
                    dec_cat = float(df_cat.iloc[idx]["dec"])

                    dra = ra_cat - ifu_coord.ra.deg
                    ddec = dec_cat - ifu_coord.dec.deg

                    print(f"Matched to catalogue star {idx} (K={df_cat.iloc[idx]['K']:.2f})")
                    print(f"Shift: dRA={dra*3600:.3f}\"  dDec={ddec*3600:.3f}\"")

                    corrected = apply_shift(hdul, dra, ddec)

                    out_cube = corrected_cube_output / f"{fits_file.stem}_astrocorr.fits"
                    corrected.writeto(out_cube, overwrite=True)
                    print(f"Saved cube: {out_cube}")

                    out_collapsed = corrected_collapsed_output / f"COLLAPSED_{fits_file.name}_astrocorr.fits"
                    save_corrected_collapsed(
                        image=image,
                        wcs=wcs,
                        dra_deg=dra,
                        ddec_deg=ddec,
                        output_path=out_collapsed
                    )
                    print(f"Saved collapsed: {out_collapsed}")

                    out_plot = check_plot_output / f"{fits_file.stem}_manual_check.png"
                    save_manual_check_plot(
                        image=image,
                        wcs=wcs,
                        df_cat=df_cat,
                        clicked_ifu=matcher.clicked_ifu,
                        clicked_cat=matcher.clicked_cat,
                        matched_index=int(idx),
                        output_path=out_plot,
                        title_info=title_info,
                    )
                    print(f"Saved manual check plot: {out_plot}")


if __name__ == "__main__":
    main()