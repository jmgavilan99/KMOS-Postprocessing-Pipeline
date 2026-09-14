#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KMOS astrometric recentering from scratch (multi-concatenation, parallel version).

Philosophy
----------
1. For each concatenation (con1, con2, ...) automatically discover all OB
   subdirectories inside it.
2. For each OB, read all IFU cubes from its sky_tweak directory.
3. For each IFU, use the header reference coordinates to extract a local
   catalogue region within a configurable radius.
4. Build a collapsed image from the cube, excluding blue/red edge wavelengths.
5. Detect stars with DAOStarFinder using a PSF FWHM estimated from header
   keywords.
6. Sort detected IFU stars by flux and catalogue stars by K magnitude.
7. Force the brightest IFU source onto the brightest catalogue source, then the
   second brightest catalogue source, etc.
8. For each trial shift, evaluate whether the remaining detected stars have
   sensible one-to-one counterparts in the catalogue.
9. Keep the best valid solution, update the WCS of the original cube and of the
   collapsed image, save both, and create a visual check plot.

Parallelisation
---------------
- OBs are fully independent (own input dir, own output dirs), so they are the
  natural parallelisation unit.
- A ProcessPoolExecutor is used. The catalogue FITS file is loaded once per
  worker through an initializer, so it is not re-pickled for every task.
- Matplotlib is forced to the non-interactive "Agg" backend before pyplot is
  imported; otherwise forked/spawned workers may try to use an interactive
  backend and crash.
- The number of workers is chosen conservatively so the script is safe on a
  shared server: it respects an explicit env override, then scheduler-provided
  CPU counts (SLURM/PBS/SGE), then the actual CPU affinity of the process, and
  only then falls back to a small default. os.cpu_count() is deliberately NOT
  used because it reports the whole node, not the user's allocation.

Notes
-----
- The local catalogue radius is controlled by CATALOGUE_SEARCH_RADIUS_ARCSEC.
- The spectral collapse excludes the blue/red edges using EDGE_TRIM_FRACTION.
- The matching requires mutual nearest neighbours.
- The brightest IFU star must be matched.
- If only one star is detected, the first valid brightest-star match is used.
- This script assumes one cube per FITS file with science data in extension 1.

Author: (your name)
"""

import datetime
import os
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# IMPORTANT: force a non-interactive backend BEFORE importing pyplot.
# Required when using ProcessPoolExecutor / multiprocessing.
import matplotlib
matplotlib.use("Agg")

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder


# =============================================================================
# CONFIGURATION - Edit only these variables
# =============================================================================
pointing = "PX"                     # pointing name
CONS = ["con3"]     # concatenations to process

# Catalogue path (no extra spaces)
GNS_TXT_PATH = Path("/home/jmgavilan/Desktop/KMOS/GNS_VIRAC_cat/FINAL_EAST_CENTRAL_WEST_VIRAC.fits")

# Template for the sky_tweak directory of each OB.
# Must match the automatic pipeline structure.
BASE_DIR_TEMPLATE = (
    # "/home/data/KMOS/PILOT/reduced/P113/{pointing}/{con_name}/{ob_name}/sky_tweak/" # Server Path
    "/home/jmgavilan/Desktop/PX/{pointing}/{con_name}/{ob_name}/sky_tweak/"  # Local test     # Local test
)



# -------------------------------------------------------------------------
# Safe worker-count detection for shared / HPC machines
# -------------------------------------------------------------------------
def _detect_safe_n_workers(default_fallback: int = 4) -> int:
    """
    Decide a safe number of worker processes on a shared / HPC machine.

    Priority order
    --------------
    1. Explicit user override via the KMOS_N_WORKERS environment variable.
    2. Scheduler-provided CPU counts:
           - SLURM_CPUS_PER_TASK (Slurm)
           - PBS_NP                (PBS/Torque)
           - NSLOTS                (SGE)
    3. CPU affinity of the current process (Linux cgroups / taskset / containers).
    4. Conservative fallback (default_fallback), NEVER os.cpu_count().

    Rationale
    ---------
    os.cpu_count() reports the physical cores of the whole node, which is
    meaningless on a shared server. Using it would spawn far more workers than
    the user is actually allowed to use, starving colleagues and the OS.
    """
    # 1) Explicit user override
    env_override = os.environ.get("KMOS_N_WORKERS")
    if env_override:
        try:
            n = int(env_override)
            if n >= 1:
                return n
        except ValueError:
            pass

    # 2) Scheduler-provided allocation
    for var in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        val = os.environ.get(var)
        if val:
            try:
                n = int(val)
                if n >= 1:
                    return n
            except ValueError:
                pass

    # 3) Actual CPU affinity of this process (respects cgroups / taskset)
    if hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
            if n >= 1:
                return n
        except OSError:
            pass

    # 4) Conservative fallback for shared machines
    return default_fallback


N_WORKERS = _detect_safe_n_workers(default_fallback=4)


# -------------------------------------------------------------------------
# User parameters (unchanged from original)
# -------------------------------------------------------------------------
INPUT_PATTERN = "COMBINE_SKY_TWEAK_*.fits"
CATALOGUE_SEARCH_RADIUS_ARCSEC = 5.0
REFERENCE_K_LIMIT = 15.0
EDGE_TRIM_FRACTION = 0.08
KARMA_THRESHOLD_SIGMA = 30.0
KARMA_MIN_SEPARATION_ARCSEC = 0.6
MATCH_RADIUS_ARCSEC = 0.5
MAX_TOTAL_SHIFT_ARCSEC = 2.5
MIN_NMATCH = 2
MIN_MATCH_FRACTION = 0.4
FLUX_TOLERANCE_RATIO = 0.70
SAVE_COLLAPSED_FITS = True
SAVE_CHECK_PLOTS = True
BRIGHT_ANCHOR_MAG_RANGE = 3
EDGE_EXCLUSION_PIXELS = 1
brightest_k_tolerance_mag = 0.75
MIN_RMS_IMPROVEMENT_ARCSEC = 0.01


# =============================================================================
# Auxiliary functions
# =============================================================================
def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def get_ob_paths(ob_name: str, base_dir: Path) -> dict:
    """
    Build all output paths for an OB given its sky_tweak base directory.
    """
    ob_dir = base_dir / f"res_{ob_name}"

    paths = {
        "NAME": ob_name,
        "OB_DIR": ob_dir,
        "IFU_ROOT": base_dir,
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


# =============================================================================
# General utilities
# =============================================================================
def filter_edge_sources(
    df_det: pd.DataFrame,
    image_shape: tuple,
    edge_pixels: int,
) -> pd.DataFrame:
    """
    Remove detections too close to the image borders.

    This avoids spurious detections caused by bright stars outside the IFU.
    """
    if len(df_det) == 0:
        return df_det

    ny, nx = image_shape
    b = edge_pixels

    mask = (
        (df_det["x_pix"] >= b) &
        (df_det["x_pix"] <= (nx - b)) &
        (df_det["y_pix"] >= b) &
        (df_det["y_pix"] <= (ny - b))
    )

    df_filtered = df_det.loc[mask].copy().reset_index(drop=True)

    print(f"  Edge filtering: {len(df_det)} -> {len(df_filtered)} detections (border={b}px)")

    return df_filtered


def build_anchor_trial_order(
    df_det: pd.DataFrame,
    df_cat: pd.DataFrame,
    bright_anchor_mag_range: float,
    det_anchor_index: int,
):
    """
    Build the catalogue anchor trial order for one chosen detected KMOS anchor.

    Strategy
    --------
    1. Choose one detected KMOS source as anchor.
    2. Find the brightest local catalogue star.
    3. Select catalogue stars within:
           K <= K_brightest + bright_anchor_mag_range
    4. Order those selected stars by distance to the chosen detected anchor.
    5. Append the remaining catalogue stars ordered by K.

    Returns
    -------
    ordered_indices : list[int]
        Indices of df_cat in the order that should be tested.
    """

    if len(df_det) == 0 or len(df_cat) == 0:
        return []

    det_anchor = SkyCoord(
        ra=float(df_det.iloc[det_anchor_index]["ra_deg"]) * u.deg,
        dec=float(df_det.iloc[det_anchor_index]["dec_deg"]) * u.deg
    )

    coords_cat = SkyCoord(
        ra=df_cat["ra_deg"].to_numpy() * u.deg,
        dec=df_cat["dec_deg"].to_numpy() * u.deg
    )

    k_array = df_cat["K"].to_numpy(dtype=float)
    k_min = np.nanmin(k_array)

    bright_mask = k_array <= (k_min + bright_anchor_mag_range)
    bright_indices = np.where(bright_mask)[0]

    if len(bright_indices) > 0:
        sep_bright = det_anchor.separation(coords_cat[bright_indices]).arcsec
        order_bright = bright_indices[np.argsort(sep_bright)]
    else:
        order_bright = np.array([], dtype=int)

    remaining_indices = np.array(
        [i for i in range(len(df_cat)) if i not in set(order_bright)],
        dtype=int
    )

    if len(remaining_indices) > 0:
        order_remaining = remaining_indices[np.argsort(k_array[remaining_indices])]
    else:
        order_remaining = np.array([], dtype=int)

    return list(order_bright) + list(order_remaining)


def make_final_check_plot(
    corrected_cube_path: Path,
    df_catalogue_local: pd.DataFrame,
    output_path: Path,
    threshold_sigma: float,
    min_separation_arcsec: float,
):
    """
    Final verification plot:
    - uses the corrected cube
    - rebuilds the collapsed image
    - re-detects stars in the shifted IFU
    - overplots the catalogue stars

    This is the final visual product to check whether the astrometry looks right.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(corrected_cube_path) as hdul:
        cube = hdul[1].data
        hdr0 = hdul[0].header
        hdr1 = hdul[1].header

        wcs_celestial = WCS(hdr1).celestial
        wcs_spectral = WCS(hdr1).sub(["spectral"])

        image_clean, _ = collapse_cube(
            cube=cube,
            wcs_spectral=wcs_spectral,
            edge_trim_fraction=EDGE_TRIM_FRACTION,
        )

        df_det = detect_sources_in_image(
            image_clean=image_clean,
            wcs_celestial=wcs_celestial,
            hdr0=hdr0,
            hdr1=hdr1,
            threshold_sigma=threshold_sigma,
            min_separation_arcsec=min_separation_arcsec,
        )

        coords_cat = SkyCoord(
            ra=df_catalogue_local["ra_deg"].to_numpy() * u.deg,
            dec=df_catalogue_local["dec_deg"].to_numpy() * u.deg
        )
        x_cat, y_cat = wcs_celestial.world_to_pixel(coords_cat)

        fig = plt.figure(figsize=(9, 7))
        ax = plt.subplot(projection=wcs_celestial)

        finite = image_clean[np.isfinite(image_clean)]
        vmin = np.nanpercentile(finite, 5) if finite.size else None
        vmax = np.nanpercentile(finite, 99.5) if finite.size else None

        im = ax.imshow(
            image_clean,
            origin="lower",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )

        # Corrected IFU detections
        if len(df_det) > 0:
            positions = np.transpose((df_det["x_pix"].to_numpy(), df_det["y_pix"].to_numpy()))
            apertures = CircularAperture(positions, r=2.5)
            apertures.plot(color="cyan", lw=1.8, ax=ax)

            for i, row in df_det.iterrows():
                ax.text(
                    row["x_pix"] + 0.8,
                    row["y_pix"] + 0.8,
                    f"D{i+1}",
                    color="white",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.7)
                )

        # Catalogue stars
        ax.scatter(
            x_cat, y_cat,
            s=45,
            color="deepskyblue",
            marker="x",
            linewidths=1.3,
            zorder=4,
        )

        for i, (xg, yg, k_mag) in enumerate(zip(x_cat, y_cat, df_catalogue_local["K"].to_numpy())):
            ax.text(
                xg + 0.8,
                yg + 0.8,
                f"C{i+1}:{k_mag:.1f}",
                color="cyan",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.7)
            )

        ax.set_title(corrected_cube_path.name, fontsize=11)
        ax.set_xlabel("R.A.")
        ax.set_ylabel("Dec.")

        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
        cbar.set_label("Collapsed flux")

        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _safe_float(value) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def robust_psf_fwhm_arcsec_from_headers(
    hdr0,
    hdr1,
    min_valid_arcsec: float = 0.2,
    max_valid_arcsec: float = 3.0,
    lambda_science_micron: float = 2.20,
    lambda_optical_micron: float = 0.50,
) -> tuple[float, dict]:
    """
    Estimate PSF FWHM in arcsec using several header keywords.

    Strategy
    --------
    1. Read several seeing / PSF-related keywords.
    2. For SKY_RES, test multiple interpretations:
       - already in arcsec
       - divided by 100
       - divided by 1000
    3. For optical seeing indicators, convert them to the science wavelength
       using:
           FWHM ~ lambda^(-1/5)
    4. Reject non-finite or implausible values.
    5. Adopt the median of the accepted values.
    """

    info = {}
    accepted = []
    rejected = []
    candidates = []

    # Seeing wavelength scaling: optical -> science wavelength
    wave_scale_opt_to_sci = (lambda_science_micron / lambda_optical_micron) ** (-0.2)

    ia_fwhm = _safe_float(
        hdr0.get("HIERARCH ESO TEL IA FWHM", hdr1.get("HIERARCH ESO TEL IA FWHM"))
    )
    ia_fwhmlin = _safe_float(
        hdr0.get("HIERARCH ESO TEL IA FWHMLIN", hdr1.get("HIERARCH ESO TEL IA FWHMLIN"))
    )
    ambi_start = _safe_float(
        hdr0.get("HIERARCH ESO TEL AMBI FWHM START", hdr1.get("HIERARCH ESO TEL AMBI FWHM START"))
    )
    ambi_end = _safe_float(
        hdr0.get("HIERARCH ESO TEL AMBI FWHM END", hdr1.get("HIERARCH ESO TEL AMBI FWHM END"))
    )
    sky_res_raw = _safe_float(
        hdr0.get("SKY_RES", hdr1.get("SKY_RES"))
    )

    info["IA_FWHM_RAW"] = ia_fwhm
    info["IA_FWHMLIN_RAW"] = ia_fwhmlin
    info["AMBI_FWHM_START_RAW"] = ambi_start
    info["AMBI_FWHM_END_RAW"] = ambi_end
    info["SKY_RES_RAW"] = sky_res_raw
    info["LAMBDA_SCIENCE_MICRON"] = lambda_science_micron
    info["LAMBDA_OPTICAL_MICRON"] = lambda_optical_micron
    info["WAVE_SCALE_OPT_TO_SCI"] = wave_scale_opt_to_sci

    def add_candidate(name: str, value: Optional[float], scale_to_science: bool = False) -> None:
        if value is None:
            rejected.append((name, None, "missing"))
            return

        val = float(value)

        if scale_to_science:
            val = val * wave_scale_opt_to_sci
            name = f"{name}_scaled_to_{lambda_science_micron:.2f}um"

        if val < min_valid_arcsec:
            rejected.append((name, val, f"<{min_valid_arcsec:.2f} arcsec"))
            return

        if val > max_valid_arcsec:
            rejected.append((name, val, f">{max_valid_arcsec:.2f} arcsec"))
            return

        candidates.append((name, val))
        accepted.append((name, val))

    def add_sky_res_candidates(raw_value: Optional[float]) -> None:
        if raw_value is None:
            rejected.append(("SKY_RES", None, "missing"))
            return

        interpretations = [
            ("SKY_RES_arcsec", raw_value),
            ("SKY_RES_div100", raw_value / 100.0),
            ("SKY_RES_div1000", raw_value / 1000.0),
        ]

        valid = []
        invalid = []

        for name, val in interpretations:
            if val < min_valid_arcsec:
                invalid.append((name, val, f"<{min_valid_arcsec:.2f} arcsec"))
            elif val > max_valid_arcsec:
                invalid.append((name, val, f">{max_valid_arcsec:.2f} arcsec"))
            else:
                valid.append((name, val))

        if len(valid) == 0:
            rejected.extend(invalid)
            return

        # Choose the valid interpretation closest to 1 arcsec
        chosen_name, chosen_val = min(valid, key=lambda x: abs(x[1] - 1.0))
        candidates.append((chosen_name, chosen_val))
        accepted.append((chosen_name, chosen_val))

        for name, val in valid:
            if name != chosen_name:
                rejected.append((name, val, "plausible_but_not_selected"))

        rejected.extend(invalid)

    # SKY_RES: unit ambiguity only
    add_sky_res_candidates(sky_res_raw)

    # Optical seeing indicators: convert to science wavelength
    add_candidate("IA_FWHM", ia_fwhm, scale_to_science=True)
    add_candidate("IA_FWHMLIN", ia_fwhmlin, scale_to_science=True)
    add_candidate("AMBI_FWHM_START", ambi_start, scale_to_science=True)
    add_candidate("AMBI_FWHM_END", ambi_end, scale_to_science=True)

    if len(candidates) == 0:
        raise ValueError(
            "Could not derive a valid PSF FWHM from headers "
            f"within [{min_valid_arcsec:.2f}, {max_valid_arcsec:.2f}] arcsec."
        )

    adopted = float(np.median([v for _, v in candidates]))

    info["ACCEPTED_CANDIDATES"] = accepted
    info["REJECTED_CANDIDATES"] = rejected
    info["USED_CANDIDATES"] = candidates
    info["ADOPTED_FWHM_ARCSEC"] = adopted

    return adopted, info


def read_catalogue_fits(path: Path) -> pd.DataFrame:
    """
    Read catalogue from a FITS binary table.
    Expected columns: RA, Dec, and a magnitude (preferentially K, but can be
    named Kmag, KMAG, etc.)
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
            # Fallback to any column containing 'mag'
            mag_candidates = [c for c in columns if 'mag' in c]
            if mag_candidates:
                mag_col = mag_candidates[0]  # take first
            else:
                raise ValueError(f"Cannot find magnitude column. Available columns: {columns}")

        # Build DataFrame with standard names
        df = pd.DataFrame({
            'ra_deg': data[ra_col].astype(float),
            'dec_deg': data[dec_col].astype(float),
            'K': data[mag_col].astype(float)
        })

        # Filter valid entries
        good = (
            np.isfinite(df['ra_deg'].to_numpy()) &
            np.isfinite(df['dec_deg'].to_numpy()) &
            np.isfinite(df['K'].to_numpy())
        )
        return df.loc[good].copy().reset_index(drop=True)


def read_catalogue_txt(path: Path) -> pd.DataFrame:
    """
    Read GNUCLEUS/GNS plain text catalogue with columns:
    ra dec J dJ H dH K dK
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["ra_deg", "dec_deg", "J", "dJ", "H", "dH", "K", "dK"],
        engine="python",
    )
    good = (
        np.isfinite(df["ra_deg"].to_numpy()) &
        np.isfinite(df["dec_deg"].to_numpy()) &
        np.isfinite(df["K"].to_numpy())
    )
    return df.loc[good].copy().reset_index(drop=True)


def get_reference_coord_from_header(header) -> SkyCoord:
    """
    Use the IFU WCS CRVAL as the reference coordinate for the catalogue cut.
    """
    ra = header.get("CRVAL1")
    dec = header.get("CRVAL2")
    if ra is None or dec is None:
        raise ValueError("CRVAL1/CRVAL2 not found in science header.")
    return SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)


def select_local_catalogue(
    df_catalogue: pd.DataFrame,
    centre: SkyCoord,
    radius_arcsec: float,
    k_limit: float,
) -> pd.DataFrame:
    coords = SkyCoord(
        ra=df_catalogue["ra_deg"].to_numpy() * u.deg,
        dec=df_catalogue["dec_deg"].to_numpy() * u.deg
    )
    sep = coords.separation(centre).arcsec
    mask = (sep <= radius_arcsec) & (df_catalogue["K"].to_numpy() <= k_limit)
    df_local = df_catalogue.loc[mask].copy()
    df_local["sep_ref_arcsec"] = sep[mask]
    df_local = df_local.sort_values(["K", "sep_ref_arcsec"], ascending=[True, True]).reset_index(drop=True)
    return df_local


def collapse_cube(
    cube: np.ndarray,
    wcs_spectral: WCS,
    edge_trim_fraction: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse the cube after trimming spectral edges.
    Returns collapsed image and indices of wavelengths used.
    """
    nz = cube.shape[0]
    trim = int(np.floor(edge_trim_fraction * nz))
    i0 = trim
    i1 = nz - trim

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


def save_collapsed_image_as_fits(
    hdul: fits.HDUList,
    image_clean: np.ndarray,
    input_path: Path,
    output_dir: Path,
    wave_idx: np.ndarray,
    overwrite: bool = True,
) -> Path:
    """
    Save collapsed image with celestial WCS in extension 1.
    """
    hdr0 = hdul[0].header.copy()
    hdr1 = hdul[1].header.copy()

    wcs_2d = WCS(hdr1).celestial
    new_data_header = wcs_2d.to_header()

    preserve = [
        "OBJECT", "TELESCOP", "INSTRUME", "OBSERVER", "DATE-OBS",
        "EXPTIME", "AIRMASS", "FILTER", "RA", "DEC", "EQUINOX",
        "RADESYS", "BUNIT", "ORIGIN"
    ]
    for key in preserve:
        if key in hdr1:
            if key == "BUNIT":
                new_data_header[key] = str(hdr1[key]).replace(".angstrom**(-1)", "")
            else:
                new_data_header[key] = hdr1[key]

    new_data_header["NAXIS"] = 2
    new_data_header["NAXIS1"] = image_clean.shape[1]
    new_data_header["NAXIS2"] = image_clean.shape[0]
    new_data_header["IMAGETYP"] = "COLLAPSED_SPECTRUM"
    new_data_header["EXTNAME"] = "DATA"
    new_data_header["ORIGFILE"] = input_path.name
    new_data_header["WAVEI0"] = int(wave_idx[0])
    new_data_header["WAVEI1"] = int(wave_idx[-1])

    primary = fits.PrimaryHDU(data=None, header=hdr0)
    data_hdu = fits.ImageHDU(data=image_clean, header=new_data_header)
    hdul_out = fits.HDUList([primary, data_hdu])

    outfile = output_dir / f"COLLAPSED_{input_path.name}"
    hdul_out.writeto(outfile, overwrite=overwrite)
    return outfile


def detect_sources_in_image(
    image_clean: np.ndarray,
    wcs_celestial: WCS,
    hdr0,
    hdr1,
    threshold_sigma: float,
    min_separation_arcsec: float,
) -> pd.DataFrame:
    """
    Detect stars in the collapsed image and return a DataFrame sorted by flux.
    """
    fwhm_arcsec, psf_info = robust_psf_fwhm_arcsec_from_headers(
        hdr0=hdr0,
        hdr1=hdr1,
        min_valid_arcsec=0.2,
        max_valid_arcsec=3.0,
        lambda_science_micron=2.20,
        lambda_optical_micron=0.50,
    )

    pixel_scale_arcsec = abs(proj_plane_pixel_scales(wcs_celestial)[0]) * u.deg
    pixel_scale_arcsec = pixel_scale_arcsec.to(u.arcsec).value

    fwhm_pix = fwhm_arcsec / pixel_scale_arcsec
    min_separation_pix = min_separation_arcsec / pixel_scale_arcsec

    image_cropped = image_clean[1:-1, 1:-1]
    _, _, std = sigma_clipped_stats(image_cropped, sigma=2)
    threshold = threshold_sigma * std

    starfinder = DAOStarFinder(
        threshold=threshold,
        fwhm=fwhm_pix,
        min_separation=min_separation_pix
    )
    sources = starfinder(image_clean)

    print(f"  PSF adopted = {fwhm_arcsec:.3f}\" ({fwhm_pix:.2f} pix)")
    print(f"  Threshold   = {threshold_sigma:.1f} sigma")
    print(f"  Min sep     = {min_separation_arcsec:.2f}\"")
    if len(psf_info.get("ACCEPTED_CANDIDATES", [])) > 0:
        print("  Accepted PSF indicators:")
        for name, val in psf_info["ACCEPTED_CANDIDATES"]:
            print(f"    - {name}: {val:.3f}\"")

    if len(psf_info.get("REJECTED_CANDIDATES", [])) > 0:
        print("  Rejected PSF indicators:")
        for name, val, reason in psf_info["REJECTED_CANDIDATES"]:
            if val is None:
                print(f"    - {name}: [{reason}]")
            else:
                print(f"    - {name}: {val:.3f}\" [{reason}]")

    if sources is None or len(sources) == 0:
        return pd.DataFrame(columns=["x_pix", "y_pix", "ra_deg", "dec_deg", "flux"])

    positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
    world = wcs_celestial.pixel_to_world(positions[:, 0], positions[:, 1])

    rows = []
    ny, nx = image_clean.shape
    for x, y, ra, dec in zip(positions[:, 0], positions[:, 1], world.ra.deg, world.dec.deg):
        xi = int(round(x))
        yi = int(round(y))
        flux = image_clean[yi, xi] if 0 <= xi < nx and 0 <= yi < ny else np.nan
        rows.append({
            "x_pix": float(x),
            "y_pix": float(y),
            "ra_deg": float(ra),
            "dec_deg": float(dec),
            "flux": float(flux),
        })

    df = pd.DataFrame(rows)
    good = (
        np.isfinite(df["ra_deg"].to_numpy()) &
        np.isfinite(df["dec_deg"].to_numpy()) &
        np.isfinite(df["flux"].to_numpy())
    )
    df = df.loc[good].copy()

    if len(df) == 0:
        return df

    return df.sort_values("flux", ascending=False).reset_index(drop=True)


def is_flux_magnitude_consistent(
    det_flux,
    cat_K,
    matched_mask,
    idx,
    flux_tolerance_ratio=0.80,
):
    matched_mask = np.asarray(matched_mask, dtype=bool)
    idx = np.asarray(idx, dtype=int)
    det_flux = np.asarray(det_flux, dtype=float)
    cat_K = np.asarray(cat_K, dtype=float)

    if np.sum(matched_mask) < 2:
        return True

    flux_matched = det_flux[matched_mask]
    k_matched = cat_K[idx[matched_mask]]

    good = np.isfinite(flux_matched) & np.isfinite(k_matched)
    flux_matched = flux_matched[good]
    k_matched = k_matched[good]

    if len(flux_matched) < 2:
        return True

    order = np.argsort(k_matched)
    flux_sorted = flux_matched[order]

    for i in range(len(flux_sorted) - 1):
        brighter_flux = flux_sorted[i]
        fainter_flux = flux_sorted[i + 1]
        if brighter_flux < flux_tolerance_ratio * fainter_flux:
            return False

    return True


def evaluate_shift_candidate(
    det_ra,
    det_dec,
    det_flux,
    coords_cat,
    cat_K_array,
    shift_ra_deg,
    shift_dec_deg,
    match_radius_arcsec,
    min_nmatch,
    min_match_fraction,
    flux_tolerance_ratio,
    require_brightest_match=True,
    enforce_brightest_photometric_consistency=True,
    brightest_k_tolerance_mag=brightest_k_tolerance_mag,
):
    """
    Evaluate one shift by mutual nearest-neighbour matching.
    Returns a dictionary with success/failure information.

    Photometric logic
    -----------------
    1. Keep the geometric conditions.
    2. Enforce global flux-magnitude consistency.
    3. Explicitly reject solutions where the brightest KMOS source is matched to
       a catalogue star that is significantly fainter than catalogue stars
       matched to weaker KMOS detections.
    """

    shifted_ra = det_ra + shift_ra_deg
    shifted_dec = det_dec + shift_dec_deg

    coords_ifu = SkyCoord(ra=shifted_ra * u.deg, dec=shifted_dec * u.deg)

    idx_k2c, sep_k2c, _ = coords_ifu.match_to_catalog_sky(coords_cat)
    idx_k2c = np.asarray(idx_k2c, dtype=int)
    sep_k2c_arcsec = np.asarray(sep_k2c.arcsec, dtype=float)

    idx_c2k, _, _ = coords_cat.match_to_catalog_sky(coords_ifu)
    idx_c2k = np.asarray(idx_c2k, dtype=int)

    matched = np.zeros(len(det_ra), dtype=bool)
    for i_det, i_cat in enumerate(idx_k2c):
        if sep_k2c_arcsec[i_det] >= match_radius_arcsec:
            continue
        if idx_c2k[i_cat] == i_det:
            matched[i_det] = True

    n_det = len(det_ra)
    nmatch = int(np.sum(matched))
    match_fraction = nmatch / n_det if n_det > 0 else 0.0
    brightest_matched = bool(matched[0]) if n_det > 0 else False

    result = {
        "accepted": False,
        "reason": "",
        "matched": matched,
        "idx": idx_k2c.copy(),
        "sep_arcsec": sep_k2c_arcsec.copy(),
        "nmatch": nmatch,
        "match_fraction": match_fraction,
        "brightest_matched": brightest_matched,
        "shift_ra_deg": shift_ra_deg,
        "shift_dec_deg": shift_dec_deg,
        "total_shift_arcsec": float(np.hypot(shift_ra_deg * 3600.0, shift_dec_deg * 3600.0)),
    }

    if n_det == 0:
        result["reason"] = "no_detected_stars"
        return result

    if n_det == 1:
        if nmatch < 1:
            result["reason"] = "single_star_not_matched"
            return result
    else:
        if nmatch < min_nmatch:
            result["reason"] = f"nmatch<{min_nmatch}"
            return result
        if match_fraction < min_match_fraction:
            result["reason"] = f"match_fraction<{min_match_fraction:.2f}"
            return result

    if require_brightest_match and not brightest_matched:
        result["reason"] = "brightest_not_matched"
        return result

    # ---------------------------------------------------------
    # STRICT photometric sanity for the brightest KMOS source
    # ---------------------------------------------------------
    if enforce_brightest_photometric_consistency and np.sum(matched) >= 2 and matched[0]:
        matched_indices = np.where(matched)[0]
        k_matched = cat_K_array[idx_k2c[matched_indices]]

        k_brightest_det = cat_K_array[idx_k2c[0]]
        k_best_among_matched = np.nanmin(k_matched)

        # Reject if the brightest KMOS source is assigned to a catalogue star
        # much fainter than another catalogue star in the same accepted match.
        if k_brightest_det > k_best_among_matched + brightest_k_tolerance_mag:
            result["reason"] = "brightest_kmos_assigned_to_too_faint_catalogue_star"
            return result

    # ---------------------------------------------------------
    # Global flux-magnitude consistency
    # ---------------------------------------------------------
    if nmatch >= 2:
        ok_flux = is_flux_magnitude_consistent(
            det_flux=det_flux,
            cat_K=cat_K_array,
            matched_mask=matched,
            idx=idx_k2c,
            flux_tolerance_ratio=flux_tolerance_ratio,
        )
        if not ok_flux:
            result["reason"] = "flux_magnitude_inconsistent"
            return result

    if np.any(matched):
        result["rms_arcsec"] = float(np.sqrt(np.mean(sep_k2c_arcsec[matched] ** 2)))
        result["sum_sep_arcsec"] = float(np.sum(sep_k2c_arcsec[matched]))
        result["max_sep_arcsec"] = float(np.max(sep_k2c_arcsec[matched]))
    else:
        result["rms_arcsec"] = np.inf
        result["sum_sep_arcsec"] = np.inf
        result["max_sep_arcsec"] = np.inf

    result["accepted"] = True
    result["reason"] = "accepted"
    return result


def solve_astrometry_for_ifu(
    df_det: pd.DataFrame,
    df_cat: pd.DataFrame,
    ref_coord: SkyCoord,
    match_radius_arcsec: float,
    max_total_shift_arcsec: float,
    min_nmatch: int,
    min_match_fraction: float,
    flux_tolerance_ratio: float,
    bright_anchor_mag_range: float,
    verbose: bool = True,
):
    """
    Solve astrometry by trying different detected KMOS anchor stars.

    Logic
    -----
    1. Order detected KMOS stars by distance to the IFU reference position.
       This makes the likely target come first.
    2. For each chosen detected anchor:
       - try nearby bright catalogue stars first
       - then the remaining catalogue stars by brightness
    3. Keep the best accepted solution.

    Returns
    -------
    best : dict or None
    best_any : dict or None
    df_debug : pd.DataFrame
    """

    if len(df_det) == 0 or len(df_cat) == 0:
        return None, None, pd.DataFrame()

    det_ra = df_det["ra_deg"].to_numpy(dtype=float)
    det_dec = df_det["dec_deg"].to_numpy(dtype=float)
    det_flux = df_det["flux"].to_numpy(dtype=float)

    coords_det = SkyCoord(ra=det_ra * u.deg, dec=det_dec * u.deg)
    det_sep_ref = coords_det.separation(ref_coord).arcsec
    det_anchor_order = np.argsort(det_sep_ref)

    coords_cat = SkyCoord(
        ra=df_cat["ra_deg"].to_numpy() * u.deg,
        dec=df_cat["dec_deg"].to_numpy() * u.deg
    )
    cat_K_array = df_cat["K"].to_numpy(dtype=float)

    best = None
    best_any = None
    debug_rows = []

    # ---------------------------------------------------------
    # Zero-shift sanity check
    # ---------------------------------------------------------
    candidate_zero = evaluate_shift_candidate(
        det_ra=det_ra,
        det_dec=det_dec,
        det_flux=det_flux,
        coords_cat=coords_cat,
        cat_K_array=cat_K_array,
        shift_ra_deg=0.0,
        shift_dec_deg=0.0,
        match_radius_arcsec=match_radius_arcsec,
        min_nmatch=min_nmatch,
        min_match_fraction=min_match_fraction,
        flux_tolerance_ratio=flux_tolerance_ratio,
        require_brightest_match=True,
    )

    candidate_zero["anchor_cat_rank"] = 0
    candidate_zero["anchor_trial_rank"] = 0
    candidate_zero["det_anchor_rank"] = 0
    candidate_zero["det_anchor_index"] = -1
    candidate_zero["det_anchor_flux"] = np.nan
    candidate_zero["det_anchor_sep_ref_arcsec"] = np.nan
    candidate_zero["anchor_cat_K"] = np.nan

    debug_rows.append({
        "det_anchor_rank": 0,
        "det_anchor_index": -1,
        "det_anchor_flux": np.nan,
        "det_anchor_sep_ref": np.nan,
        "trial_rank": 0,
        "cat_index": -1,
        "catalogue_rank_by_K": 0,
        "K": np.nan,
        "shift_arcsec": 0.0,
        "nmatch": int(candidate_zero["nmatch"]),
        "match_fraction": float(candidate_zero["match_fraction"]),
        "brightest_matched": bool(candidate_zero["brightest_matched"]),
        "rms_arcsec": float(candidate_zero.get("rms_arcsec", np.nan)),
        "max_sep_arcsec": float(candidate_zero.get("max_sep_arcsec", np.nan)),
        "sum_sep_arcsec": float(candidate_zero.get("sum_sep_arcsec", np.nan)),
        "reason": "zero_shift_" + candidate_zero["reason"],
    })

    best_any = candidate_zero.copy()

    for det_anchor_rank, det_anchor_index in enumerate(det_anchor_order, start=1):

        trial_order = build_anchor_trial_order(
            df_det=df_det,
            df_cat=df_cat,
            bright_anchor_mag_range=bright_anchor_mag_range,
            det_anchor_index=det_anchor_index,
        )

        for trial_rank, i_cat in enumerate(trial_order, start=1):
            cat_row = df_cat.iloc[i_cat]

            shift_ra_deg = cat_row["ra_deg"] - det_ra[det_anchor_index]
            shift_dec_deg = cat_row["dec_deg"] - det_dec[det_anchor_index]

            total_shift_arcsec = np.hypot(shift_ra_deg * 3600.0, shift_dec_deg * 3600.0)

            if total_shift_arcsec > max_total_shift_arcsec:
                debug_rows.append({
                    "det_anchor_rank": det_anchor_rank,
                    "det_anchor_index": int(det_anchor_index),
                    "det_anchor_flux": float(det_flux[det_anchor_index]),
                    "det_anchor_sep_ref": float(det_sep_ref[det_anchor_index]),
                    "trial_rank": trial_rank,
                    "cat_index": int(i_cat),
                    "catalogue_rank_by_K": int(i_cat + 1),
                    "K": float(cat_row["K"]),
                    "shift_arcsec": float(total_shift_arcsec),
                    "nmatch": 0,
                    "match_fraction": 0.0,
                    "brightest_matched": False,
                    "rms_arcsec": np.nan,
                    "max_sep_arcsec": np.nan,
                    "sum_sep_arcsec": np.nan,
                    "reason": f"shift>{max_total_shift_arcsec:.2f}\"",
                })
                continue

            candidate = evaluate_shift_candidate(
                det_ra=det_ra,
                det_dec=det_dec,
                det_flux=det_flux,
                coords_cat=coords_cat,
                cat_K_array=cat_K_array,
                shift_ra_deg=shift_ra_deg,
                shift_dec_deg=shift_dec_deg,
                match_radius_arcsec=match_radius_arcsec,
                min_nmatch=min_nmatch,
                min_match_fraction=min_match_fraction,
                flux_tolerance_ratio=flux_tolerance_ratio,
                require_brightest_match=True,
            )

            candidate["anchor_cat_rank"] = i_cat + 1
            candidate["anchor_trial_rank"] = trial_rank
            candidate["det_anchor_rank"] = det_anchor_rank
            candidate["det_anchor_index"] = int(det_anchor_index)
            candidate["det_anchor_flux"] = float(det_flux[det_anchor_index])
            candidate["det_anchor_sep_ref_arcsec"] = float(det_sep_ref[det_anchor_index])
            candidate["anchor_cat_K"] = float(cat_row["K"])

            # Require that the chosen detected anchor is matched
            if not candidate["matched"][det_anchor_index]:
                candidate["accepted"] = False
                candidate["reason"] = "chosen_det_anchor_not_matched"

            debug_rows.append({
                "det_anchor_rank": det_anchor_rank,
                "det_anchor_index": int(det_anchor_index),
                "det_anchor_flux": float(det_flux[det_anchor_index]),
                "det_anchor_sep_ref": float(det_sep_ref[det_anchor_index]),
                "trial_rank": trial_rank,
                "cat_index": int(i_cat),
                "catalogue_rank_by_K": int(i_cat + 1),
                "K": float(cat_row["K"]),
                "shift_arcsec": float(total_shift_arcsec),
                "nmatch": int(candidate["nmatch"]),
                "match_fraction": float(candidate["match_fraction"]),
                "brightest_matched": bool(candidate["brightest_matched"]),
                "rms_arcsec": float(candidate.get("rms_arcsec", np.nan)),
                "max_sep_arcsec": float(candidate.get("max_sep_arcsec", np.nan)),
                "sum_sep_arcsec": float(candidate.get("sum_sep_arcsec", np.nan)),
                "reason": candidate["reason"],
            })

            if best_any is None:
                best_any = candidate.copy()
            else:
                if candidate["nmatch"] > best_any["nmatch"]:
                    best_any = candidate.copy()
                elif candidate["nmatch"] == best_any["nmatch"]:
                    if candidate.get("sum_sep_arcsec", np.inf) < best_any.get("sum_sep_arcsec", np.inf):
                        best_any = candidate.copy()
                    elif candidate.get("sum_sep_arcsec", np.inf) == best_any.get("sum_sep_arcsec", np.inf):
                        if candidate.get("rms_arcsec", np.inf) < best_any.get("rms_arcsec", np.inf):
                            best_any = candidate.copy()

            if not candidate["accepted"]:
                continue

            if best is None:
                best = candidate.copy()
            else:
                if candidate["nmatch"] > best["nmatch"]:
                    best = candidate.copy()
                elif candidate["nmatch"] == best["nmatch"]:
                    if candidate["sum_sep_arcsec"] < best["sum_sep_arcsec"]:
                        best = candidate.copy()
                    elif candidate["sum_sep_arcsec"] == best["sum_sep_arcsec"]:
                        if candidate["rms_arcsec"] < best["rms_arcsec"]:
                            best = candidate.copy()

    # ---------------------------------------------------------
    # Final comparison against zero-shift solution
    # ---------------------------------------------------------
    if candidate_zero["accepted"]:
        if best is None:
            print("  No shifted solution beats the original WCS. Keeping zero shift.")
            best = candidate_zero.copy()
        else:
            rms_zero = candidate_zero.get("rms_arcsec", np.inf)
            rms_best = best.get("rms_arcsec", np.inf)

            improvement = rms_zero - rms_best

            # Keep zero shift only if it is truly comparable or better
            if (
                candidate_zero["nmatch"] > best["nmatch"]
                or (
                    candidate_zero["nmatch"] == best["nmatch"]
                    and improvement < MIN_RMS_IMPROVEMENT_ARCSEC
                )
            ):
                print(
                    "  Zero-shift solution retained "
                    f"(Delta RMS={improvement:.3f}\", "
                    f"Nmatch zero={candidate_zero['nmatch']}, "
                    f"Nmatch shifted={best['nmatch']})."
                )
                best = candidate_zero.copy()

    df_debug = pd.DataFrame(debug_rows)

    if verbose and len(df_debug) > 0:
        print("\n  Trial summary:")
        print(df_debug.to_string(index=False))

    return best, best_any, df_debug


def apply_shift_to_hdul_wcs(hdul: fits.HDUList, shift_ra_deg: float, shift_dec_deg: float) -> fits.HDUList:
    """
    Return a corrected copy of the HDUList with shifted CRVAL1/CRVAL2 in ext 1.
    """
    hdul_new = fits.HDUList([hdu.copy() for hdu in hdul])

    hdr = hdul_new[1].header
    hdr["CRVAL1"] = float(hdr.get("CRVAL1", 0.0)) + shift_ra_deg
    hdr["CRVAL2"] = float(hdr.get("CRVAL2", 0.0)) + shift_dec_deg
    hdr["HISTORY"] = "=" * 50
    hdr["HISTORY"] = f"Astrometry corrected on {now()}"
    hdr["HISTORY"] = f"Shift applied: dRA={shift_ra_deg*3600:.3f} arcsec"
    hdr["HISTORY"] = f"Shift applied: dDec={shift_dec_deg*3600:.3f} arcsec"

    if len(hdul_new) > 2 and hdul_new[2].data is not None and hdul_new[1].data is not None:
        hdul_new[2].header["EXTNAME"] = "STAT"
        hdul_new[1].header["EXTNAME"] = "DATA"

    return hdul_new


def apply_shift_to_collapsed_fits(collapsed_path: Path, shift_ra_deg: float, shift_dec_deg: float, output_path: Path) -> None:
    with fits.open(collapsed_path) as hdul:
        hdul_new = fits.HDUList([hdu.copy() for hdu in hdul])
        hdr = hdul_new[1].header
        hdr["CRVAL1"] = float(hdr.get("CRVAL1", 0.0)) + shift_ra_deg
        hdr["CRVAL2"] = float(hdr.get("CRVAL2", 0.0)) + shift_dec_deg
        hdr["HISTORY"] = "=" * 50
        hdr["HISTORY"] = f"Astrometry corrected on {now()}"
        hdul_new.writeto(output_path, overwrite=True)


def make_initial_diagnostic_plot(
    image_clean: np.ndarray,
    wcs_celestial: WCS,
    df_det: pd.DataFrame,
    df_catalogue_local: pd.DataFrame,
    ref_coord: SkyCoord,
    output_path: Path,
    title: str,
):
    """
    Plot the original collapsed IFU image with:
    - detected KMOS stars
    - local catalogue stars
    - header reference position
    """
    coords_cat = SkyCoord(
        ra=df_catalogue_local["ra_deg"].to_numpy() * u.deg,
        dec=df_catalogue_local["dec_deg"].to_numpy() * u.deg
    )
    x_cat, y_cat = wcs_celestial.world_to_pixel(coords_cat)

    x_ref, y_ref = wcs_celestial.world_to_pixel(ref_coord)

    fig = plt.figure(figsize=(9, 7))
    ax = plt.subplot(projection=wcs_celestial)

    finite = image_clean[np.isfinite(image_clean)]
    vmin = np.nanpercentile(finite, 5) if finite.size else None
    vmax = np.nanpercentile(finite, 99.5) if finite.size else None

    im = ax.imshow(
        image_clean,
        origin="lower",
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    if len(df_det) > 0:
        positions = np.transpose((df_det["x_pix"].to_numpy(), df_det["y_pix"].to_numpy()))
        apertures = CircularAperture(positions, r=2.5)
        apertures.plot(color="cyan", lw=1.8, ax=ax)

        for i, row in df_det.iterrows():
            ax.text(
                row["x_pix"] + 0.8,
                row["y_pix"] + 0.8,
                f"D{i+1}",
                color="white",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.7)
            )

    ax.scatter(
        x_cat, y_cat,
        s=45,
        color="deepskyblue",
        marker="x",
        linewidths=1.3,
        zorder=4,
    )

    for i, (xg, yg, k_mag) in enumerate(zip(x_cat, y_cat, df_catalogue_local["K"].to_numpy())):
        ax.text(
            xg + 0.8,
            yg + 0.8,
            f"C{i+1}:{k_mag:.1f}",
            color="cyan",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.7)
        )

    ax.scatter(
        [x_ref], [y_ref],
        s=80,
        color="lime",
        marker="+",
        linewidths=2.0,
        zorder=5,
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("R.A.")
    ax.set_ylabel("Dec.")
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
    cbar.set_label("Collapsed flux")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_shifted_diagnostic_plot(
    image_clean: np.ndarray,
    wcs_celestial: WCS,
    df_det: pd.DataFrame,
    df_catalogue_local: pd.DataFrame,
    trial_solution: dict,
    output_path: Path,
    title: str,
):
    """
    Plot IFU detections shifted by the trial solution and overplot local catalogue.
    Useful even when the trial was rejected.
    """
    coords_cat = SkyCoord(
        ra=df_catalogue_local["ra_deg"].to_numpy() * u.deg,
        dec=df_catalogue_local["dec_deg"].to_numpy() * u.deg
    )
    x_cat, y_cat = wcs_celestial.world_to_pixel(coords_cat)

    det_ra = df_det["ra_deg"].to_numpy(dtype=float)
    det_dec = df_det["dec_deg"].to_numpy(dtype=float)

    shifted_ra = det_ra + trial_solution["shift_ra_deg"]
    shifted_dec = det_dec + trial_solution["shift_dec_deg"]

    coords_shifted = SkyCoord(ra=shifted_ra * u.deg, dec=shifted_dec * u.deg)
    x_shift, y_shift = wcs_celestial.world_to_pixel(coords_shifted)

    fig = plt.figure(figsize=(9, 7))
    ax = plt.subplot(projection=wcs_celestial)

    finite = image_clean[np.isfinite(image_clean)]
    vmin = np.nanpercentile(finite, 5) if finite.size else None
    vmax = np.nanpercentile(finite, 99.5) if finite.size else None

    im = ax.imshow(
        image_clean,
        origin="lower",
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    # Original detections
    ax.scatter(
        df_det["x_pix"].to_numpy(),
        df_det["y_pix"].to_numpy(),
        s=35,
        color="yellow",
        marker="o",
        facecolors="none",
        linewidths=1.2,
        label="Original KMOS"
    )

    # Shifted detections
    ax.scatter(
        x_shift, y_shift,
        s=45,
        color="cyan",
        marker="o",
        facecolors="none",
        linewidths=1.8,
        label="Shifted KMOS"
    )

    # Catalogue
    ax.scatter(
        x_cat, y_cat,
        s=45,
        color="deepskyblue",
        marker="x",
        linewidths=1.3,
        label="Catalogue"
    )

    matched = trial_solution.get("matched", np.zeros(len(df_det), dtype=bool))
    idx = trial_solution.get("idx", np.zeros(len(df_det), dtype=int))

    for i in range(len(df_det)):
        label = f"D{i+1}"
        colour = "lime" if matched[i] else "white"
        ax.text(
            x_shift[i] + 0.8,
            y_shift[i] + 0.8,
            label,
            color=colour,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.7)
        )

        if matched[i]:
            j = int(idx[i])
            ax.plot(
                [x_shift[i], x_cat[j]],
                [y_shift[i], y_cat[j]],
                color="lime",
                lw=1.2,
                alpha=0.8
            )

    for i, (xg, yg, k_mag) in enumerate(zip(x_cat, y_cat, df_catalogue_local["K"].to_numpy())):
        ax.text(
            xg + 0.8,
            yg + 0.8,
            f"C{i+1}:{k_mag:.1f}",
            color="cyan",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.7)
        )

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("R.A.")
    ax.set_ylabel("Dec.")
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
    cbar.set_label("Collapsed flux")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_trial_summary_plot(
    df_debug: pd.DataFrame,
    output_path: Path,
    title: str,
):
    """
    Plot trial rank versus shift, nmatch, RMS.
    """
    if len(df_debug) == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    x = df_debug["trial_rank"].to_numpy(dtype=float)

    shift = df_debug["shift_arcsec"].to_numpy(dtype=float)
    nmatch = df_debug["nmatch"].to_numpy(dtype=float)
    rms = df_debug["rms_arcsec"].to_numpy(dtype=float)

    axes[0].plot(x, shift, marker="o")
    axes[0].set_ylabel("Shift (arcsec)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, nmatch, marker="o")
    axes[1].set_ylabel("N match")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x, rms, marker="o")
    axes[2].set_ylabel("RMS (arcsec)")
    axes[2].set_xlabel("Trial rank")
    axes[2].grid(alpha=0.3)

    # Mark accepted trials
    accepted = df_debug["reason"].astype(str).to_numpy() == "accepted"
    if np.any(accepted):
        axes[2].scatter(
            x[accepted],
            rms[accepted],
            s=40,
            marker="s",
            color="lime",
            zorder=5,
            label="accepted"
        )
        axes[2].legend(loc="best", fontsize=8)

    # Summary text
    n_total = len(df_debug)
    n_acc = int(np.sum(accepted))
    unique_reasons = df_debug["reason"].astype(str).value_counts().to_dict()

    summary_text = (
        f"Trials: {n_total}\n"
        f"Accepted: {n_acc}\n"
        f"Top reasons:\n" +
        "\n".join([f"{k}: {v}" for k, v in list(unique_reasons.items())[:5]])
    )

    axes[2].text(
        0.02, 0.98,
        summary_text,
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9)
    )

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


# =============================================================================
# Main OB processing function
# =============================================================================
def run_one_ob(ob_name: str, base_dir: Path, df_catalogue: pd.DataFrame) -> list:
    """
    Process a single OB and return a list of summary dicts (one per IFU).
    This function is safe to call inside a worker process.
    """
    print_header(f"START KMOS ASTROMETRY FOR {ob_name} (base_dir={base_dir})")

    paths = get_ob_paths(ob_name, base_dir)

    NAME = paths["NAME"]
    OB_DIR = paths["OB_DIR"]
    IFU_ROOT = paths["IFU_ROOT"]
    COLLAPSED_OUTPUT = paths["COLLAPSED_OUTPUT"]
    CORRECTED_CUBE_OUTPUT = paths["CORRECTED_CUBE_OUTPUT"]
    CORRECTED_COLLAPSED_OUTPUT = paths["CORRECTED_COLLAPSED_OUTPUT"]
    CHECK_PLOT_OUTPUT = paths["CHECK_PLOT_OUTPUT"]
    DIAGNOSTIC_PLOT_OUTPUT = paths["DIAGNOSTIC_PLOT_OUTPUT"]
    FINAL_CHECK_PLOT_OUTPUT = paths["FINAL_CHECK_PLOT_OUTPUT"]

    fits_files = sorted(IFU_ROOT.glob(INPUT_PATTERN))
    if len(fits_files) == 0:
        print(f"No FITS files found in {IFU_ROOT} with pattern {INPUT_PATTERN}")
        return []

    print(f"IFU cubes found in {ob_name}: {len(fits_files)}")

    summary_rows = []

    for i, fits_path in enumerate(fits_files, 1):
        print_header(f"[{ob_name} | {i}/{len(fits_files)}] {fits_path.name}")

        try:
            with fits.open(fits_path) as hdul:
                cube = hdul[1].data
                hdr0 = hdul[0].header
                hdr1 = hdul[1].header

                wcs_celestial = WCS(hdr1).celestial
                wcs_spectral = WCS(hdr1).sub(["spectral"])

                ref_coord = get_reference_coord_from_header(hdr1)
                df_local_cat = select_local_catalogue(
                    df_catalogue=df_catalogue,
                    centre=ref_coord,
                    radius_arcsec=CATALOGUE_SEARCH_RADIUS_ARCSEC,
                    k_limit=REFERENCE_K_LIMIT,
                )

                print(f"  Local catalogue stars: {len(df_local_cat)} within {CATALOGUE_SEARCH_RADIUS_ARCSEC:.2f}\"")

                if len(df_local_cat) == 0:
                    print("  No local catalogue stars. Skipping IFU.")
                    continue

                image_clean, wave_idx = collapse_cube(
                    cube=cube,
                    wcs_spectral=wcs_spectral,
                    edge_trim_fraction=EDGE_TRIM_FRACTION
                )

                if SAVE_COLLAPSED_FITS:
                    collapsed_path = save_collapsed_image_as_fits(
                        hdul=hdul,
                        image_clean=image_clean,
                        input_path=fits_path,
                        output_dir=COLLAPSED_OUTPUT,
                        wave_idx=wave_idx,
                        overwrite=True,
                    )
                    print(f"  Collapsed FITS saved: {collapsed_path.name}")
                else:
                    collapsed_path = None

                df_det = detect_sources_in_image(
                    image_clean=image_clean,
                    wcs_celestial=wcs_celestial,
                    hdr0=hdr0,
                    hdr1=hdr1,
                    threshold_sigma=KARMA_THRESHOLD_SIGMA,
                    min_separation_arcsec=KARMA_MIN_SEPARATION_ARCSEC,
                )

                df_det = filter_edge_sources(
                    df_det=df_det,
                    image_shape=image_clean.shape,
                    edge_pixels=EDGE_EXCLUSION_PIXELS,
                )

                print(f"  IFU detections: {len(df_det)}")
                if len(df_det) == 0:
                    print("  No stars detected. Skipping IFU.")
                    continue

                print("\n  Detected IFU stars:")
                print(df_det[["flux", "ra_deg", "dec_deg"]].to_string(index=True))

                print("\n  Local catalogue stars:")
                print(df_local_cat[["K", "ra_deg", "dec_deg", "sep_ref_arcsec"]].to_string(index=True))

                diagnostic_base = fits_path.stem

                make_initial_diagnostic_plot(
                    image_clean=image_clean,
                    wcs_celestial=wcs_celestial,
                    df_det=df_det,
                    df_catalogue_local=df_local_cat,
                    ref_coord=ref_coord,
                    output_path=DIAGNOSTIC_PLOT_OUTPUT / f"{diagnostic_base}_initial_overlay.png",
                    title=f"{fits_path.name} | Initial overlay",
                )

                best, best_any, df_debug = solve_astrometry_for_ifu(
                    df_det=df_det,
                    df_cat=df_local_cat,
                    ref_coord=ref_coord,
                    match_radius_arcsec=MATCH_RADIUS_ARCSEC,
                    max_total_shift_arcsec=MAX_TOTAL_SHIFT_ARCSEC,
                    min_nmatch=MIN_NMATCH,
                    min_match_fraction=MIN_MATCH_FRACTION,
                    flux_tolerance_ratio=FLUX_TOLERANCE_RATIO,
                    bright_anchor_mag_range=BRIGHT_ANCHOR_MAG_RANGE,
                )

                if df_debug is not None and len(df_debug) > 0:
                    df_debug.to_csv(
                        DIAGNOSTIC_PLOT_OUTPUT / f"{diagnostic_base}_trial_summary.csv",
                        index=False
                    )

                    make_trial_summary_plot(
                        df_debug=df_debug,
                        output_path=DIAGNOSTIC_PLOT_OUTPUT / f"{diagnostic_base}_trial_summary.png",
                        title=f"{fits_path.name} | Trial summary",
                    )

                if best_any is not None:
                    make_shifted_diagnostic_plot(
                        image_clean=image_clean,
                        wcs_celestial=wcs_celestial,
                        df_det=df_det,
                        df_catalogue_local=df_local_cat,
                        trial_solution=best_any,
                        output_path=DIAGNOSTIC_PLOT_OUTPUT / f"{diagnostic_base}_best_trial_overlay.png",
                        title=(
                            f"{fits_path.name} | Best tested trial | "
                            f"reason={best_any.get('reason', 'unknown')} | "
                            f"Nmatch={best_any.get('nmatch', 0)}"
                        ),
                    )

                if best is None:
                    print("  No valid astrometric solution found for this IFU.")
                    continue

                print(
                    f"  Best solution: "
                    f"Nmatch={best['nmatch']} | "
                    f"RMS={best['rms_arcsec']:.3f}\" | "
                    f"dRA={best['shift_ra_deg']*3600:.3f}\" | "
                    f"dDec={best['shift_dec_deg']*3600:.3f}\" | "
                    f"anchor rank by K={best['anchor_cat_rank']} | "
                    f"trial rank={best.get('anchor_trial_rank', np.nan)} | "
                    f"anchor K={best['anchor_cat_K']:.2f}"
                )

                corrected_hdul = apply_shift_to_hdul_wcs(
                    hdul=hdul,
                    shift_ra_deg=best["shift_ra_deg"],
                    shift_dec_deg=best["shift_dec_deg"],
                )

                corrected_cube_path = CORRECTED_CUBE_OUTPUT / f"{fits_path.stem}_astrocorr.fits"
                corrected_hdul.writeto(corrected_cube_path, overwrite=True)
                print(f"  Corrected cube saved: {corrected_cube_path.name}")

                final_check_path = FINAL_CHECK_PLOT_OUTPUT / f"{corrected_cube_path.stem}_final_check.png"
                make_final_check_plot(
                    corrected_cube_path=corrected_cube_path,
                    df_catalogue_local=df_local_cat,
                    output_path=final_check_path,
                    threshold_sigma=KARMA_THRESHOLD_SIGMA,
                    min_separation_arcsec=KARMA_MIN_SEPARATION_ARCSEC,
                )
                print(f"  Final check plot saved: {final_check_path.name}")

                if collapsed_path is not None:
                    corrected_collapsed_path = CORRECTED_COLLAPSED_OUTPUT / f"{collapsed_path.stem}_astrocorr.fits"
                    apply_shift_to_collapsed_fits(
                        collapsed_path=collapsed_path,
                        shift_ra_deg=best["shift_ra_deg"],
                        shift_dec_deg=best["shift_dec_deg"],
                        output_path=corrected_collapsed_path,
                    )
                    print(f"  Corrected collapsed saved: {corrected_collapsed_path.name}")
                else:
                    corrected_collapsed_path = None

                if SAVE_CHECK_PLOTS:
                    make_shifted_diagnostic_plot(
                        image_clean=image_clean,
                        wcs_celestial=wcs_celestial,
                        df_det=df_det,
                        df_catalogue_local=df_local_cat,
                        trial_solution=best,
                        output_path=CHECK_PLOT_OUTPUT / f"{fits_path.stem}_accepted_solution.png",
                        title=(
                            f"{fits_path.name} | Accepted solution | "
                            f"Nmatch={best['nmatch']} | RMS={best['rms_arcsec']:.3f}\""
                        ),
                    )

                summary_rows.append({
                    "ob_name": ob_name,
                    "input_file": fits_path.name,
                    "n_detected": len(df_det),
                    "n_cat_local": len(df_local_cat),
                    "n_match": best["nmatch"],
                    "match_fraction": best["match_fraction"],
                    "fit_rms_arcsec": best["rms_arcsec"],
                    "shift_ra_arcsec": best["shift_ra_deg"] * 3600.0,
                    "shift_dec_arcsec": best["shift_dec_deg"] * 3600.0,
                    "total_shift_arcsec": best["total_shift_arcsec"],
                    "anchor_cat_rank": best["anchor_cat_rank"],
                    "anchor_trial_rank": best.get("anchor_trial_rank", np.nan),
                    "anchor_cat_K": best["anchor_cat_K"],
                    "corrected_cube": corrected_cube_path.name,
                    "corrected_collapsed": corrected_collapsed_path.name if corrected_collapsed_path else "",
                })

        except Exception as exc:
            print(f"  ERROR processing {fits_path.name}: {exc}")
            traceback.print_exc()

    if len(summary_rows) > 0:
        df_summary = pd.DataFrame(summary_rows)
        summary_path = CORRECTED_CUBE_OUTPUT / "astrometry_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print_header(f"PIPELINE SUMMARY FOR {ob_name}")
        print(df_summary.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")
    else:
        print_header(f"PIPELINE SUMMARY FOR {ob_name}")
        print("No IFUs were successfully corrected.")

    return summary_rows


# =============================================================================
# Worker-side globals (populated by the pool initializer)
# =============================================================================
_WORKER_CATALOGUE: Optional[pd.DataFrame] = None


def _init_worker(catalogue_path_str: str) -> None:
    """
    Load the catalogue once per worker process.
    This avoids re-pickling a large DataFrame for every OB task.
    """
    global _WORKER_CATALOGUE
    _WORKER_CATALOGUE = read_catalogue_fits(Path(catalogue_path_str))


def _process_ob_task(ob_name: str, base_dir_str: str) -> list:
    """
    Thin wrapper that runs run_one_ob using the worker-local catalogue.
    Returns the summary rows so the parent process can aggregate them.
    """
    assert _WORKER_CATALOGUE is not None, "Worker catalogue was not initialised."
    return run_one_ob(ob_name, Path(base_dir_str), _WORKER_CATALOGUE)


# =============================================================================
# Main function: loops over concatenations and discovers OB subdirectories
# =============================================================================
def main() -> None:
    print_header("START MULTI-CONCATENATION KMOS ASTROMETRY (PARALLEL)")
    print(f"Workers: {N_WORKERS}")
    print("  (override with:  KMOS_N_WORKERS=N python3 this_script.py)")

    # -----------------------------------------------------------------
    # 1) Discover all (ob_name, sky_tweak_dir) tasks in the parent
    #    process. This makes path errors visible before spawning workers.
    # -----------------------------------------------------------------
    tasks: list[tuple[str, Path]] = []

    for con_name in CONS:
        print_header(f"DISCOVERING OBs FOR CONCATENATION: {con_name}")

        dummy_sky = Path(BASE_DIR_TEMPLATE.format(
            pointing=pointing, con_name=con_name, ob_name="dummy"
        ))
        con_dir = dummy_sky.parent.parent  # dummy_sky -> OB dir -> con dir

        if not con_dir.exists():
            print(f"  Concatenation directory not found: {con_dir}")
            continue

        ob_dirs = sorted([
            d for d in con_dir.iterdir()
            if d.is_dir() and d.name.startswith("OB")
        ])

        if not ob_dirs:
            print(f"  No OB folders found inside {con_dir}")
            continue

        print(f"  Found {len(ob_dirs)} OB(s): {[d.name for d in ob_dirs]}")

        for ob_dir in ob_dirs:
            ob_name = ob_dir.name
            base_dir = Path(BASE_DIR_TEMPLATE.format(
                pointing=pointing, con_name=con_name, ob_name=ob_name
            ))
            if not base_dir.exists():
                print(f"  WARNING: sky_tweak folder missing for {ob_name}, skipping")
                continue
            tasks.append((ob_name, base_dir))

    if not tasks:
        print_header("NO TASKS TO RUN")
        return

    print_header(f"TOTAL OBs TO PROCESS: {len(tasks)}")

    # -----------------------------------------------------------------
    # 2) Execute tasks in parallel.
    #    The catalogue is loaded once per worker via the initializer.
    # -----------------------------------------------------------------
    all_summary_rows: list[dict] = []
    failed_tasks: list[tuple[str, str]] = []

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_init_worker,
        initargs=(str(GNS_TXT_PATH),),
    ) as executor:
        futures = {
            executor.submit(_process_ob_task, ob_name, str(base_dir)): (ob_name, base_dir)
            for ob_name, base_dir in tasks
        }

        for future in as_completed(futures):
            ob_name, base_dir = futures[future]
            try:
                rows = future.result()
                if rows:
                    all_summary_rows.extend(rows)
                print(f"[DONE] {ob_name}")
            except Exception as exc:
                print_header(f"FATAL ERROR IN {ob_name}")
                print(exc)
                traceback.print_exc()
                failed_tasks.append((ob_name, str(exc)))

    # -----------------------------------------------------------------
    # 3) Write a global summary across all OBs.
    # -----------------------------------------------------------------
    if all_summary_rows:
        df_all = pd.DataFrame(all_summary_rows)
        global_summary_path = GNS_TXT_PATH.parent / "global_astrometry_summary.csv"
        df_all.to_csv(global_summary_path, index=False)
        print_header("GLOBAL PIPELINE SUMMARY")
        print(df_all.to_string(index=False))
        print(f"\nGlobal summary saved to: {global_summary_path}")

    if failed_tasks:
        print_header("TASKS THAT FAILED")
        for name, err in failed_tasks:
            print(f"  - {name}: {err}")

    print_header("ALL CONCATENATIONS FINISHED")


if __name__ == "__main__":
    main()