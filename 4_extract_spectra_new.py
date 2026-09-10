#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KMOS IFU stellar spectrum extraction from a catalogue.
Local annulus sky subtraction with proper error propagation.

Adapted for concatenation folder structure.
All output files for a given OB are placed in a single folder
(OB_dir/spectra_out). File names are prefixed with the IFU name,
e.g., COMBINE_SKY_TWEAK_173_astrocorr_ra266p561000_decm28p850430.txt,
so that sources from different IFUs never overwrite each other.

Added parameter SKY_ANNULUS: set to False to disable sky subtraction.
"""

from pathlib import Path
import warnings
import traceback

import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import splrep, splev

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.stats import sigma_clip


# ============================================================
# Global configuration
# ============================================================

pointing = "P3"
CONS = ["con3"]#, "con5"]#, "con6"]

BASE_DIR_TEMPLATE = (
    "/home/data/KMOS/PILOT/reduced/P115/{pointing}/{con_name}/{ob_name}/sky_tweak/"
)
BASE_DIR = None

CATALOGUE_FILE = Path("/home/data/KMOS/PILOT/GNS_cat/FINAL_EAST_CENTRAL_WEST_VIRAC.fits")

INPUT_PATTERN = "*_astrocorr.fits"
LIST_FILENAME = "ifu_list.txt"
CORRECTED_DIRNAME = "corrected_fits_new"
OUTPUT_DIRNAME = "spectra_out"

# ============================================================
# User parameters
# ============================================================
EXP_TIME_RUN = (10*7)

APERTURE_NOMINAL_PSF_FRACTION = 0.6   # única apertura usada

MIN_SOURCE_FRACTION_INSIDE = 0.6   # mínimo 60% de la apertura dentro del IFU

# Sky annulus factors
SKY_ANNULUS = True           # <<<<<<<<<<< Set to False to disable sky subtraction
SKY_ANNULUS_INNER_FACTOR    = 1.5
SKY_ANNULUS_OUTER_FACTOR    = 2.3
SKY_NEIGHBOUR_EXCLUSION_FACTOR = 1.5

INCLUDE_POISSON_NOISE = True

SKY_SIGMA_CLIP_SIGMA  = 3.0
SKY_SIGMA_CLIP_MAXITERS = 5

NEIGHBOUR_EXCLUSION_FACTOR           = 1.5
SOURCE_CONTAMINATION_FACTOR          = 0.75
RECENTER_NEIGHBOUR_EXCLUSION_FACTOR  = 1.00

LOW_REJ  = 1.8
HIGH_REJ = 0.0
NITER    = 10
ORDER    = 3

MIN_SKY_PIXELS      = 10
FIG_DPI             = 180
RECENTER_BOX_HALF_SIZE = 5   # <-- 5 píxeles
K_LIMIT_EXTRACT     = 14.0
K_LIMIT_SKY         = 15.0
SIGMA_CLIP_SIGMA    = 3.0
SIGMA_CLIP_MAXITERS = 5
SOURCE_CLIP_SIGMA   = 4.0
SOURCE_CLIP_MAXITERS = 3
SOURCE_CLIP_MIN_PIXELS = 8
EDGE_FRACTION_TO_IGNORE = 0.10
Y_PERCENTILE_LOW    = 0.05
Y_PERCENTILE_HIGH   = 99.95
Y_PADDING_FRACTION  = 0.70
PLOT_WAVE_MIN       = 2.07
PLOT_WAVE_MAX       = 2.40
WHITE_LIGHT_MIN     = 2.05
WHITE_LIGHT_MAX     = 2.38
WHITE_CLIP_SIGMA    = 3.0
WHITE_CLIP_MAXITERS = 3
SNR_SMOOTH_WINDOW   = 101
SNR_LINE_REJECTION_SIGMA = 2.5

NOMINAL_AP_COLOR = "black"

PROP_ERR_RELIABILITY_THRESHOLD = 0.80
SKY_OVERSUBTRACTION_WARN_RATIO = 0.95

MAX_RV_KMS = 300.0
C_KMS = 299792.458
EXTRA_LINE_MASK_WIDTH_MICRON = 0.0015

EMISSION_LINES = [
    (2.0404, r'[Al IX]'),
    (2.0587, r'He I'),
    (2.0888, r'Fe II'),
    (2.1127, r'He I'),
    (2.1218, r'H$_2$ 1–0 S(1)'),
    (2.1367, r'Fe II'),
    (2.1542, r'H$_2$ 2–1 S(2)'),
    (2.1661, r'Br$\gamma$'),
    (2.1888, r'He II (10–7)'),
    (2.2235, r'H$_2$ 1–0 S(0)'),
    (2.2247, r'Fe II'),
    (2.2477, r'H$_2$ 2–1 S(1)'),
    (2.2935, r'$^{12}$CO(2,0)'),
    (2.3213, r'[Ca VIII]'),
    (2.3227, r'$^{12}$CO(3,1)'),
    (2.3525, r'$^{12}$CO(4,2)'),
    (2.3835, r'$^{12}$CO(5,3)'),
]

def get_line_color(label):
    if 'CO' in label:
        return 'red'
    if 'H$_2$' in label or 'H2' in label:
        return 'green'
    if 'Br$\\gamma$' in label:
        return 'blue'
    if 'He I' in label:
        return 'magenta'
    if 'He II' in label:
        return 'darkmagenta'
    if 'Fe II' in label:
        return 'orange'
    if 'Al IX' in label or 'Ca VIII' in label:
        return 'purple'
    return 'gray'

def create_line_free_mask(wave, emission_lines, rv_max_kms, c_kms, extra_width_micron=0.0):
    mask = np.ones_like(wave, dtype=bool)
    doppler_factor = rv_max_kms / c_kms
    for w0, _ in emission_lines:
        dw_doppler = w0 * doppler_factor
        half_width = dw_doppler + extra_width_micron
        mask &= (np.abs(wave - w0) > half_width)
    mask &= (wave > 2.10) & (wave < 2.38)
    return mask

# ============================================================
# Helper functions
# ============================================================

def create_ifu_list_file(fits_dir, list_file, pattern="*_astrocorr.fits"):
    fits_files = sorted(Path(fits_dir).glob(pattern))
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in {fits_dir} with pattern {pattern}")
    Path(list_file).parent.mkdir(parents=True, exist_ok=True)
    with open(list_file, "w") as f:
        for path in fits_files:
            f.write(path.name + "\n")
    print(f"Created IFU list: {list_file}")
    print(f"Number of FITS files: {len(fits_files)}")

def format_coord_for_filename(value, ndigits=6):
    s = f"{float(value):.{ndigits}f}"
    return s.replace("-", "m").replace(".", "p")

def make_source_basename(ra_deg, dec_deg):
    return f"ra{format_coord_for_filename(ra_deg)}_dec{format_coord_for_filename(dec_deg)}"

def get_ob_paths(ob_name):
    ob_dir = BASE_DIR / f"res_{ob_name}"
    return {
        "name": ob_name,
        "base_dir": BASE_DIR,
        "ob_dir": ob_dir,
        "corrected_dir": ob_dir / CORRECTED_DIRNAME,
        "list_file": ob_dir / CORRECTED_DIRNAME / LIST_FILENAME,
        "output_base": ob_dir / OUTPUT_DIRNAME,
    }

def get_arm_number(header):
    """
    Extract the IFU arm number from the header of extension 1.
    Looks for keywords containing 'ARM' followed by a digit (e.g., ARM2)
    and returns the integer.
    """
    pattern = re.compile(r'ARM(\d+)', re.IGNORECASE)
    for key in header:
        match = pattern.search(key)
        if match:
            return int(match.group(1))
    raise ValueError("Could not determine ARM number from header (extension 1).")

# ============================================================
# I/O utilities
# ============================================================

def load_catalogue(catalogue_file):
    """
    Load catalogue from a FITS binary table.
    Detects RA, Dec, and magnitude columns, and returns an astropy Table
    with standardized column names: ra, dec, J, dJ, H, dH, K, dK.
    """
    with fits.open(catalogue_file) as hdul:
        table_hdu = None
        for hdu in hdul:
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                table_hdu = hdu
                break
        if table_hdu is None:
            raise ValueError("No table HDU found in catalogue FITS file.")

        tab = Table(table_hdu.data)
        lower_to_original = {c.lower(): c for c in tab.colnames}
        colnames_lower = list(lower_to_original.keys())

        def find_col(possible_names):
            for name in possible_names:
                if name in colnames_lower:
                    return lower_to_original[name]
            return None

        ra_col = find_col(['ra', 'raj2000', 'ra_deg', 'alpha'])
        if ra_col is None:
            raise ValueError(f"Cannot find RA column. Available columns: {tab.colnames}")

        dec_col = find_col(['dec', 'dej2000', 'dec_deg', 'delta'])
        if dec_col is None:
            raise ValueError(f"Cannot find Dec column. Available columns: {tab.colnames}")

        mag_map = {}
        for std, possible_list in [
            ('K', ['k', 'kmag', 'k_mag', 'mag_k', 'ks', 'ksmag']),
            ('H', ['h', 'hmag', 'h_mag', 'mag_h']),
            ('J', ['j', 'jmag', 'j_mag', 'mag_j']),
        ]:
            col = find_col(possible_list)
            if col is not None:
                mag_map[std] = col

        err_map = {}
        for std, possible_list in [
            ('dK', ['dk', 'k_err', 'kmag_err', 'e_k', 'k_error']),
            ('dH', ['dh', 'h_err', 'hmag_err', 'e_h', 'h_error']),
            ('dJ', ['dj', 'j_err', 'jmag_err', 'e_j', 'j_error']),
        ]:
            col = find_col(possible_list)
            if col is not None:
                err_map[std] = col

        new_tab = Table()
        new_tab['ra'] = tab[ra_col].astype(float)
        new_tab['dec'] = tab[dec_col].astype(float)

        for std_name in ['J', 'H', 'K']:
            if std_name in mag_map:
                new_tab[std_name] = tab[mag_map[std_name]].astype(float)
            else:
                new_tab[std_name] = np.nan

        for std_name in ['dJ', 'dH', 'dK']:
            if std_name in err_map:
                new_tab[std_name] = tab[err_map[std_name]].astype(float)
            else:
                new_tab[std_name] = np.nan

        return new_tab

def select_catalogue_by_k(tab, k_limit):
    good = np.isfinite(tab["K"]) & (tab["K"] < k_limit)
    return tab[good]

def load_cube(cube_file, ext=1):
    with fits.open(cube_file) as hdul:
        cube   = hdul[ext].data
        header = hdul[ext].header
        primary_header = hdul[0].header
    if cube is None or cube.ndim != 3:
        raise ValueError(f"Extension {ext} is not a 3D cube.")
    return np.array(cube, dtype=float, copy=True), header, primary_header

def build_wavelength_axis(header, nw):
    crpix = header.get("CRPIX3")
    crval = header.get("CRVAL3")
    cdelt = header.get("CDELT3")
    if crpix is None or crval is None or cdelt is None:
        raise ValueError("Missing CRPIX3/CRVAL3/CDELT3 in FITS header.")
    return crval + (np.arange(nw) + 1 - crpix) * cdelt

def get_celestial_wcs(header):
    return WCS(header).celestial

def world_to_pixel(cel_wcs, ra_deg, dec_deg):
    x, y = cel_wcs.all_world2pix(ra_deg, dec_deg, 0)
    return np.asarray(x), np.asarray(y)

def get_pixel_scale_arcsec(header):
    w = WCS(header).celestial
    try:
        scales = w.proj_plane_pixel_scales()
        return float(abs(np.mean([s.to_value("arcsec") for s in scales])))
    except Exception:
        cdelt1 = header.get("CDELT1")
        if cdelt1 is None:
            raise ValueError("Could not determine pixel scale from WCS/header.")
        return abs(cdelt1) * 3600.0

def _safe_header_float(*values):
    for val in values:
        try:
            if val is None:
                continue
            out = float(val)
            if np.isfinite(out):
                return out
        except Exception:
            continue
    return None

def get_psf_from_primary_header(
    primary_header, ext_header, pixel_scale_arcsec,
    min_valid_arcsec=0.20, max_valid_arcsec=3.00,
    lambda_science_micron=2.20, lambda_optical_micron=0.50,
):
    candidates_arcsec = []
    accepted = []
    rejected = []
    wave_scale_opt_to_sci = (lambda_science_micron / lambda_optical_micron) ** (-0.2)

    def register_value(name, value, already_at_science_wavelength=True):
        if value is None:
            rejected.append((name, None, "missing")); return
        try:
            val = float(value)
        except Exception:
            rejected.append((name, value, "not_float")); return
        if not np.isfinite(val):
            rejected.append((name, value, "not_finite")); return
        if not already_at_science_wavelength:
            val = val * wave_scale_opt_to_sci
            name = f"{name} [scaled_to_{lambda_science_micron:.2f}um]"
        if val < min_valid_arcsec:
            rejected.append((name, val, f"<{min_valid_arcsec:.2f} arcsec")); return
        if val > max_valid_arcsec:
            rejected.append((name, val, f">{max_valid_arcsec:.2f} arcsec")); return
        candidates_arcsec.append(val); accepted.append((name, val))

    def register_sky_res(name, raw_value):
        if raw_value is None:
            rejected.append((name, None, "missing")); return
        try:
            raw = float(raw_value)
        except Exception:
            rejected.append((name, raw_value, "not_float")); return
        if not np.isfinite(raw):
            rejected.append((name, raw_value, "not_finite")); return
        interpretations = [
            (f"{name}_arcsec", raw),
            (f"{name}_div100", raw / 100.0),
            (f"{name}_div1000", raw / 1000.0),
        ]
        valid_local, invalid_local = [], []
        for label, val in interpretations:
            if val < min_valid_arcsec:
                invalid_local.append((label, val, f"<{min_valid_arcsec:.2f} arcsec"))
            elif val > max_valid_arcsec:
                invalid_local.append((label, val, f">{max_valid_arcsec:.2f} arcsec"))
            else:
                valid_local.append((label, val))
        if not valid_local:
            for item in invalid_local: rejected.append(item)
            return
        chosen_label, chosen_val = min(valid_local, key=lambda x: abs(x[1] - 1.0))
        candidates_arcsec.append(chosen_val); accepted.append((chosen_label, chosen_val))
        for label, val in valid_local:
            if label != chosen_label:
                rejected.append((label, val, "plausible_but_not_selected"))
        for item in invalid_local:
            rejected.append(item)

    register_sky_res(
        "SKY_RES",
        _safe_header_float(primary_header.get("SKY_RES"), ext_header.get("SKY_RES")),
    )
    for key_name, already_sci in [
        ("TEL_IA_FWHM",         False),
        ("TEL_IA_FWHMLIN",      False),
        ("TEL_AMBI_FWHM_START", False),
        ("TEL_AMBI_FWHM_END",   False),
    ]:
        fits_key = "HIERARCH ESO " + key_name.replace("_", " ")
        register_value(
            key_name,
            _safe_header_float(primary_header.get(fits_key), ext_header.get(fits_key)),
            already_at_science_wavelength=already_sci,
        )

    if not candidates_arcsec:
        raise ValueError(
            f"No valid PSF/seeing in headers within [{min_valid_arcsec:.2f}, {max_valid_arcsec:.2f}] arcsec."
        )

    fwhm_arcsec = float(np.median(candidates_arcsec))
    fwhm_pix    = fwhm_arcsec / pixel_scale_arcsec
    diagnostics = dict(
        adopted_arcsec=fwhm_arcsec, adopted_pix=fwhm_pix,
        n_valid=len(candidates_arcsec), accepted=accepted, rejected=rejected,
        lambda_science_micron=lambda_science_micron,
        lambda_optical_micron=lambda_optical_micron,
        wave_scale_opt_to_sci=wave_scale_opt_to_sci,
    )
    return fwhm_arcsec, fwhm_pix, diagnostics

# ============================================================
# Masks / geometry
# ============================================================

def circular_mask(shape, x0, y0, radius):
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    return (xx - x0) ** 2 + (yy - y0) ** 2 <= radius ** 2

def annulus_mask(shape, x0, y0, r_in, r_out):
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    rr2 = (xx - x0) ** 2 + (yy - y0) ** 2
    return (rr2 >= r_in ** 2) & (rr2 <= r_out ** 2)

def source_fraction_inside_ifu(x, y, radius, nx, ny):
    pad = int(np.ceil(radius)) + 2
    big_ny, big_nx = ny + 2*pad, nx + 2*pad
    x_big, y_big = x + pad, y + pad
    full_mask_big = circular_mask((big_ny, big_nx), x_big, y_big, radius)
    n_total = np.sum(full_mask_big)
    if n_total == 0:
        return 0.0
    inside_ifu = np.zeros((big_ny, big_nx), dtype=bool)
    inside_ifu[pad:pad+ny, pad:pad+nx] = True
    return np.sum(full_mask_big & inside_ifu) / n_total

def source_is_usable(x, y, radius, nx, ny, min_fraction=0.75):
    return source_fraction_inside_ifu(x, y, radius, nx, ny) >= min_fraction

def select_catalogue_sources_near_ifu(x_all, y_all, nx, ny, margin_pix):
    return (
        (x_all > -margin_pix) & (x_all < nx + margin_pix) &
        (y_all > -margin_pix) & (y_all < ny + margin_pix)
    )

def build_neighbour_exclusion_mask(shape, x_all, y_all, exclusion_radius, target_index=None):
    mask = np.zeros(shape, dtype=bool)
    for i, (x, y) in enumerate(zip(x_all, y_all)):
        if target_index is not None and i == target_index:
            continue
        if np.isfinite(x) and np.isfinite(y):
            mask |= circular_mask(shape, x, y, exclusion_radius)
    return mask

def build_source_contamination_mask(shape, x_all, y_all, contam_radius, target_index=None):
    mask = np.zeros(shape, dtype=bool)
    for i, (x, y) in enumerate(zip(x_all, y_all)):
        if target_index is not None and i == target_index:
            continue
        if np.isfinite(x) and np.isfinite(y):
            mask |= circular_mask(shape, x, y, contam_radius)
    return mask

def build_recentering_exclusion_mask(shape, x_all, y_all, exclusion_radius):
    mask = np.zeros(shape, dtype=bool)
    for x, y in zip(x_all, y_all):
        if np.isfinite(x) and np.isfinite(y):
            mask |= circular_mask(shape, x, y, exclusion_radius)
    return mask

# ============================================================
# Extraction / recentering
# ============================================================

def make_whitelight_image(cube, wave, wmin=WHITE_LIGHT_MIN, wmax=WHITE_LIGHT_MAX):
    mask = np.isfinite(wave) & (wave >= wmin) & (wave <= wmax)
    if np.sum(mask) < 10:
        collapsed = cube
    else:
        collapsed = cube[mask, :, :]
    clipped = sigma_clip(collapsed, sigma=WHITE_CLIP_SIGMA, maxiters=WHITE_CLIP_MAXITERS,
                         axis=0, masked=True)
    white = np.ma.median(clipped, axis=0).filled(np.nan)
    return np.array(white, dtype=float, copy=True)

def recenter_on_peak(image, x0, y0, half_size=3):
    ny, nx = image.shape
    x0i, y0i = int(round(x0)), int(round(y0))
    x1, x2 = max(0, x0i - half_size), min(nx, x0i + half_size + 1)
    y1, y2 = max(0, y0i - half_size), min(ny, y0i + half_size + 1)
    sub = np.array(image[y1:y2, x1:x2], dtype=float, copy=True)
    if sub.size == 0 or np.all(~np.isfinite(sub)):
        return x0, y0
    iy, ix = np.unravel_index(np.nanargmax(sub), sub.shape)
    return float(x1 + ix), float(y1 + iy)

def recenter_on_peak_excluding_neighbours(
    image, x0, y0, forbidden_mask,
    half_size=3, distance_sigma_pix=1.5, centroid_box_half_size=1,
):
    ny, nx = image.shape
    x0i, y0i = int(round(x0)), int(round(y0))
    x1, x2 = max(0, x0i - half_size), min(nx, x0i + half_size + 1)
    y1, y2 = max(0, y0i - half_size), min(ny, y0i + half_size + 1)
    sub          = np.array(image[y1:y2, x1:x2],       dtype=float, copy=True)
    sub_forbidden= np.array(forbidden_mask[y1:y2, x1:x2], dtype=bool, copy=False)
    if sub.size == 0:
        return x0, y0
    yy, xx = np.indices(sub.shape)
    xx_full, yy_full = x1 + xx, y1 + yy
    sub_work = np.array(sub, copy=True)
    sub_work[sub_forbidden] = np.nan
    good = np.isfinite(sub_work)
    if np.sum(good) == 0:
        return x0, y0
    d2     = (xx_full - x0) ** 2 + (yy_full - y0) ** 2
    weight = np.exp(-0.5 * d2 / (distance_sigma_pix ** 2))
    score  = sub_work * weight
    score[~good] = np.nan
    if np.all(~np.isfinite(score)):
        return x0, y0
    iy_peak, ix_peak = np.unravel_index(np.nanargmax(score), score.shape)
    x_peak, y_peak = float(x1 + ix_peak), float(y1 + iy_peak)

    cx1 = max(0, int(round(x_peak)) - centroid_box_half_size)
    cx2 = min(nx, int(round(x_peak)) + centroid_box_half_size + 1)
    cy1 = max(0, int(round(y_peak)) - centroid_box_half_size)
    cy2 = min(ny, int(round(y_peak)) + centroid_box_half_size + 1)
    sub_c  = np.array(image[cy1:cy2, cx1:cx2], dtype=float, copy=True)
    forb_c = np.array(forbidden_mask[cy1:cy2, cx1:cx2], dtype=bool, copy=False)
    sub_c[forb_c] = np.nan
    good_c = np.isfinite(sub_c)
    if np.sum(good_c) == 0:
        return x_peak, y_peak
    yy_c, xx_c = np.indices(sub_c.shape)
    xx_c, yy_c = cx1 + xx_c, cy1 + yy_c
    flux_c = np.array(sub_c, copy=True)
    flux_c[~good_c] = np.nan
    floor = np.nanmedian(flux_c)
    flux_c = flux_c - floor
    flux_c[~np.isfinite(flux_c)] = np.nan
    flux_c[flux_c < 0] = 0
    if np.nansum(flux_c) <= 0:
        return x_peak, y_peak
    x_cent = np.nansum(xx_c * flux_c) / np.nansum(flux_c)
    y_cent = np.nansum(yy_c * flux_c) / np.nansum(flux_c)
    return float(x_cent), float(y_cent)

def extract_aperture_spectrum(
    cube, mask, mode="sum",
    source_clip_sigma=SOURCE_CLIP_SIGMA,
    source_clip_maxiters=SOURCE_CLIP_MAXITERS,
    source_clip_min_pixels=SOURCE_CLIP_MIN_PIXELS,
):
    pixels = cube[:, mask]
    if pixels.shape[1] == 0:
        return np.full(cube.shape[0], np.nan)
    if mode == "sum":
        if pixels.shape[1] >= source_clip_min_pixels:
            clipped = sigma_clip(pixels, sigma=source_clip_sigma,
                                 maxiters=source_clip_maxiters,
                                 axis=1, masked=True)
            return np.ma.sum(clipped, axis=1).filled(np.nan)
        else:
            return np.nansum(pixels, axis=1)
    elif mode == "mean":
        return np.nanmean(pixels, axis=1)
    elif mode == "median":
        return np.nanmedian(pixels, axis=1)
    else:
        raise ValueError("mode must be 'sum', 'mean', or 'median'")

def running_median_1d(x, window):
    window = int(window)
    if window < 3:
        return x.copy()
    if window % 2 == 0:
        window += 1
    n    = len(x)
    half = window // 2
    out  = np.full(n, np.nan)
    for i in range(n):
        i1 = max(0, i - half)
        i2 = min(n, i + half + 1)
        vals = x[i1:i2]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            out[i] = np.nanmedian(vals)
    return out

def estimate_snr_from_continuum(
    wave, flux, edge_fraction=0.10, smooth_window=51,
    sigma_clip_value=3.0, max_iter=5, co_limit=2.29
):
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good_wave = np.isfinite(wave) & np.isfinite(flux) & (wave < co_limit)
    if np.sum(good_wave) < 20:
        return np.nan
    wave, flux = wave[good_wave], flux[good_wave]
    n = len(flux)
    if n < 20:
        return np.nan
    i1 = int(edge_fraction * n)
    i2 = int((1.0 - edge_fraction) * n)
    if i2 <= i1:
        return np.nan
    f = flux[i1:i2]
    f = f[np.isfinite(f)]
    if len(f) < 20:
        return np.nan
    cont  = running_median_1d(f, smooth_window)
    resid = f - cont
    mask  = np.isfinite(resid) & np.isfinite(cont)
    for _ in range(max_iter):
        r = resid[mask]
        if len(r) < 10:
            break
        med   = np.nanmedian(r)
        mad   = np.nanmedian(np.abs(r - med))
        if not np.isfinite(mad) or mad <= 0:
            break
        sigma   = 1.4826 * mad
        new_mask = mask & (np.abs(resid - med) < sigma_clip_value * sigma)
        if np.sum(new_mask) == np.sum(mask):
            break
        mask = new_mask
    if np.sum(mask) < 10:
        return np.nan
    r      = resid[mask]
    med    = np.nanmedian(r)
    noise  = 1.4826 * np.nanmedian(np.abs(r - med))
    signal = np.nanmedian(cont[mask])
    if not np.isfinite(noise) or noise <= 0:
        return np.nan
    return float(signal / noise)

def robust_snr_estimate_from_final_spectrum(
    flux, unc, edge_fraction=0.10, smooth_window=31, line_rejection_sigma=3.0
):
    n  = len(flux)
    i1 = int(edge_fraction * n)
    i2 = int((1.0 - edge_fraction) * n)
    if i2 <= i1:
        return np.nan
    f, e = flux[i1:i2], unc[i1:i2]
    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    if np.sum(good) < 10:
        return np.nan
    f, e   = f[good], e[good]
    smooth = running_median_1d(f, smooth_window)
    resid  = f - smooth
    med_res = np.nanmedian(resid)
    mad_res = np.nanmedian(np.abs(resid - med_res))
    if not np.isfinite(mad_res) or mad_res <= 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            sn = f / e
        sn = sn[np.isfinite(sn)]
        return float(np.nanmedian(sn)) if len(sn) >= 5 else np.nan
    sigma_res = 1.4826 * mad_res
    keep = np.abs(resid - med_res) < line_rejection_sigma * sigma_res
    if np.sum(keep) < 5:
        keep = np.isfinite(f) & np.isfinite(e) & (e > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sn = smooth[keep] / e[keep]
    sn = sn[np.isfinite(sn)]
    return float(np.nanmedian(sn)) if len(sn) >= 5 else np.nan

def get_white_display_limits(white):
    vals = white[np.isfinite(white)]
    if vals.size == 0:
        return 0.0, 1.0
    vmax = np.nanpercentile(vals, 99.5)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return 0.0, vmax

def save_spectrum_ascii(outfile, wave, flux, flux_err, sky_median, sky_sigma, n_ap, n_sky, flux_raw):
    good = np.isfinite(flux)
    if not np.any(good):
        return
    arr = np.column_stack([wave[good], flux[good], flux_err[good],
                           sky_median[good], sky_sigma[good], flux_raw[good]])
    header = ("wave flux flux_err sky_median sky_sigma flux_raw  "
              f"# N_ap={n_ap} N_sky={n_sky}")
    np.savetxt(outfile, arr, header=header)

def save_spectrum_fits(outfile, wave, flux, flux_err, sky_median, sky_sigma, n_ap, n_sky, flux_raw):
    good = np.isfinite(flux)
    if not np.any(good):
        return
    tab = Table()
    tab["wave"]       = wave[good]
    tab["flux"]       = flux[good]
    tab["flux_err"]   = flux_err[good]
    tab["sky_median"] = sky_median[good]
    tab["sky_sigma"]  = sky_sigma[good]
    tab["flux_raw"]   = flux_raw[good]
    tab.meta["N_ap"]  = n_ap
    tab.meta["N_sky"] = n_sky
    tab.write(outfile, overwrite=True)

def save_starkit_normalised_txt(outfile, wave_ang, flux_norm, flux_norm_err):
    data = np.column_stack([wave_ang, flux_norm, flux_norm_err])
    np.savetxt(outfile, data, header="lambda flux error_flux", fmt='%.8e')

def save_normalised_fits(outfile, wave, flux_norm, flux_norm_err):
    col1 = fits.Column(name='lambda',        format='D', array=wave)
    col2 = fits.Column(name='flux_norm',     format='D', array=flux_norm)
    col3 = fits.Column(name='flux_norm_err', format='D', array=flux_norm_err)
    hdu  = fits.BinTableHDU.from_columns([col1, col2, col3])
    hdu.writeto(outfile, overwrite=True)

# ============================================================
# Local continuum function
# ============================================================

def continuum(wave, flux, low_rej=1.8, high_rej=0.0, niter=10, order=3, plots=False):
    """
    Iterative sigma-clipping spline fit with a fixed small number of interior knots.
    Returns (flux_norm, y_fit, w_excl, f_excl).
    If the continuum level is essentially zero, flux_norm will be all-NaN.
    """
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)

    sort_idx = np.argsort(w)
    w = w[sort_idx]
    f = f[sort_idx]

    mask = np.isfinite(f)
    if np.sum(mask) < 10:
        return np.full_like(f, np.nan), np.full_like(f, np.nan), [], []

    n_knots = 3
    w_min, w_max = np.min(w[mask]), np.max(w[mask])
    interior_knots = np.linspace(w_min, w_max, n_knots + 2)[1:-1]

    for _ in range(niter):
        tck = splrep(w[mask], f[mask], k=order, t=interior_knots, task=-1)
        y_fit = splev(w, tck)
        resid = f - y_fit
        if high_rej == 0:
            clipped = sigma_clip(resid, sigma_lower=low_rej, sigma_upper=np.inf,
                                 maxiters=1, masked=True)
        else:
            clipped = sigma_clip(resid, sigma_lower=low_rej, sigma_upper=high_rej,
                                 maxiters=1, masked=True)
        new_mask = ~clipped.mask & mask
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    tck = splrep(w[mask], f[mask], k=order, t=interior_knots, task=-1)
    y_fit = splev(w, tck)

    excluded_idx = np.where(~mask)[0]
    w_excl = w[excluded_idx].tolist()
    f_excl = f[excluded_idx].tolist()

    y_fit_scale = np.nanmedian(np.abs(y_fit[np.isfinite(y_fit)]))
    y_fit_threshold = max(y_fit_scale * 1e-4, 1e-30)
    with np.errstate(divide='ignore', invalid='ignore'):
        flux_norm = np.where(np.abs(y_fit) > y_fit_threshold, f / y_fit, np.nan)

    return flux_norm, y_fit, w_excl, f_excl


# ============================================================
# Plotting functions (sin aperturas small/large)
# ============================================================

def plot_source_diagnostic(
    white, x0, y0, x0_ref, y0_ref,
    src_mask_full, src_mask_used,
    ring_mask, excluded_annulus_mask, source_excluded_mask,
    source_id, outfile,
    aperture_radius_pix,
    sky_annulus_r_in, sky_annulus_r_out,
):
    vmin, vmax = get_white_display_limits(white)
    ny, nx = white.shape

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(white, origin="lower", interpolation="nearest",
                   extent=[-0.5, nx-0.5, -0.5, ny-0.5], vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label="White-light flux")

    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.set_aspect('equal')

    ax.plot(x0_ref, y0_ref, marker="x", color="white", ms=8, mew=1.2, label="Catalogue")
    ax.plot(x0,     y0,     marker="+", color="yellow", ms=10, mew=1.5, label="Recentred")

    # Apertura nominal (extracción)
    usedy, usedx = np.where(src_mask_used)
    if len(usedx):
        ax.scatter(usedx, usedy, s=3, color=NOMINAL_AP_COLOR, alpha=0.90,
                   label=f"Apertura nominal ({APERTURE_NOMINAL_PSF_FRACTION*100:.0f}% PSF)")

    ringy, ringx = np.where(ring_mask)
    if len(ringx):
        ax.scatter(ringx, ringy, s=1, color="lime", alpha=0.30, label="Sky annulus")
    excy, excx = np.where(excluded_annulus_mask)
    if len(excx):
        ax.scatter(excx, excy, s=2, color="magenta", alpha=0.35, label="Excluded annulus pix.")

    excsrcy, excsrcx = np.where(source_excluded_mask)
    if len(excsrcx):
        ax.scatter(excsrcx, excsrcy, s=3, color="orange", alpha=0.55, label="Removed source pix.")

    for r, ls in [(sky_annulus_r_in, '--'), (sky_annulus_r_out, '--')]:
        ax.add_patch(plt.Circle((x0, y0), r, color="green", fill=False, lw=1.2, ls=ls))

    ax.legend(loc="best", fontsize=7)
    ax.set_title(f"Diagnostic plot: {source_id}", pad=15)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    fig.tight_layout(pad=0.1)
    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def _add_emission_lines(ax, wave_min, wave_max):
    c_kms = 299792.458
    delta_v = 300.0
    delta_lambda_factor = delta_v / c_kms

    visible_lines = [
        (wl, lbl) for wl, lbl in EMISSION_LINES if wave_min <= wl <= wave_max
    ]
    if not visible_lines:
        return

    y_levels = [0.92, 0.78]

    for i, (wl, label) in enumerate(visible_lines):
        wl_min = wl * (1.0 - delta_lambda_factor)
        wl_max = wl * (1.0 + delta_lambda_factor)

        ax.axvspan(wl_min, wl_max, color='lightgrey', alpha=0.5, zorder=0)
        color = get_line_color(label)
        ax.axvline(wl, color=color, linestyle='--', linewidth=0.8, alpha=0.8)

        y_pos = y_levels[i % len(y_levels)]
        ax.text(wl_max + (wave_max - wave_min) * 0.005, y_pos,
                label, transform=ax.get_xaxis_transform(),
                ha='left', va='center', fontsize=7, color=color, clip_on=False)

def compute_local_noise_from_residual(resid, half_win=25, sigma_clip=2.0, max_iters=3):
    """
    Estimate local noise from the normalised residual (flux_norm - 1).
    For each pixel, take a window of ±half_win channels, perform iterative
    sigma clipping (rejecting points > sigma_clip * std) and return the
    final standard deviation.
    """
    n = len(resid)
    noise = np.full(n, np.nan)
    for i in range(n):
        i1 = max(0, i - half_win)
        i2 = min(n, i + half_win + 1)
        vals = resid[i1:i2]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            continue
        # iterative sigma clipping
        for _ in range(max_iters):
            med = np.median(vals)
            std = np.std(vals)
            if std == 0:
                break
            mask = np.abs(vals - med) < sigma_clip * std
            new_vals = vals[mask]
            if len(new_vals) == len(vals):
                break
            vals = new_vals
        if len(vals) >= 5:
            noise[i] = np.std(vals)
    return noise

def plot_normalised_spectrum_dual(
    wave, flux_norm, flux_norm_err_prop, flux_norm_err_local,
    source_id, outfile,
    wave_min=2.10, wave_max=2.40, ks_mag=None,
    snr_value=None,
):
    """
    Plot normalised spectrum with two error envelopes (propagated and local)
    and two histograms in separate side panels.
    """
    good = np.isfinite(wave) & (wave >= wave_min) & (wave <= wave_max)
    if np.sum(good) < 10:
        good = np.isfinite(wave)
    w = wave[good]
    fn = flux_norm[good]
    fe_prop = flux_norm_err_prop[good]
    fe_local = flux_norm_err_local[good]

    # Determine y limits based on both errors
    good_y = np.isfinite(fn) & (np.isfinite(fe_prop) | np.isfinite(fe_local))
    if np.sum(good_y) > 0:
        y1 = np.nanpercentile((fn - np.maximum(fe_prop, fe_local))[good_y], 0.1)
        y2 = np.nanpercentile((fn + np.maximum(fe_prop, fe_local))[good_y], 99.9)
        dy = y2 - y1
        ymin, ymax = y1 - 0.35*dy, y2 + 0.35*dy
    else:
        ymin, ymax = 0.5, 1.5

    fig = plt.figure(figsize=(24, 5))
    gs = GridSpec(1, 3, width_ratios=[5, 1, 1], wspace=0.15)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_hist_prop = fig.add_subplot(gs[0, 1])
    ax_hist_local = fig.add_subplot(gs[0, 2])

    # Main spectrum
    ax_spec.plot(w, fn, lw=0.9, color='black', label='Normalised flux')

    # Propagated error envelope (blue)
    ax_spec.fill_between(w, fn - fe_prop, fn + fe_prop,
                         color='blue', alpha=0.2, label='Error propagado')
    ax_spec.plot(w, fn + fe_prop, lw=0.5, color='blue', alpha=0.6, ls='--')
    ax_spec.plot(w, fn - fe_prop, lw=0.5, color='blue', alpha=0.6, ls='--')

    # Local error envelope (green)
    ax_spec.fill_between(w, fn - fe_local, fn + fe_local,
                         color='green', alpha=0.2, label='Error local')
    ax_spec.plot(w, fn + fe_local, lw=0.5, color='green', alpha=0.6, ls='--')
    ax_spec.plot(w, fn - fe_local, lw=0.5, color='green', alpha=0.6, ls='--')

    ax_spec.set_ylim(ymin, ymax)
    ax_spec.set_xlim(wave_min, wave_max)
    ax_spec.set_xlabel('Wavelength [micron]')
    ax_spec.set_ylabel('Normalized flux')
    ax_spec.set_title(f'Normalised spectrum: {source_id}', pad=20)
    ax_spec.legend(loc='best', fontsize=7)

    _add_emission_lines(ax_spec, wave_min, wave_max)

    text_lines = []
    if ks_mag is not None:
        text_lines.append(f"Ks = {ks_mag:.2f}")
    if snr_value is not None and np.isfinite(snr_value):
        text_lines.append(f"rough S/N ≈ {snr_value:.1f}")
    if text_lines:
        ax_spec.text(0.98, 0.05, "\n".join(text_lines),
                     transform=ax_spec.transAxes, ha='right', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Histograms
    # Propagated
    good_prop = np.isfinite(fe_prop)
    if np.sum(good_prop) > 0:
        ax_hist_prop.hist(fe_prop[good_prop], bins=100, color='blue', alpha=0.5)
        med_prop = np.median(fe_prop[good_prop])
        ax_hist_prop.axvline(med_prop, color='darkblue', ls='--', lw=1,
                             label=f'Med={med_prop:.4f}')
        p1, p99 = np.percentile(fe_prop[good_prop], [1, 99])
        if np.isfinite(p1) and np.isfinite(p99):
            padding = 0.1 * (p99 - p1)
            ax_hist_prop.set_xlim(p1 - padding, p99 + padding)
    else:
        ax_hist_prop.text(0.5, 0.5, 'No data', transform=ax_hist_prop.transAxes,
                          ha='center', va='center', color='grey', fontsize=8)
    ax_hist_prop.yaxis.tick_right()
    ax_hist_prop.yaxis.set_label_position("right")
    ax_hist_prop.set_xlabel('Error (norm.)')
    ax_hist_prop.set_ylabel('Counts')
    ax_hist_prop.set_title('Propagated', fontsize=8)
    ax_hist_prop.legend(fontsize=7)

    # Local
    good_local = np.isfinite(fe_local)
    if np.sum(good_local) > 0:
        ax_hist_local.hist(fe_local[good_local], bins=100, color='green', alpha=0.5)
        med_local = np.median(fe_local[good_local])
        ax_hist_local.axvline(med_local, color='darkgreen', ls='--', lw=1,
                              label=f'Med={med_local:.4f}')
        p1, p99 = np.percentile(fe_local[good_local], [1, 99])
        if np.isfinite(p1) and np.isfinite(p99):
            padding = 0.1 * (p99 - p1)
            ax_hist_local.set_xlim(p1 - padding, p99 + padding)
    else:
        ax_hist_local.text(0.5, 0.5, 'No data', transform=ax_hist_local.transAxes,
                           ha='center', va='center', color='grey', fontsize=8)
    ax_hist_local.yaxis.tick_right()
    ax_hist_local.yaxis.set_label_position("right")
    ax_hist_local.set_xlabel('Error (norm.)')
    ax_hist_local.set_ylabel('Counts')
    ax_hist_local.set_title('Local', fontsize=8)
    ax_hist_local.legend(fontsize=7)

    fig.savefig(outfile, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)

def plot_final_spectrum(
    wave, flux, flux_err, source_id, ks_mag, outfile,
    wave_min=2.05, wave_max=2.40, edge_fraction=0.10,
    p_low=0.05, p_high=99.95, pad_fraction=0.70, snr_value=None,
):
    good_wave = np.isfinite(wave) & (wave >= wave_min) & (wave <= wave_max)
    if np.sum(good_wave) < 10:
        good_wave = np.isfinite(wave)
    w = wave[good_wave]
    f = flux[good_wave]
    e = flux_err[good_wave]

    n  = len(w)
    i1 = int(edge_fraction * n)
    i2 = int((1.0 - edge_fraction) * n)
    if i2 <= i1:
        i1, i2 = 0, n

    central_flux = f[i1:i2]
    central_unc  = e[i1:i2]
    good = np.isfinite(central_flux) & np.isfinite(central_unc)
    if np.sum(good) > 10:
        yvals = np.concatenate([central_flux[good],
                                central_flux[good] - central_unc[good],
                                central_flux[good] + central_unc[good]])
        y1, y2 = np.nanpercentile(yvals, p_low), np.nanpercentile(yvals, p_high)
        dy = y2 - y1
        if np.isfinite(dy) and dy > 0:
            ymin, ymax = y1 - pad_fraction*dy, y2 + pad_fraction*dy
        else:
            ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
    else:
        good2 = np.isfinite(f)
        if np.sum(good2) > 0:
            ymin, ymax = np.nanmin(f[good2]), np.nanmax(f[good2])
            dy = ymax - ymin
            ymin -= 0.70*dy; ymax += 0.70*dy
        else:
            ymin, ymax = -1, 1

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(w, f, lw=0.9, color="black", label="Final (annulus sky)")
    ax.fill_between(w, f - e, f + e, color="gray", alpha=0.30, label="Unc (propagated)")

    if i1 > 0:
        ax.axvspan(w[0], w[i1], color="lightgray", alpha=0.20)
    if i2 < n:
        ax.axvspan(w[i2-1], w[-1], color="lightgray", alpha=0.20)

    _add_emission_lines(ax, wave_min, wave_max)

    text_lines = [f"Ks = {ks_mag:.2f}"]
    if snr_value is not None and np.isfinite(snr_value):
        text_lines.append(f"Main S/N ≈ {snr_value:.1f}")
    
    ax.text(0.98, 0.05, "\n".join(text_lines), transform=ax.transAxes, ha="right", va="bottom")
    ax.set_xlim(wave_min, wave_max)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Flux")
    ax.set_title(f"Final spectrum: {source_id}", pad=20)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(outfile, dpi=FIG_DPI)
    plt.close(fig)

# ============================================================
# Main extraction for one IFU
# ============================================================

def extract_one_ifu(
    cube_file, catalogue_file, output_dir, extraction_mode="sum",
    ifu_name=None
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cube, header, primary_header = load_cube(cube_file, ext=1)
    # Determine IFU arm number and create dedicated output subfolder
    arm_number = get_arm_number(header)
    output_dir = Path(output_dir) / f"IFU{arm_number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    nw, ny, nx = cube.shape
    shape = (ny, nx)

    wave             = build_wavelength_axis(header, nw)
    pixel_scale_arcsec = get_pixel_scale_arcsec(header)
    fwhm_arcsec, fwhm_pix, psf_diag = get_psf_from_primary_header(
        primary_header, header, pixel_scale_arcsec
    )

    aperture_radius_pix = APERTURE_NOMINAL_PSF_FRACTION * fwhm_pix

    # Prepare sky annulus settings
    if SKY_ANNULUS:
        sky_inner_pix       = SKY_ANNULUS_INNER_FACTOR       * fwhm_pix
        sky_outer_pix       = SKY_ANNULUS_OUTER_FACTOR       * fwhm_pix
        sky_exclusion_radius= SKY_NEIGHBOUR_EXCLUSION_FACTOR * fwhm_pix
        use_sky_annulus = True
    else:
        sky_inner_pix = 0.0
        sky_outer_pix = 0.0
        sky_exclusion_radius = 0.0
        use_sky_annulus = False

    print(f"\n=== Processing IFU: {Path(cube_file).name} ===")
    print(f"Pixel scale               = {pixel_scale_arcsec:.4f} arcsec/pix")
    print(f"Adopted PSF FWHM          = {fwhm_arcsec:.3f} arcsec  ({fwhm_pix:.3f} pix)")
    print(f"Aperture radius           = {aperture_radius_pix:.3f} pix")
    if use_sky_annulus:
        print(f"Sky annulus radii         = ({sky_inner_pix:.3f}, {sky_outer_pix:.3f}) pix")
        print(f"Sky neighbour excl. radius= {sky_exclusion_radius:.3f} pix")
    else:
        print("Sky annulus disabled (SKY_ANNULUS=False). Using raw spectra.")
    print(f"Extraction K limit        = {K_LIMIT_EXTRACT:.1f}")
    print(f"Sky-protection K limit    = {K_LIMIT_SKY:.1f}")
    print(f"Extraction mode           = {extraction_mode}")
    print(f"Min aperture fraction     = {MIN_SOURCE_FRACTION_INSIDE:.2f}")
    print(f"IFU size                  = {nx} x {ny} pix")
    print("Accepted PSF indicators:")
    for name, val in psf_diag["accepted"]:
        print(f"   - {name}: {val:.3f} arcsec")
    if psf_diag["rejected"]:
        print("Rejected PSF indicators:")
        for name, val, reason in psf_diag["rejected"]:
            print(f"   - {name}: {val}  [{reason}]")

    if use_sky_annulus:
        ifu_half_min = min(nx, ny) / 2.0
        if sky_outer_pix > 0.8 * ifu_half_min:
            print(f"  *** WARNING: sky outer radius ({sky_outer_pix:.1f} pix) is large "
                  f"relative to IFU half-size ({ifu_half_min:.1f} pix). "
                  f"Sky ring may be very small or empty for off-centre stars.")

    full_cat      = load_catalogue(catalogue_file)
    cat_extract   = select_catalogue_by_k(full_cat, K_LIMIT_EXTRACT)
    cat_sky       = select_catalogue_by_k(full_cat, K_LIMIT_SKY)
    cel_wcs       = get_celestial_wcs(header)
    x_all_sky, y_all_sky = world_to_pixel(cel_wcs, cat_sky["ra"], cat_sky["dec"])

    contamination_margin = max(aperture_radius_pix, sky_outer_pix, sky_exclusion_radius) * 2.0
    near_ifu = select_catalogue_sources_near_ifu(
        x_all_sky, y_all_sky, nx, ny, margin_pix=contamination_margin
    )
    x_near = x_all_sky[near_ifu]
    y_near = y_all_sky[near_ifu]

    x_all_extract, y_all_extract = world_to_pixel(cel_wcs, cat_extract["ra"], cat_extract["dec"])
    candidate = select_catalogue_sources_near_ifu(
        x_all_extract, y_all_extract, nx, ny, margin_pix=RECENTER_BOX_HALF_SIZE
    )
    cat_ifu  = cat_extract[candidate]
    x_ifu    = x_all_extract[candidate]
    y_ifu    = y_all_extract[candidate]

    print(f"Stars K<{K_LIMIT_EXTRACT:.1f} for extraction: {len(cat_extract)}")
    print(f"Stars K<{K_LIMIT_SKY:.1f}  for sky protection: {len(cat_sky)}")
    print(f"Extraction candidates near IFU: {len(cat_ifu)}")

    if len(cat_ifu) == 0:
        raise RuntimeError("No extraction stars fall inside or near the IFU.")

    white = make_whitelight_image(cube, wave)

    sort_idx   = np.argsort(np.asarray(cat_ifu["K"], dtype=float))
    cat_ifu    = cat_ifu[sort_idx]
    x_ifu_ref  = x_ifu[sort_idx]
    y_ifu_ref  = y_ifu[sort_idx]
    x_ifu_rec  = np.full(len(cat_ifu), np.nan, dtype=float)
    y_ifu_rec  = np.full(len(cat_ifu), np.nan, dtype=float)
    done_x, done_y = [], []

    for i in range(len(cat_ifu)):
        x0g, y0g = x_ifu_ref[i], y_ifu_ref[i]
        if done_x:
            excl_mask = build_recentering_exclusion_mask(
                shape, np.asarray(done_x), np.asarray(done_y), aperture_radius_pix
            )
            x_rec, y_rec = recenter_on_peak_excluding_neighbours(
                white, x0g, y0g, forbidden_mask=excl_mask,
                half_size=RECENTER_BOX_HALF_SIZE,
                distance_sigma_pix=max(1.0, 0.6*fwhm_pix),
                centroid_box_half_size=1,
            )
        else:
            x_rec, y_rec = recenter_on_peak(white, x0g, y0g, half_size=RECENTER_BOX_HALF_SIZE)
        x_ifu_rec[i] = x_rec
        y_ifu_rec[i] = y_rec
        done_x.append(x_rec)
        done_y.append(y_rec)

    x_ifu = x_ifu_rec
    y_ifu = y_ifu_rec

    fraction_inside = np.array([
        source_fraction_inside_ifu(x, y, aperture_radius_pix, nx, ny)
        for x, y in zip(x_ifu, y_ifu)
    ])
    inside_final = np.array([
        source_is_usable(x, y, aperture_radius_pix, nx, ny, MIN_SOURCE_FRACTION_INSIDE)
        for x, y in zip(x_ifu, y_ifu)
    ])
    cat_ifu       = cat_ifu[inside_final]
    x_ifu_ref     = x_ifu_ref[inside_final]
    y_ifu_ref     = y_ifu_ref[inside_final]
    x_ifu         = x_ifu[inside_final]
    y_ifu         = y_ifu[inside_final]
    fraction_inside = fraction_inside[inside_final]

    print(f"Sources accepted after recentering: {len(cat_ifu)}")
    if len(cat_ifu) == 0:
        raise RuntimeError("No sources satisfy the aperture fraction threshold.")

    summary_rows = []

    for i, (row, x0, y0, x0_ref, y0_ref, frac_in) in enumerate(
        zip(cat_ifu, x_ifu, y_ifu, x_ifu_ref, y_ifu_ref, fraction_inside)
    ):
        source_base = make_source_basename(row["ra"], row["dec"])
        full_id = f"{ifu_name}_{source_base}" if ifu_name else source_base
        source_id = full_id
        display_id = source_base
        print(f"\nExtracting {source_id}   Ks={row['K']:.2f}   frac_inside={frac_in:.3f}")

        src_mask_full = circular_mask(shape, x0, y0, aperture_radius_pix)
        source_contam_mask = build_source_contamination_mask(
            shape, x_ifu, y_ifu,
            contam_radius=SOURCE_CONTAMINATION_FACTOR * aperture_radius_pix,
            target_index=i,
        )
        src_mask_used    = src_mask_full & (~source_contam_mask)
        source_excluded_mask = src_mask_full & source_contam_mask

        if np.sum(src_mask_used) == 0:
            warnings.warn(f"{source_id}: no uncontaminated source pixels. Skipping.")
            continue

        src_spec_raw = extract_aperture_spectrum(
            cube, src_mask_used, mode=extraction_mode,
            source_clip_sigma=SOURCE_CLIP_SIGMA,
            source_clip_maxiters=SOURCE_CLIP_MAXITERS,
            source_clip_min_pixels=SOURCE_CLIP_MIN_PIXELS,
        )
        N_ap = int(np.sum(src_mask_used))

        # --- Sky annulus handling ---
        if use_sky_annulus:
            raw_annulus = annulus_mask(shape, x0, y0, sky_inner_pix, sky_outer_pix)
            sky_excl_neighbour = build_neighbour_exclusion_mask(
                shape, x_near, y_near,
                exclusion_radius=sky_exclusion_radius,
                target_index=None,
            )
            sky_mask = raw_annulus & (~src_mask_full) & (~sky_excl_neighbour)

            if np.sum(sky_mask) < MIN_SKY_PIXELS:
                warnings.warn(
                    f"{source_id}: too few sky pixels ({np.sum(sky_mask)} < {MIN_SKY_PIXELS}). "
                    f"Try increasing SKY_ANNULUS_OUTER_FACTOR. Skipping."
                )
                continue

            sky_pixels  = cube[:, sky_mask]
            sky_clipped = sigma_clip(sky_pixels, sigma=SKY_SIGMA_CLIP_SIGMA,
                                     maxiters=SKY_SIGMA_CLIP_MAXITERS,
                                     axis=1, masked=True)

            S_sky   = np.ma.median(sky_clipped, axis=1).filled(np.nan)
            med_sky = np.ma.median(sky_clipped, axis=1)
            mad     = np.ma.median(np.abs(sky_clipped - med_sky[:, np.newaxis]), axis=1)
            sigma_sky = 1.4826 * mad.filled(np.nan)

            N_sky_per_wave  = np.sum(~sky_clipped.mask, axis=1).astype(float)
            N_sky_per_wave[N_sky_per_wave < 1] = 1
            N_sky_median    = max(1, int(np.median(N_sky_per_wave)))

            if extraction_mode == "sum":
                flux = src_spec_raw - N_ap * S_sky
            else:
                flux = src_spec_raw - S_sky
        else:
            # No sky subtraction
            sky_mask = np.zeros(shape, dtype=bool)
            S_sky = np.zeros(nw, dtype=float)
            sigma_sky = np.zeros(nw, dtype=float)
            N_sky_median = 0
            flux = src_spec_raw.copy()

        raw_med = np.nanmedian(src_spec_raw)
        sky_med = np.nanmedian(S_sky) if use_sky_annulus else 0.0
        sky_contribution = N_ap * sky_med if (extraction_mode == "sum" and use_sky_annulus) else sky_med
        flux_med = np.nanmedian(flux)
        flux_std = np.nanstd(flux)

        print(f"  [DIAG] N_ap={N_ap}, N_sky_median={N_sky_median}")
        print(f"  [DIAG] src_spec_raw  : median = {raw_med:.4e}")
        print(f"  [DIAG] S_sky/pix     : median = {sky_med:.4e}  →  "
              f"{'N_ap*' if (extraction_mode=='sum' and use_sky_annulus) else ''}S_sky = {sky_contribution:.4e}")
        print(f"  [DIAG] flux (src−sky): median = {flux_med:.4e},  std = {flux_std:.4e}")
        print(f"  [DIAG] sigma_sky/pix : median = {np.nanmedian(sigma_sky):.4e}")

        # --- Variance estimation (only Poisson + sky) ---
        if use_sky_annulus:
            sigma_sky_safe = np.where(np.isfinite(sigma_sky) & (sigma_sky >= 0),
                                      sigma_sky, 0.0)
            var_sky_estimator = (np.pi/2.0) * sigma_sky_safe**2 / N_sky_median

            if extraction_mode == 'sum':
                sky_variance = (N_ap ** 2) * var_sky_estimator
            elif extraction_mode == 'mean':
                source_var = sigma_sky_safe**2 / N_ap
                sky_variance = source_var + var_sky_estimator
            elif extraction_mode == 'median':
                source_var = (np.pi/2.0) * sigma_sky_safe**2 / N_ap
                sky_variance = source_var + var_sky_estimator
            else:
                raise ValueError(f"Unknown extraction_mode: {extraction_mode}")
            sky_pix_var = sigma_sky_safe ** 2
            aperture_sky_var = N_ap * sky_pix_var
        else:
            sky_variance = 0.0
            aperture_sky_var = 0.0

        if INCLUDE_POISSON_NOISE:
            HC_ERG_CM = 1.98644586e-16
            TEL_AREA_CM2 = 5.281e5
            EXP_TIME = EXP_TIME_RUN
            EFF = 0.25
            delta_lambda_A = abs(header['CDELT3']) * 1e4
            wave_A = wave * 1e4
            E_photon = HC_ERG_CM / (wave_A * 1e-8)

            source_poisson_var = (
                np.maximum(flux, 0.0) * E_photon /
                (TEL_AREA_CM2 * EXP_TIME * delta_lambda_A * EFF)
            )

            total_variance = source_poisson_var + aperture_sky_var + sky_variance
        else:
            total_variance = aperture_sky_var + sky_variance

        flux_err = np.sqrt(total_variance)

        snr_empirical = estimate_snr_from_continuum(
            wave, flux, edge_fraction=EDGE_FRACTION_TO_IGNORE,
            smooth_window=SNR_SMOOTH_WINDOW,
            sigma_clip_value=SNR_LINE_REJECTION_SIGMA, co_limit=2.29,
        )
        snr_sky = robust_snr_estimate_from_final_spectrum(
            flux, flux_err,
            edge_fraction=EDGE_FRACTION_TO_IGNORE,
            smooth_window=SNR_SMOOTH_WINDOW,
            line_rejection_sigma=SNR_LINE_REJECTION_SIGMA,
        )

        save_spectrum_ascii(
            output_dir / f"{full_id}.txt",
            wave, flux, flux_err, S_sky, sigma_sky, N_ap, N_sky_median, src_spec_raw,
        )
        save_spectrum_fits(
            output_dir / f"{full_id}.fits",
            wave, flux, flux_err, S_sky, sigma_sky, N_ap, N_sky_median, src_spec_raw,
        )

        # Diagnostic plot
        if use_sky_annulus:
            plot_source_diagnostic(
                white, x0, y0, x0_ref, y0_ref,
                src_mask_full, src_mask_used,
                ring_mask=sky_mask,
                excluded_annulus_mask=sky_excl_neighbour & raw_annulus,
                source_excluded_mask=source_excluded_mask,
                source_id=display_id,
                outfile=output_dir / f"{full_id}_diagnostic.png",
                aperture_radius_pix=aperture_radius_pix,
                sky_annulus_r_in=sky_inner_pix, sky_annulus_r_out=sky_outer_pix,
            )
        else:
            empty_mask = np.zeros(shape, dtype=bool)
            plot_source_diagnostic(
                white, x0, y0, x0_ref, y0_ref,
                src_mask_full, src_mask_used,
                ring_mask=empty_mask,
                excluded_annulus_mask=empty_mask,
                source_excluded_mask=source_excluded_mask,
                source_id=display_id,
                outfile=output_dir / f"{full_id}_diagnostic.png",
                aperture_radius_pix=aperture_radius_pix,
                sky_annulus_r_in=0.0, sky_annulus_r_out=0.0,
            )

        plot_final_spectrum(
            wave, flux, flux_err, display_id, float(row["K"]),
            outfile=output_dir / f"{full_id}_final_spectrum.png",
            wave_min=PLOT_WAVE_MIN, wave_max=PLOT_WAVE_MAX,
            edge_fraction=EDGE_FRACTION_TO_IGNORE,
            p_low=Y_PERCENTILE_LOW, p_high=Y_PERCENTILE_HIGH,
            pad_fraction=Y_PADDING_FRACTION, snr_value=snr_empirical,
        )

        # Normalisation
        good_norm = np.isfinite(flux) & np.isfinite(flux_err)
        n_good = int(np.sum(good_norm))
        print(f"  [{source_id}] Valid points for normalisation: {n_good}")
        if n_good < 20:
            print(f"  [{source_id}] Not enough valid points — skipping normalisation.")
            summary_rows.append(_make_summary_row(
                full_id, row, x0_ref, y0_ref, x0, y0, frac_in,
                src_mask_full, src_mask_used, source_excluded_mask, sky_mask,
                snr_empirical, snr_sky, N_sky_median,
                sky_inner_pix, sky_outer_pix,
            ))
            continue

        try:
            w_fit = wave[good_norm]
            f_fit = flux[good_norm]
            e_fit = flux_err[good_norm]

            flux_norm, y_fit, w_excl, f_excl = continuum(
                w_fit, f_fit, low_rej=LOW_REJ, high_rej=HIGH_REJ,
                niter=NITER, order=ORDER, plots=False,
            )

            with np.errstate(divide='ignore', invalid='ignore'):
                cont_snr = np.abs(y_fit) / e_fit
            MIN_CONT_SNR = 3.0
            reliable = np.isfinite(cont_snr) & (cont_snr >= MIN_CONT_SNR)

            flux_norm_err_prop = np.full_like(y_fit, np.nan)
            flux_norm_err_prop[reliable] = e_fit[reliable] / np.abs(y_fit[reliable])

            # Compute local noise from residual (flux_norm - 1)
            resid = flux_norm - 1.0
            noise_local = compute_local_noise_from_residual(resid, half_win=25, sigma_clip=2.0, max_iters=3)

            wave_ang = w_fit * 1e4

            # Guardar error propagado
            save_starkit_normalised_txt(
                output_dir / f"{full_id}_norm_prop_err.txt",
                wave_ang, flux_norm, flux_norm_err_prop,
            )
            save_normalised_fits(
                output_dir / f"{full_id}_norm_prop_err.fits",
                w_fit * 1e4, flux_norm, flux_norm_err_prop,
            )

            # Guardar error local
            save_starkit_normalised_txt(
                output_dir / f"{full_id}_norm_local_err.txt",
                wave_ang, flux_norm, noise_local,
            )
            save_normalised_fits(
                output_dir / f"{full_id}_norm_local_err.fits",
                w_fit * 1e4, flux_norm, noise_local,
            )

            # Plot con ambos errores
            plot_normalised_spectrum_dual(
                w_fit, flux_norm,
                flux_norm_err_prop, noise_local,
                display_id,
                outfile=output_dir / f"{full_id}_norm_spectrum.png",
                wave_min=PLOT_WAVE_MIN, wave_max=PLOT_WAVE_MAX,
                ks_mag=float(row["K"]),
                snr_value=snr_empirical,
            )

            print(f"  [{source_id}] Normalised files saved (propagated and local errors).")
        except Exception:
            print(f"  [{source_id}] Normalisation failed:")
            traceback.print_exc()

        summary_rows.append(_make_summary_row(
            full_id, row, x0_ref, y0_ref, x0, y0, frac_in,
            src_mask_full, src_mask_used, source_excluded_mask, sky_mask,
            snr_empirical, snr_sky, N_sky_median,
            sky_inner_pix, sky_outer_pix,
        ))

    if summary_rows:
        summary_table = Table(
            rows=summary_rows,
            names=[
                "source_id", "ra", "dec",
                "J", "dJ", "H", "dH", "K", "dK",
                "x_cat", "y_cat", "x_rec", "y_rec",
                "fraction_inside",
                "n_source_pix_full", "n_source_pix_used", "n_source_pix_removed",
                "n_sky_pix",
                "snr_empirical", "snr_sky",
                "N_sky_median", "sky_inner_pix", "sky_outer_pix",
            ],
        )
        summary_table.write(output_dir / f"source_summary_{ifu_name}.fits", overwrite=True)
        summary_table.write(output_dir / f"source_summary_{ifu_name}.txt",
                            format="ascii.fixed_width", overwrite=True)

    with open(output_dir / f"extraction_config_{ifu_name}.txt", "w") as f:
        f.write("Extraction configuration\n")
        f.write(f"pixel_scale_arcsec {pixel_scale_arcsec}\n")
        f.write(f"psf_fwhm_arcsec {fwhm_arcsec}\n")
        f.write(f"psf_fwhm_pix {fwhm_pix}\n")
        f.write(f"aperture_radius_pix {aperture_radius_pix}\n")
        f.write(f"use_sky_annulus {use_sky_annulus}\n")
        f.write(f"sky_inner_pix {sky_inner_pix}\n")
        f.write(f"sky_outer_pix {sky_outer_pix}\n")
        f.write(f"sky_neighbour_exclusion_radius {sky_exclusion_radius}\n")
        f.write(f"extraction_mode {extraction_mode}\n")
        f.write(f"min_source_fraction_inside {MIN_SOURCE_FRACTION_INSIDE}\n")
        f.write(f"k_limit_extract {K_LIMIT_EXTRACT}\n")
        f.write(f"k_limit_sky {K_LIMIT_SKY}\n")
        for name, val in psf_diag["accepted"]:
            safe = (name.replace(" ", "_").replace("/", "_")
                    .replace("[", "").replace("]", "").replace(".", "p"))
            f.write(f"psf_indicator_accepted_{safe} {val}\n")

    print(f"\nFinished IFU: {Path(cube_file).name}")

def _make_summary_row(
    full_id, row, x0_ref, y0_ref, x0, y0, frac_in,
    src_mask_full, src_mask_used, source_excluded_mask, sky_mask,
    snr_empirical, snr_sky, N_sky_median,
    sky_inner_pix, sky_outer_pix,
):
    return (
        full_id,
        row["ra"], row["dec"],
        row["J"], row["dJ"], row["H"], row["dH"], row["K"], row["dK"],
        x0_ref, y0_ref, x0, y0, frac_in,
        int(np.sum(src_mask_full)), int(np.sum(src_mask_used)),
        int(np.sum(source_excluded_mask)), int(np.sum(sky_mask)),
        snr_empirical, snr_sky,
        N_sky_median, sky_inner_pix, sky_outer_pix,
    )

def extract_from_list(list_file, catalogue_file, extraction_mode="sum", output_base=None):
    list_file = Path(list_file)
    fits_dir  = list_file.parent
    if output_base is None:
        output_base = fits_dir / "spectra_out"
    else:
        output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    fail_file = output_base / "fail_extraction.txt"

    with open(list_file) as f:
        names = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    for name in names:
        cube_file  = fits_dir / name
        ifu_name   = Path(name).stem
        try:
            extract_one_ifu(
                cube_file=cube_file,
                catalogue_file=catalogue_file,
                output_dir=output_base,
                extraction_mode=extraction_mode,
                ifu_name=ifu_name,
            )
        except Exception as e:
            error_msg = f"{ifu_name}: {type(e).__name__}: {e}"
            print(f"\n*** ERROR processing {ifu_name}: {e}")
            print(f"    Logging to {fail_file} and continuing with next IFU.\n")
            with open(fail_file, "a") as f_fail:
                f_fail.write(error_msg + "\n")
            continue

# ============================================================
# MAIN block
# ============================================================
if __name__ == "__main__":
    for con_name in CONS:
        print("\n" + "#" * 80)
        print(f"CONCATENATION: {con_name}")
        print("#" * 80)

        dummy_sky = Path(
            BASE_DIR_TEMPLATE.format(
                pointing=pointing, con_name=con_name, ob_name="dummy"
            )
        )
        con_dir = dummy_sky.parent.parent

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
            BASE_DIR = Path(
                BASE_DIR_TEMPLATE.format(
                    pointing=pointing, con_name=con_name, ob_name=ob_name
                )
            )
            if not BASE_DIR.exists():
                print(f"WARNING: sky_tweak folder missing for {ob_name}, skipping")
                continue

            paths = get_ob_paths(ob_name)

            print("\n" + "=" * 70)
            print(f"Processing {con_name} / {ob_name}")
            print("=" * 70)

            if not paths["corrected_dir"].exists():
                print(f"Skipping: corrected FITS directory not found → {paths['corrected_dir']}")
                continue
 

            if not any(paths["corrected_dir"].glob(INPUT_PATTERN)):
                print(f"No FITS files found in {paths['corrected_dir']}; skipping OB {ob_name}.")
                continue


            create_ifu_list_file(
                fits_dir=paths["corrected_dir"],
                list_file=paths["list_file"],
                pattern=INPUT_PATTERN,
            )
            extract_from_list(
                list_file=paths["list_file"],
                catalogue_file=CATALOGUE_FILE,
                extraction_mode="sum",
                output_base=paths["output_base"],
            )

    print("\nAll concatenations processed.")