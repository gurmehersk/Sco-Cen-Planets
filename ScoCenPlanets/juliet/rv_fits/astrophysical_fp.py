import os
import math
import numpy as np
import pandas as pd
from numpy import array as nparr
from scipy.interpolate import interp1d, PchipInterpolator

from astropy import units as u
from astropy import constants as const
from astropy.modeling.models import BlackBody

import matplotlib as mpl
import matplotlib.pyplot as plt

# Apply publication styling
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    plt.style.use('default')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 5,
        'xtick.minor.size': 2.5,
        'ytick.major.size': 5,
        'ytick.minor.size': 2.5,
    })

# =======================================================================
# CONFIG & PATHS
# =======================================================================
DATADIR = "./data/companion_isochrones/"
RESULTSDIR = "./results/fpscenarios/"

AGE_MYR = 21.0            
FEH = 0.00                
MSTAR_MSUN = 0.30         
DIST_PC = 99.7056         
TMAG_TARGET = 13.1        
A_V = 0.1331              

RV_SENSITIVITY_CSV = "./alternate_analysis_approach/rv_limit_curve.csv"
TESS_FILTER_PATH = os.path.join(DATADIR, "TESS_TESS.Red.dat")
MIST_ISOCMD_PATH = os.path.join(DATADIR, "MIST_iso_6a556a38f0b10/MIST_iso_6a556a38f0b10.iso.UBVRIplus")
MIST_ISO_THEO = os.path.join(DATADIR, "MIST_iso_6a556a38f0b10/MIST_iso_6a556a38f0b10.iso")
BARAFFE_COND_PATH = os.path.join(DATADIR, "cond03_10Myr.csv")

gemini_dir = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/Gemini/"
GEMINI_ZORRO_832 = os.path.join(gemini_dir, "TIC88297141_20260430_832.dat")
GEMINI_ZORRO_562 = os.path.join(gemini_dir, "TIC88297141_20260430_562.dat")
ZORRO832_FILTER_PATH = os.path.join(gemini_dir, "Gemini_Zorro.EO_832.dat")
ZORRO562_FILTER_PATH = os.path.join(gemini_dir, "Gemini_Zorro.EO_562.dat")
GAIA_CSV_PATH = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/gaia/gaia_contrast.csv"

OUTPUT_PLOT = "false_positive_validation_twopanel.pdf"

# =======================================================================
# 1. ISOCHRONE & MAGNITUDE MAPPING FUNCTIONS
# =======================================================================
def abs_tess_mag_from_distance(app_mag, dist_pc, A_V=0.1331):
    mu = 5 * np.log10(dist_pc / 10.0)
    A_T = 0.79 * A_V
    return app_mag - mu - A_T

M_TESS_TARGET = abs_tess_mag_from_distance(TMAG_TARGET, DIST_PC, A_V)

_tess_filt = None
def _load_tess_filter():
    global _tess_filt
    if _tess_filt is None:
        _tess_filt = pd.read_csv(TESS_FILTER_PATH, sep=r'\s+', names=['wvlen_angst', 'transmission'], comment='#')
    return _tess_filt

def abs_mag_in_tess_blackbody(lum_lsun, teff_k):
    filt = _load_tess_filter()
    wvlen = nparr(filt.wvlen_angst) * u.AA
    T_lambda = nparr(filt.transmission)
    M_bol_sun = 4.83
    M_Xs = []
    for L, Teff in zip(np.atleast_1d(lum_lsun) * u.Lsun, np.atleast_1d(teff_k) * u.K):
        bb = BlackBody(temperature=Teff)
        B_lambda = bb(wvlen) * (const.c / wvlen ** 2)
        F_X = 4 * np.pi * u.sr * np.trapz(B_lambda * T_lambda, wvlen)
        F_bol = const.sigma_sb * Teff ** 4
        M_bol_star = -2.5 * np.log10(L.to(u.Lsun).value) + M_bol_sun
        M_X = M_bol_star - 2.5 * np.log10((F_X / F_bol).decompose().value)
        M_Xs.append(M_X)
    return nparr(M_Xs)

def get_mist_stellar_grid(mstar_max):
    from read_mist_models import ISOCMD
    isocmd = ISOCMD(MIST_ISOCMD_PATH)
    hdr = isocmd.hdr_list
    tess_col = next((c for c in hdr if 'tess' in c.lower()), None)
    arr = isocmd.isocmds[0]
    mist_df = pd.DataFrame({'mass': arr['initial_mass'], 'M_TESS': arr[tess_col]})
    return mist_df[(mist_df.mass > 0.1) & (mist_df.mass < mstar_max)]

def get_baraffe_substellar_grid():
    bar_df = pd.read_csv(BARAFFE_COND_PATH).rename(columns={'M/Ms': 'mass', 'Teff': 'teff', 'L/Ls': 'lum'})
    bar_df['lum'] = 10 ** bar_df['lum']
    bar_df = bar_df[bar_df.mass < 0.1][['mass', 'lum', 'teff']]
    bar_df['M_TESS'] = abs_mag_in_tess_blackbody(nparr(bar_df.lum), nparr(bar_df.teff))
    return bar_df[['mass', 'M_TESS']]

mist_df = get_mist_stellar_grid(mstar_max=1.0)
bar_df = get_baraffe_substellar_grid()
iso_df = pd.concat([bar_df, mist_df], ignore_index=True).sort_values('mass')
iso_df['dmag_TESS'] = iso_df['M_TESS'] - M_TESS_TARGET

# Pchip Interpolators for Mass <-> dmag
fn_mass_to_dmag = PchipInterpolator(iso_df['mass'].values, iso_df['dmag_TESS'].values, extrapolate=False)
iso_by_dmag = iso_df.sort_values('dmag_TESS').reset_index(drop=True)
fn_dmag_to_mass = PchipInterpolator(iso_by_dmag['dmag_TESS'].values, iso_by_dmag['mass'].values, extrapolate=False)

mass_lo, mass_hi = iso_df['mass'].min(), iso_df['mass'].max()
dmag_lo, dmag_hi = sorted([iso_by_dmag['dmag_TESS'].min(), iso_by_dmag['dmag_TESS'].max()])

def fn_mass_to_dmag_axis(mass):
    return fn_mass_to_dmag(np.clip(np.atleast_1d(mass), mass_lo, mass_hi))

def fn_dmag_to_mass_axis(dmag):
    return fn_dmag_to_mass(np.clip(np.atleast_1d(dmag), dmag_lo, dmag_hi))

epsilon = float(fn_mass_to_dmag_axis(MSTAR_MSUN))

def fn_mass_to_dmag_calibrated(mass):
    return fn_mass_to_dmag_axis(mass) - epsilon

def fn_dmag_to_mass_calibrated(dmag):
    return fn_dmag_to_mass_axis(np.atleast_1d(dmag) + epsilon)

# Zorro Filters & Calibration
_zorro832_filt, _zorro562_filt = None, None
def _load_zorro832_filter():
    global _zorro832_filt
    if _zorro832_filt is None:
        _zorro832_filt = pd.read_csv(ZORRO832_FILTER_PATH, sep=r'\s+', names=['wvlen_angst', 'transmission'], comment='#')
    return _zorro832_filt

def _load_zorro562_filter():
    global _zorro562_filt
    if _zorro562_filt is None:
        _zorro562_filt = pd.read_csv(ZORRO562_FILTER_PATH, sep=r'\s+', names=['wvlen_angst', 'transmission'], comment='#')
    return _zorro562_filt

def abs_mag_in_zorro832_blackbody(lum_lsun, teff_k):
    filt = _load_zorro832_filter()
    wvlen = nparr(filt.wvlen_angst) * u.AA
    T_lambda = nparr(filt.transmission)
    M_bol_sun = 4.83
    M_Xs = []
    for L, Teff in zip(np.atleast_1d(lum_lsun)*u.Lsun, np.atleast_1d(teff_k)*u.K):
        bb = BlackBody(temperature=Teff)
        B_lambda = bb(wvlen) * (const.c / wvlen**2)
        F_X = 4*np.pi*u.sr*np.trapz(B_lambda*T_lambda, wvlen)
        F_bol = const.sigma_sb * Teff**4
        M_bol_star = -2.5*np.log10(L.to(u.Lsun).value) + M_bol_sun
        M_Xs.append(M_bol_star - 2.5*np.log10((F_X/F_bol).decompose().value))
    return nparr(M_Xs)

def abs_mag_in_zorro562_blackbody(lum_lsun, teff_k):
    filt = _load_zorro562_filter()
    wvlen = nparr(filt.wvlen_angst) * u.AA
    T_lambda = nparr(filt.transmission)
    M_bol_sun = 4.83
    M_Xs = []
    for L, Teff in zip(np.atleast_1d(lum_lsun)*u.Lsun, np.atleast_1d(teff_k)*u.K):
        bb = BlackBody(temperature=Teff)
        B_lambda = bb(wvlen) * (const.c / wvlen**2)
        F_X = 4*np.pi*u.sr*np.trapz(B_lambda*T_lambda, wvlen)
        F_bol = const.sigma_sb * Teff**4
        M_bol_star = -2.5*np.log10(L.to(u.Lsun).value) + M_bol_sun
        M_Xs.append(M_bol_star - 2.5*np.log10((F_X/F_bol).decompose().value))
    return nparr(M_Xs)

def get_mist_teff_lum_grid(mstar_max):
    from read_mist_models import ISO
    iso = ISO(MIST_ISO_THEO)
    arr = iso.isos[0]
    df = pd.DataFrame({'mass': arr['initial_mass'], 'teff': 10**arr['log_Teff'], 'lum': 10**arr['log_L']})
    return df[(df.mass > 0.1) & (df.mass < mstar_max)][['mass', 'teff', 'lum']]

bar_raw = pd.read_csv(BARAFFE_COND_PATH).rename(columns={'M/Ms': 'mass', 'Teff': 'teff', 'L/Ls': 'lum'})
bar_raw['lum'] = 10 ** bar_raw['lum']
bar_raw = bar_raw[bar_raw.mass < 0.1][['mass', 'teff', 'lum']]
mist_raw = get_mist_teff_lum_grid(mstar_max=1.0)

zorro_iso_df = pd.concat([bar_raw, mist_raw], ignore_index=True).sort_values('mass')

# 832nm mapping
zorro_iso_df['M_832'] = abs_mag_in_zorro832_blackbody(nparr(zorro_iso_df.lum), nparr(zorro_iso_df.teff))
M_832_TARGET_MODEL = float(np.interp(MSTAR_MSUN, zorro_iso_df.mass, zorro_iso_df.M_832))
zorro_iso_df['dmag_832'] = zorro_iso_df['M_832'] - M_832_TARGET_MODEL
zorro_by_dmag = zorro_iso_df.sort_values('dmag_832')
fn_dmag832_to_mass = PchipInterpolator(zorro_by_dmag.dmag_832.values, zorro_by_dmag.mass.values, extrapolate=False)

# 562nm mapping
zorro_iso_df['M_562'] = abs_mag_in_zorro562_blackbody(nparr(zorro_iso_df.lum), nparr(zorro_iso_df.teff))
M_562_TARGET_MODEL = float(np.interp(MSTAR_MSUN, zorro_iso_df.mass, zorro_iso_df.M_562))
zorro_iso_df['dmag_562'] = zorro_iso_df['M_562'] - M_562_TARGET_MODEL
zorro_by_dmag_562 = zorro_iso_df.sort_values('dmag_562')
fn_dmag562_to_mass = PchipInterpolator(zorro_by_dmag_562.dmag_562.values, zorro_by_dmag_562.mass.values, extrapolate=False)

# =======================================================================
# 2. LOAD OBSERVATIONAL DATA ARRAYS
# =======================================================================
# RVs
rv_df = pd.read_csv(RV_SENSITIVITY_CSV)
rv_df['sma_au'] = 10 ** rv_df.log10_a
rv_df['mp_msun'] = 10 ** rv_df.log10_mpsini
rv_df['dmag_TESS'] = fn_mass_to_dmag(rv_df.mp_msun)
sel_rv = ~rv_df['dmag_TESS'].isna()
rv_sma = rv_df.sma_au[sel_rv].values
rv_dmag = rv_df.dmag_TESS[sel_rv].values

# Zorro 832nm
gem_df = np.loadtxt(GEMINI_ZORRO_832, comments="#", skiprows=33)
zorro832_sep_arcsec = gem_df[:, 0]
zorro832_sep_au = zorro832_sep_arcsec * DIST_PC
dmag_832 = gem_df[:, 1]
dmag_tess_zorro832 = fn_mass_to_dmag_axis(fn_dmag832_to_mass(dmag_832))

# Zorro 562nm
gem_df_562 = np.loadtxt(GEMINI_ZORRO_562, comments="#", skiprows=32)
zorro562_sep_arcsec = gem_df_562[:, 0]
zorro562_sep_au = zorro562_sep_arcsec * DIST_PC
dmag_562 = gem_df_562[:, 1]
dmag_tess_zorro562 = fn_mass_to_dmag_axis(fn_dmag562_to_mass(dmag_562))

# Gaia
gaia_df = pd.read_csv(GAIA_CSV_PATH)
gaia_sep_arcsec = gaia_df['sep_milliarcsec'].values / 1000.0
gaia_dmag = gaia_df['dmag'].values
gaia_sep_au = gaia_sep_arcsec * DIST_PC

# Constants
delta_obs = 0.008
dmag_limit = -2.5 * np.log10(2 * delta_obs)
transit_local_arcsec = 1.945
transit_local_au = transit_local_arcsec * DIST_PC

EXCL_COLOR = 'black'
EXCL_ALPHA = 0.12
y_top = 0.0
y_bottom = 8.5

# =======================================================================
# 3. BUILD THE TWO-PANEL FIGURE
# =======================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 8.5), dpi=300)

# -----------------------------------------------------------------------
# SUBPLOT 1: ASSOCIATED COMPANIONS (AU)
# -----------------------------------------------------------------------
# Curves
l_rv = ax1.plot(rv_sma, rv_dmag, color='forestgreen', lw=2, label='PFS RVs', zorder=5)[0]
l_z832 = ax1.plot(zorro832_sep_au, dmag_tess_zorro832, color='mediumblue', lw=2, label='Gemini/Zorro 832nm', zorder=5)[0]
l_z562 = ax1.plot(zorro562_sep_au, dmag_tess_zorro562, color='cornflowerblue', lw=1.8, label='Gemini/Zorro 562nm', zorder=5)[0]
l_gaia = ax1.plot(gaia_sep_au, gaia_dmag, color='saddlebrown', lw=2, label='Gaia', zorder=4)[0]
l_depth = ax1.plot([np.nanmin(rv_sma), transit_local_au], [dmag_limit, dmag_limit], color='black', lw=2, label=r'Transit depth ($\delta_{\mathrm{TESS}}$)', zorder=6)[0]

# Vertical Boundary Walls
ax1.plot([transit_local_au, transit_local_au], [dmag_limit, y_top], color='black', lw=2, zorder=6)

start_idx832 = np.argmax(dmag_tess_zorro832 > 0.1669)
ax1.plot([zorro832_sep_au[start_idx832], zorro832_sep_au[start_idx832]], [y_top, dmag_tess_zorro832[start_idx832]], color='mediumblue', lw=2, zorder=5)
ax1.plot([zorro832_sep_au[-1], zorro832_sep_au[-1]], [y_top, dmag_tess_zorro832[-1]], color='mediumblue', lw=2, zorder=5)

start_idx562 = np.argmax(dmag_tess_zorro562 > 0.1)
ax1.plot([zorro562_sep_au[start_idx562], zorro562_sep_au[start_idx562]], [y_top, dmag_tess_zorro562[start_idx562]], color='cornflowerblue', lw=1.8, zorder=5)

ax1.plot([gaia_sep_au[-1], gaia_sep_au[-1]], [y_top, gaia_dmag[-1]], color='saddlebrown', lw=2, zorder=4)

# Dynamic Limits
xmin_au = np.nanmin(rv_sma)
xmax_au = max(ax1.get_xlim()[1], transit_local_au * 1.5)

# Exclusion Shading
ax1.fill_between([xmin_au, xmax_au], dmag_limit, y_bottom, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=1)
ax1.fill_between([transit_local_au, xmax_au], y_top, dmag_limit, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=1)
ax1.fill_between(zorro832_sep_au, dmag_tess_zorro832, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=2)
ax1.fill_between(zorro562_sep_au, dmag_tess_zorro562, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=2)
ax1.fill_between(rv_sma, rv_dmag, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=3)
ax1.fill_between(gaia_sep_au, gaia_dmag, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=1)

# Formatting Top Subplot
ax1.set_xscale('log')
ax1.set_xlim(xmin_au, xmax_au)
ax1.set_ylim(y_bottom, y_top)
ax1.set_xlabel('projected separation (AU)', fontsize=11)
ax1.set_ylabel(r'$\Delta T$ (mag)', fontsize=11)
ax1.set_title('Associated Companions', fontsize=12, fontweight='bold', pad=8)

# Secondary Mass Axis (Top Plot Only)
mass_ticks = np.array([0.003, 0.006, 0.012, 0.03, 0.1])
secax = ax1.secondary_yaxis('right', functions=(fn_dmag_to_mass_calibrated, fn_mass_to_dmag_calibrated))
secax.set_ylabel(r'Companion Mass ($M_\odot$)', fontsize=11)
secax.set_yticks(mass_ticks)
secax.set_yticklabels([f"{m:.2g}" for m in mass_ticks])

# -----------------------------------------------------------------------
# SUBPLOT 2: CHANCE ALIGNMENTS (Arcseconds, TESS-band equivalent)
# -----------------------------------------------------------------------
# Curves (using the TESS-converted arrays for 1-to-1 consistency with ax1!)
ax2.plot(zorro832_sep_arcsec, dmag_tess_zorro832, color='mediumblue', lw=2, zorder=5)
ax2.plot(zorro562_sep_arcsec, dmag_tess_zorro562, color='cornflowerblue', lw=1.8, zorder=5)
ax2.plot(gaia_sep_arcsec, gaia_dmag, color='saddlebrown', lw=2, zorder=4)

# Dynamic Limits (Calculated first so we can draw the L-shape flush to the edge)
xmin_arcsec = xmin_au / DIST_PC
xmax_arcsec = xmax_au / DIST_PC

# Transit depth & Photometric localization limits (Solid "L" shape matching ax1)
ax2.plot([xmin_arcsec, transit_local_arcsec], [dmag_limit, dmag_limit], color='black', lw=2, zorder=6)
l_loc = ax2.plot([transit_local_arcsec, transit_local_arcsec], [dmag_limit, y_top], color='black', lw=2, zorder=6)[0]

# Vertical Boundary Walls (matching ax1 logic)
start_idx832 = np.argmax(dmag_tess_zorro832 > 0.1669)
ax2.plot([zorro832_sep_arcsec[start_idx832], zorro832_sep_arcsec[start_idx832]], [y_top, dmag_tess_zorro832[start_idx832]], color='mediumblue', lw=2, zorder=5)
ax2.plot([zorro832_sep_arcsec[-1], zorro832_sep_arcsec[-1]], [y_top, dmag_tess_zorro832[-1]], color='mediumblue', lw=2, zorder=5)

start_idx562 = np.argmax(dmag_tess_zorro562 > 0.1)
ax2.plot([zorro562_sep_arcsec[start_idx562], zorro562_sep_arcsec[start_idx562]], [y_top, dmag_tess_zorro562[start_idx562]], color='cornflowerblue', lw=1.8, zorder=5)

ax2.plot([gaia_sep_arcsec[-1], gaia_sep_arcsec[-1]], [y_top, gaia_dmag[-1]], color='saddlebrown', lw=2, zorder=4)

# Exclusion Shading (Matching ax1's clean stacked shading perfectly)
ax2.fill_between(zorro832_sep_arcsec, dmag_tess_zorro832, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=2)
ax2.fill_between(zorro562_sep_arcsec, dmag_tess_zorro562, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=2)
ax2.fill_between(gaia_sep_arcsec, gaia_dmag, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=1)
ax2.fill_between([xmin_arcsec, xmax_arcsec], dmag_limit, y_bottom, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=1)
ax2.fill_between([transit_local_arcsec, xmax_arcsec], y_top, dmag_limit, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=1)

# Formatting Bottom Subplot
ax2.set_xscale('log')
ax2.set_xlim(xmin_arcsec, xmax_arcsec)
ax2.set_ylim(y_bottom, y_top)
ax2.set_xlabel('Angular Separation (arcsec)', fontsize=11)
ax2.set_ylabel(r'$\Delta T$ (mag)', fontsize=11)
ax2.set_title('Chance Alignments', fontsize=12, fontweight='bold', pad=8)

# -----------------------------------------------------------------------
# 4. SHARED UNIFIED LEGEND (Center-Right)
# -----------------------------------------------------------------------
# Squeeze the plots to the left to leave white space on the right,
# and reduce hspace since the legend is no longer wedged between them.
plt.subplots_adjust(hspace=0.25, right=0.72)

# Combine handles from both subplots
legend_handles = [l_rv, l_z832, l_z562, l_gaia, l_depth]
legend_labels = [h.get_label() for h in legend_handles]

# Draw single figure-level legend in the right-side white space
fig.legend(
    legend_handles, 
    legend_labels, 
    loc='center left', 
    bbox_to_anchor=(0.74, 0.5), # Anchored just outside the right edge of plots
    ncol=1,                     # Stack vertically!
    fontsize=9, 
    frameon=True, 
    facecolor='white', 
    framealpha=0.9,
    edgecolor='0.8'             # Subtle grey border
)

# Save
fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"Saved publication plot to {OUTPUT_PLOT}")
plt.show()


