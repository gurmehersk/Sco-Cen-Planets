"""
rv_msini_to_tess_dmag.py

Convert the RV-derived 3-sigma companion-mass sensitivity curve
(rvoutersensitivity_3sigma.csv, columns: log10sma, log10msini) into a
Delta_TESS_mag vs. semi-major-axis curve, for the false-positive validation
plot of TIC 88297141b.

METHODOLOGY
-----------
Companion masses are mapped to a TESS-band absolute magnitude using two
grids, merged at 0.1 Msun:

  - Stellar range (M > 0.1 Msun): MIST synthetic photometry (.iso.cmd file),
    which gives TESS magnitudes directly from real bolometric-correction
    tables (built on ATLAS12/PHOENIX synthetic spectra) -- no blackbody
    approximation needed here.

  - Sub-stellar range (M < 0.1 Msun): Baraffe/COND03 models, which only give
    Teff and Lbol (no photometry tables), so these are converted to a TESS
    magnitude via a blackbody approximation integrated through the TESS
    transmission curve. This mirrors your mentor's contrast_to_masslimit.py
    approach, but only for the mass range where it's actually necessary.

KEY DIFFERENCE from contrast_to_masslimit.py:
That script solves the inverse problem (observed imaging Delta-mag ->
companion mass), which needs a smoothed/resampled interpolation because it's
inverting a curve. Here we already have companion masses (from RV Msini),
and just want Delta-mag(mass) directly -- the forward direction, no
inversion needed.

ALSO DIFFERENT: TIC 88297141's own absolute TESS magnitude is computed from
its actual measured Tmag + Gaia distance, not a blackbody approximation --
more accurate, and means we don't need the target's Teff/Rstar at all here
(we still need Teff/Rstar -> Lbol for the *substellar companion* isochrone
grid; that part is unavoidable since Baraffe/COND doesn't include TESS
photometry).

YOU NEED TO SUPPLY (see CONFIG below and inline TODOs):
  1. TESS transmission curve (SVO Filter Profile Service, id=TESS/TESS.Red):
     http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=TESS/TESS.Red
     -- only needed for the substellar blackbody calculation now.
  2. A MIST isochrone at your adopted age (~19 Myr) with SYNTHETIC PHOTOMETRY
     output (not theoretical), from
     https://waps.cfa.harvard.edu/MIST/interp_isos.html
     Single age, v/vcrit=0.4 (check header of Luke's other MIST files if you
     have access, but this choice barely matters at these masses), [Fe/H]=0.0.
  3. A Baraffe/COND03 grid at the nearest tabulated age to your system
     (check if your mentor's data/companion_isochrones/ already has one
     close to ~10-20 Myr).
  4. Your existing rvoutersensitivity_3sigma.csv (already have this).
"""

import os
import numpy as np
import pandas as pd
from numpy import array as nparr
from scipy.interpolate import interp1d

from astropy import units as u
from astropy import constants as const
from astropy.modeling.models import BlackBody

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# CONFIG -- fill in / double check these against your actual repo paths
# -----------------------------------------------------------------------

DATADIR = "./data/companion_isochrones/"
RESULTSDIR = "./results/fpscenarios/"

# Adopted system parameters
AGE_MYR = 21.0            # midpoint of your 18-21 Myr EAGLES range -- change if you prefer an endpoint
FEH = 0.00                # solar; Sco-Cen literature is close to solar, negligible effect here
MSTAR_MSUN = 0.30         # TIC 88297141's own mass -- confirm this is your adopted final value

DIST_PC = 99.7056         # Gaia distance
# DIST_PC_ERR_HI = 0.4988 # not propagated below -- see note at bottom on adding this
# DIST_PC_ERR_LO = 0.2670

TMAG_TARGET = 13.1        # apparent TESS magnitude of TIC 88297141
A_V = 0.1331               # taken from astroAridane fit in Star.txt from dropbox 

RV_SENSITIVITY_CSV = "./alternate_analysis_approach/rv_limit_curve.csv"

TESS_FILTER_PATH = os.path.join(DATADIR, "TESS_TESS.Red.dat")
# download: http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=TESS/TESS.Red ---> DONE! 
# (two whitespace-separated columns: wavelength [Angstrom], transmission [0-1], no header)
# only needed for the substellar (Baraffe/COND) blackbody calculation below.

### Why? Because the baraffe csv does not give u the tess magnitude.. So you need something that can do the conversion. Luckily we have tess filters that can do that for us!!  

MIST_ISOCMD_PATH = os.path.join(DATADIR, "MIST_iso_6a556a38f0b10/MIST_iso_6a556a38f0b10.iso.UBVRIplus")
MIST_ISO_THEO = os.path.join(DATADIR, "MIST_iso_6a556a38f0b10/MIST_iso_6a556a38f0b10.iso")
# download SYNTHETIC PHOTOMETRY (not theoretical) from
# https://waps.cfa.harvard.edu/MIST/interp_isos.html -- single age, UBVRIplus --> DONE! 
# bandpass set (includes TESS), v/vcrit=0.4, [Fe/H]=FEH.

BARAFFE_COND_PATH = os.path.join(DATADIR, "cond03_10Myr.csv")
# TODO: confirm this file exists in your repo, or find the nearest available
# tabulated age to AGE_MYR (standard COND03 grid ages include 1, 5, 10, 20,
# 50, 100 Myr, ... -- prefer a 20 Myr file over 10 Myr if one exists, since
# it's closer to 19 Myr)

OUTPUT_PLOT = "rv_dmag_vs_sma.png"
OUTPUT_CSV = "rv_dmag_vs_sma.csv"

# -----------------------------------------------------------------------
# 1. Target star's absolute TESS magnitude, from real photometry
#    (no isochrone / blackbody approximation needed for the target itself)
# -----------------------------------------------------------------------

def abs_tess_mag_from_distance(app_mag, dist_pc, A_V=0.1331):
    """
    M = m - 5*log10(d/10pc) - A_T
    A_T/A_V ~ 0.79 is a reasonable approximation for the TESS band (similar
    extinction behavior to Rc/Ic); replace with your adopted value if you
    have one from your SED fit.
    """
    mu = 5 * np.log10(dist_pc / 10.0)
    A_T = 0.79 * A_V
    return app_mag - mu - A_T


M_TESS_TARGET = abs_tess_mag_from_distance(TMAG_TARGET, DIST_PC, A_V)
print(f"Target absolute TESS mag: {M_TESS_TARGET:.3f}")


# -----------------------------------------------------------------------
# 2a. Stellar range (M > 0.1 Msun): MIST synthetic photometry, direct TESS mag
# -----------------------------------------------------------------------

def get_mist_stellar_grid(mstar_max):
    """
    Read a MIST *synthetic photometry* (.iso.cmd) file and return mass vs.
    absolute TESS magnitude for 0.1 Msun < M < mstar_max.

    Uses the public read_mist_models.py reader's ISOCMD class (from the MIST
    website's "Reading the Isochrone/Track Files" page) -- your mentor's ISO
    class only reads the theoretical (.iso) files; ISOCMD reads the
    synthetic-photometry (.iso.cmd) files, which is what we want here.
    """
    from read_mist_models import ISOCMD

    isocmd = ISOCMD(MIST_ISOCMD_PATH)

    # isocmd.hdr_list holds the column names for this file; find whichever
    # one is the TESS band (name varies slightly by MIST version, so we
    # search rather than hardcode it -- print available columns if this
    # fails so you can see what's actually in your file)
    hdr = isocmd.hdr_list
    tess_col = next((c for c in hdr if 'tess' in c.lower()), None)
    if tess_col is None:
        raise ValueError(
            f"No TESS-like column found in {MIST_ISOCMD_PATH}. "
            f"Available columns: {hdr}\n"
            f"Make sure you downloaded SYNTHETIC PHOTOMETRY (not theoretical) "
            f"output, with a bandpass set that includes TESS (e.g. UBVRIplus)."
        )

    # single age -> isocmd.isocmds[0]; if you downloaded a multi-age file
    # instead, pick the right index the same way your mentor did for the
    # 5 Gyr MIST grid (iso.age_index(...))
    arr = isocmd.isocmds[0]

    mist_df = pd.DataFrame({
        'mass': arr['initial_mass'],
        'M_TESS': arr[tess_col],
    })
    mist_df = mist_df[(mist_df.mass > 0.1) & (mist_df.mass < mstar_max)]
    return mist_df


# -----------------------------------------------------------------------
# 2b. Sub-stellar range (M < 0.1 Msun): Baraffe/COND03 + blackbody -> TESS mag
# -----------------------------------------------------------------------

_tess_filt = None


### TESS_FILTER_PATH = os.path.join(DATADIR, "TESS_TESS.Red.dat") this is the thing thats being loaded here and then used in the abs_mag_in_tess_blackbody
def _load_tess_filter():
    global _tess_filt
    if _tess_filt is None:
        _tess_filt = pd.read_csv(
            TESS_FILTER_PATH, sep=r'\s+',
            names=['wvlen_angst', 'transmission'], comment='#'
        )
    return _tess_filt


def abs_mag_in_tess_blackbody(lum_lsun, teff_k):
    """
    Blackbody approximation for absolute TESS magnitude -- only used for the
    substellar (Baraffe/COND) range, which doesn't have real TESS photometry
    tables. Mirrors contrast_to_masslimit.abs_mag_in_bandpass, swapped to
    the TESS transmission curve.
    """

    filt = _load_tess_filter()
    wvlen = nparr(filt.wvlen_angst) * u.AA
    T_lambda = nparr(filt.transmission)

    M_bol_sun = 4.83

    M_Xs = []
    for L, Teff in zip(np.atleast_1d(lum_lsun) * u.Lsun,
                        np.atleast_1d(teff_k) * u.K):
        bb = BlackBody(temperature=Teff)
        B_nu = bb(wvlen)
        B_lambda = B_nu * (const.c / wvlen ** 2)

        F_X = 4 * np.pi * u.sr * np.trapz(B_lambda * T_lambda, wvlen)
        F_bol = const.sigma_sb * Teff ** 4

        M_bol_star = -2.5 * np.log10(L.to(u.Lsun).value) + M_bol_sun
        M_X = M_bol_star - 2.5 * np.log10((F_X / F_bol).decompose().value)

        M_Xs.append(M_X)

    return nparr(M_Xs)


def get_baraffe_substellar_grid():
    """
    Read the Baraffe/COND03 grid and return mass vs. absolute TESS magnitude
    for M < 0.1 Msun, via the blackbody approximation above.
    """
    bar_df = pd.read_csv(BARAFFE_COND_PATH)
    bar_df = bar_df.rename(
        columns={'M/Ms': 'mass', 'Teff': 'teff', 'L/Ls': 'lum'}
    )
    bar_df['lum'] = 10 ** bar_df['lum']  # COND03 tables report log10(L/Lsun)
    bar_df = bar_df[['mass', 'lum', 'teff']]
    bar_df = bar_df[bar_df.mass < 0.1]

    bar_df['M_TESS'] = abs_mag_in_tess_blackbody(
        nparr(bar_df.lum), nparr(bar_df.teff)
    )
    return bar_df[['mass', 'M_TESS']]


# -----------------------------------------------------------------------
# 3. Merge the two grids and build mass -> Delta_TESS_mag
# -----------------------------------------------------------------------

mist_df = get_mist_stellar_grid(mstar_max=1.0) ## changed this from 0.3 [MSTAR_MSUN]
bar_df = get_baraffe_substellar_grid()

iso_df = pd.concat([bar_df, mist_df], ignore_index=True).sort_values('mass')
iso_df['dmag_TESS'] = iso_df['M_TESS'] - M_TESS_TARGET

iso_df.to_csv(os.path.join(DATADIR, f"merged_iso_tessmag_{AGE_MYR:.0f}myr.csv"),
              index=False)

fn_mass_to_dmag = interp1d(
    nparr(iso_df['mass']), nparr(iso_df['dmag_TESS']),
    kind='quadratic', bounds_error=False, fill_value=np.nan
)

# -----------------------------------------------------------------------
# 4. Apply to your RV 3-sigma sensitivity curve
# -----------------------------------------------------------------------

rv_df = pd.read_csv(RV_SENSITIVITY_CSV)
rv_df['sma_au'] = 10 ** rv_df.log10_a
rv_df['mp_msun'] = 10 ** rv_df.log10_mpsini   # Msin(i) plotted directly, conservative
rv_df['dmag_TESS'] = fn_mass_to_dmag(rv_df.mp_msun)

rv_df = rv_df.sort_values('sma_au').reset_index(drop=True)

n_nan = rv_df['dmag_TESS'].isna().sum()
if n_nan > 0:
    print(f"WARNING: {n_nan}/{len(rv_df)} points fell outside the isochrone's "
          f"mass range ({iso_df.mass.min():.4f}-{iso_df.mass.max():.4f} Msun) "
          f"and are NaN. Check whether your RV Msini values extend below the "
          f"Baraffe/COND03 grid's minimum mass -- if so you'll need a grid "
          f"that extends further into the planetary-mass regime.")

rv_df.to_csv(OUTPUT_CSV, index=False)
print(f"saved {OUTPUT_CSV}")


print(rv_df['dmag_TESS'].notna().sum(), "of", len(rv_df), "are finite")
print(rv_df['mp_msun'].describe())

from scipy.interpolate import PchipInterpolator

iso_df = iso_df.sort_values('mass').reset_index(drop=True)

# --- sanity check: is dmag_TESS actually monotonic in mass? ---
d = np.diff(iso_df['dmag_TESS'].values)
if not np.all(d <= 0):
    bad_idx = np.where(d > 0)[0]
    print(f"WARNING: dmag_TESS is non-monotonic in mass at {len(bad_idx)} point(s), "
          f"e.g. near mass = {iso_df['mass'].values[bad_idx[:5]]} Msun. "
          f"This is either a mismatch between the Baraffe/COND blackbody mags and "
          f"the MIST synthetic-photometry mags at the 0.1 Msun stitch, or a real "
          f"deuterium/H-burning luminosity feature near ~0.07-0.09 Msun. "
          f"The dmag->mass inversion will be unreliable across that stretch -- "
          f"look at iso_df around those masses before trusting the secondary axis there.")

# forward: mass -> dmag. Shape-preserving (won't overshoot at the stitch),
# unlike the quadratic spline you had before.
fn_mass_to_dmag = PchipInterpolator(
    iso_df['mass'].values, iso_df['dmag_TESS'].values, extrapolate=False
)

# inverse: dmag -> mass, built from the SAME table, just sorted the other way
# (dmag_TESS decreases as mass increases, so sort by dmag ascending here)
iso_by_dmag = iso_df.sort_values('dmag_TESS').reset_index(drop=True)
fn_dmag_to_mass = PchipInterpolator(
    iso_by_dmag['dmag_TESS'].values, iso_by_dmag['mass'].values, extrapolate=False
)

mass_lo, mass_hi = iso_df['mass'].min(), iso_df['mass'].max()
dmag_lo, dmag_hi = sorted([iso_by_dmag['dmag_TESS'].min(), iso_by_dmag['dmag_TESS'].max()])

def fn_mass_to_dmag_axis(mass):
    return fn_mass_to_dmag(np.clip(np.atleast_1d(mass), mass_lo, mass_hi))

def fn_dmag_to_mass_axis(dmag):
    return fn_dmag_to_mass(np.clip(np.atleast_1d(dmag), dmag_lo, dmag_hi))

def fn_mass_to_dmag_calibrated(mass):
    return fn_mass_to_dmag_axis(mass) - epsilon

def fn_dmag_to_mass_calibrated(dmag):
    return fn_dmag_to_mass_axis(np.atleast_1d(dmag) + epsilon)

    # ---- secondary mass axis in log10(mass) space ----
#
# Linear mass ticks (0.2, 0.4, 0.6 ... Msun) all cluster near dmag~0 because
# the mass-luminosity relation is nearly flat across 0.2-1.0 Msun at this age,
# then extremely steep below it. Log-mass ticks spread evenly instead.

# choose your own "nice" log-spaced mass ticks rather than trusting the
# default locator, which has no idea it's secretly working in dex
mass_ticks = np.array([0.003, 0.006, 0.009, .012,  0.018,  0.03, 0.1])   # Msun
mass_ticks = mass_ticks[(mass_ticks >= mass_lo) & (mass_ticks <= mass_hi)]

test_masses = np.array([0.01, 0.03, 0.08, 0.1, 0.3, 0.6])
recovered = fn_dmag_to_mass_axis(fn_mass_to_dmag_axis(test_masses))
print(np.c_[test_masses, recovered])   # should match closely, except right at the stitch/kink
# -----------------------------------------------------------------------
# 5. Plot
# -----------------------------------------------------------------------

epsilon = float(fn_mass_to_dmag_axis(MSTAR_MSUN))
print(f"isochrone/photometry zero-point offset: {epsilon:+.3f} mag")

recovered_mass = float(fn_dmag_to_mass_axis(0.0 + epsilon))
print(f"round-trip check: {recovered_mass:.4f} Msun (should read ~{MSTAR_MSUN})")

fig, ax = plt.subplots(figsize=(5, 4))

sel = ~rv_df['dmag_TESS'].isna()
ax.plot(rv_df.sma_au[sel], rv_df.dmag_TESS[sel], color='green', lw=1.5)
ax.fill_between(rv_df.sma_au[sel], rv_df.dmag_TESS[sel], 0,
                 alpha=0.6, color='0.75', label=r'PFS RV')

ax.set_xscale('log')
ax.set_xlabel('semi-major axis (AU)')
ax.set_ylabel(r'$\Delta T$ (mag)')
ax.invert_yaxis()   # brighter = up, standard convention for contrast plots
ax.legend()

def fn_dmag_to_log10mass(dmag):
    mass = fn_dmag_to_mass_axis(dmag)     # clamped Pchip inverse, in Msun
    return np.log10(mass)

def fn_log10mass_to_dmag(log10mass):
    mass = 10 ** np.atleast_1d(log10mass)
    return fn_mass_to_dmag_axis(mass)

secax = ax.secondary_yaxis(
    'right', functions=(fn_dmag_to_mass_calibrated, fn_mass_to_dmag_calibrated)
)
secax.set_ylabel(r'companion mass ($M_\odot$)')
secax.set_yticks(mass_ticks)
secax.set_yticklabels([f"{m:.2g}" for m in mass_ticks])



# ---- Zorro 832nm: blackbody across the FULL mass range ----
# MIST has no Gemini/Zorro synthetic photometry, so unlike the TESS grid,
# there's no "real photometry" stellar branch here -- blackbody + this
# transmission curve is used for everything, stellar and substellar alike.
gemini_dir = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/Gemini/"
GEMINI_ZORRO_832 = os.path.join(gemini_dir, "TIC88297141_20260430_832.dat")
GEMINI_ZORRO_562 = os.path.join(gemini_dir, "TIC88297141_20260430_562.dat")
ZORRO832_FILTER_PATH = os.path.join(gemini_dir, "Gemini_Zorro.EO_832.dat")
# download: https://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id=Gemini/Zorro.EO_832

_zorro832_filt = None
def _load_zorro832_filter():
    global _zorro832_filt
    if _zorro832_filt is None:
        _zorro832_filt = pd.read_csv(ZORRO832_FILTER_PATH, sep=r'\s+',
                                      names=['wvlen_angst', 'transmission'], comment='#')
    return _zorro832_filt

def abs_mag_in_zorro832_blackbody(lum_lsun, teff_k):
    """Identical machinery to abs_mag_in_tess_blackbody, swapped to the
    Zorro 832nm transmission curve."""
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
        M_X = M_bol_star - 2.5*np.log10((F_X/F_bol).decompose().value)
        M_Xs.append(M_X)
    return nparr(M_Xs)

def get_mist_teff_lum_grid(mstar_max):
    """Plain theoretical .iso file (Teff, log L) -- NOT the .iso.cmd
    synthetic-photometry file, since MIST has nothing in Gemini/Zorro
    bands for us to use anyway."""
    from read_mist_models import ISO
    iso = ISO(MIST_ISO_THEO)   # TODO: your plain .iso file path
    arr = iso.isos[0]
    df = pd.DataFrame({
        'mass': arr['initial_mass'],
        'teff': 10**arr['log_Teff'],
        'lum': 10**arr['log_L'],
    })
    return df[(df.mass > 0.1) & (df.mass < mstar_max)][['mass', 'teff', 'lum']]

# reuse your existing Baraffe reader for the Teff/L columns (skip its
# TESS-specific abs_mag call -- we need raw teff/lum here, not M_TESS)
bar_raw = pd.read_csv(BARAFFE_COND_PATH).rename(
    columns={'M/Ms': 'mass', 'Teff': 'teff', 'L/Ls': 'lum'})
bar_raw['lum'] = 10 ** bar_raw['lum']
bar_raw = bar_raw[bar_raw.mass < 0.1][['mass', 'teff', 'lum']]

mist_raw = get_mist_teff_lum_grid(mstar_max=1.0)

zorro_iso_df = pd.concat([bar_raw, mist_raw], ignore_index=True).sort_values('mass')
zorro_iso_df['M_832'] = abs_mag_in_zorro832_blackbody(
    nparr(zorro_iso_df.lum), nparr(zorro_iso_df.teff))

M_832_TARGET_MODEL = float(np.interp(MSTAR_MSUN, zorro_iso_df.mass, zorro_iso_df.M_832))
zorro_iso_df['dmag_832'] = zorro_iso_df['M_832'] - M_832_TARGET_MODEL

fn_mass_to_dmag832 = PchipInterpolator(
    zorro_iso_df.mass.values, zorro_iso_df.dmag_832.values, extrapolate=False)
zorro_by_dmag = zorro_iso_df.sort_values('dmag_832')
fn_dmag832_to_mass = PchipInterpolator(
    zorro_by_dmag.dmag_832.values, zorro_by_dmag.mass.values, extrapolate=False)

gem_df = np.loadtxt(GEMINI_ZORRO_832, comments = "#", skiprows = 29)   # not a csv, its a .dat file... gotta extract properly 

sep_arcsec = gem_df[:,0]
sep_distance = sep_arcsec * DIST_PC
dmag = gem_df[:,1]

mass = fn_dmag832_to_mass(dmag)
dmag_tess = fn_mass_to_dmag_axis(mass)

ax.plot(sep_distance, dmag_tess, color='royalblue', lw=1.5,
        label='Gemini/Zorro 832nm')

ax.fill_between(sep_distance,
                dmag_tess,
                0,
                color='0.75',
                alpha=0.6)

# Inner working angle
ax.axvline(sep_distance.min(),
           color='royalblue',
           lw=2)

# Outer limit of the contrast curve
ax.axvline(sep_distance.max(),
           color='royalblue',
           lw=2)


### TESS TRANSIT DEPTH
### --------------------

delta_obs = 0.008
dmag_limit = -2.5*np.log10(2*delta_obs)
print(dmag_limit)

transit_localization = 2.5 * DIST_PC    # 2.5 arcsec example
print("TRASNTI LOCALIZATOION VERTICAL LINE = " + str(transit_localization))
# horizontal line
ax.axhline(dmag_limit,
           color='k',
           lw=2,
           label='Transit depth')

# vertical localization line
ax.axvline(transit_localization,
           color='k',
           lw=2)

xmin, xmax = ax.get_xlim()

y_bottom, y_top = ax.get_ylim()   # remember: inverted axis!

ax.fill_betweenx(
    np.linspace(dmag_limit, y_bottom, 100),
    transit_localization,
    xmax,
    color='0.75',
    alpha=0.5
)


fig.tight_layout()
fig.savefig(OUTPUT_PLOT, dpi=300)
print(f"saved {OUTPUT_PLOT}")
plt.close(fig)

# -----------------------------------------------------------------------
# NOTES
# -----------------------------------------------------------------------
# - Distance uncertainty (+0.4988/-0.2670 pc) is not propagated above; at
#   ~100 pc this is a ~0.5% effect on M_TESS_TARGET (~0.01 mag), almost
#   certainly negligible next to the isochrone/age systematics, but flagging
#   in case you want to add it to an error budget.
# - The MIST synthetic photometry (stellar range) is real bolometric-
#   correction-table photometry, not a blackbody -- this is the more
#   accurate half of the calculation now. The blackbody approximation is
#   still used for the Baraffe/COND substellar range, since those models
#   don't include TESS photometry tables; that remains the dominant
#   systematic in the mass->dmag mapping at low companion mass, and is
#   worth a caveat sentence in the paper if this ends up in it.
# - If your RV Msini values dip below the Baraffe/COND03 grid's minimum
#   tabulated mass, you'll see NaNs (see warning above) -- COND03 grids
#   typically extend down to ~1 Mjup, so this is more likely to bite at your
#   shortest periods/smallest semi-major axes.



ZORRO562_FILTER_PATH = os.path.join(gemini_dir, "Gemini_Zorro.EO_562.dat")

_zorro562_filt = None
def _load_zorro562_filter():
    global _zorro562_filt
    if _zorro562_filt is None:
        _zorro562_filt = pd.read_csv(ZORRO562_FILTER_PATH, sep=r'\s+',
                                      names=['wvlen_angst', 'transmission'], comment='#')
    return _zorro562_filt

def abs_mag_in_zorro562_blackbody(lum_lsun, teff_k):
    """Calculates absolute magnitude in the Gemini/Zorro 562nm band via blackbody."""
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
        M_X = M_bol_star - 2.5*np.log10((F_X/F_bol).decompose().value)
        M_Xs.append(M_X)
    return nparr(M_Xs)






# =======================================================================
# 5. Publication-Quality False Positive Validation Plot
# =======================================================================

import math
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Apply Publication Styling ---
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    # High-quality fallback style if scienceplots isn't installed
    plt.style.use('default')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.major.size': 6,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
    })

fig, ax = plt.subplots(figsize=(6, 4.8), dpi=300)

# -----------------------------------------------------------------------
# A. EXTRACT DATA & PLOT CURVES TO ESTABLISH NATURAL BOUNDS
# -----------------------------------------------------------------------

# 1. RV Data
sel_rv = ~rv_df['dmag_TESS'].isna()
rv_sma = rv_df.sma_au[sel_rv].values
rv_dmag = rv_df.dmag_TESS[sel_rv].values

# 2. Speckle Imaging Data (Gemini/Zorro 832nm)
gem_df = np.loadtxt(GEMINI_ZORRO_832, comments="#", skiprows=33)
sep_arcsec = gem_df[:, 0]
sep_au = sep_arcsec * DIST_PC
dmag_832 = gem_df[:, 1]
mass_zorro = fn_dmag832_to_mass(dmag_832)
dmag_tess_zorro = fn_mass_to_dmag_axis(mass_zorro)

### (Gemini/Zorro 562nm)
#gem2_df = np.loadtxt(GEMINI_ZORRO_562, comments = "#" )

# Plot the curves first so Matplotlib auto-scales the X-axis properly
ax.plot(rv_sma, rv_dmag, color='forestgreen', lw=2, label='PFS RVs', zorder=5)
ax.plot(sep_au, dmag_tess_zorro, color='mediumblue', lw=2, label='Gemini/Zorro 832nm', zorder=5)

ax.set_xscale('log')

# Now pull the dynamic limits so nothing gets cut off! --> IF WE EXECTURE THIS VERSION INSTEAD OF THE ONE JUST BELOW IN LINES 640 + 641.... THERE'S A BIT OF WHITESPACE THERE THEN TOWARD THE LEFT OF THE FINAL PLOT! 
#xmin, xmax = ax.get_xlim()

xmin = np.nanmin(rv_sma)
_, xmax = ax.get_xlim() # Keep Matplotlib's auto-scaled right edge for a moment

# 3. Transit Depth Exclusion Constants
delta_obs = 0.008
dmag_limit = -2.5 * np.log10(2 * delta_obs)
transit_local_au = 1.945 * DIST_PC  

# Ensure the x-axis extends far enough to show the transit localization limit
xmax = max(xmax, transit_local_au * 1.5)

# Calculate a dynamic y-bottom based on the deepest contrast limits
max_data_dmag = np.nanmax([np.nanmax(rv_dmag), np.nanmax(dmag_tess_zorro), dmag_limit])
y_bottom = 8.5 # Ensure it goes down to at least 7.0
y_top = 0.0

# -----------------------------------------------------------------------
# B. DRAW BOUNDARY LINES
# -----------------------------------------------------------------------

# Transit Depth Boundary (Top-Left Box Outline)
ax.plot([xmin, transit_local_au], [dmag_limit, dmag_limit], color='black', lw=2, label=r'Transit depth ($\delta_{\mathrm{TESS}}$)', zorder=6)
ax.plot([transit_local_au, transit_local_au], [dmag_limit, y_top], color='black', lw=2, zorder=6)

# --- Find where the Speckle curve *actually* starts (ignoring flat/zero regions) ---
# This finds the first index where the contrast is greater than 0.1 mag
start_idx = np.argmax(dmag_tess_zorro > 0.1669) 
true_iwa_au = sep_au[start_idx]
true_start_dmag = dmag_tess_zorro[start_idx]

# Left side wall (drops exactly where the curve starts diving)
ax.plot([true_iwa_au, true_iwa_au], [y_top, true_start_dmag], color='mediumblue', lw=2, zorder=5)

# Right side wall (Outer Working Angle)
ax.plot([sep_au.max(), sep_au.max()], [y_top, dmag_tess_zorro[-1]], color='mediumblue', lw=2, zorder=5)

'''
# Right side wall (Outer Working Angle)
ax.plot([sep_au.max(), sep_au.max()], [y_top, dmag_tess_zorro[-1]], color='mediumblue', lw=2, zorder=5)
ax.plot([sep_au.min(), sep_au.min()], [y_top, dmag_tess_zorro[0]], color='mediumblue', lw=2, zorder=5)
ax.plot([sep_au.max(), sep_au.max()], [y_top, dmag_tess_zorro[-1]], color='mediumblue', lw=2, zorder=5)
'''

# --- Gemini/Zorro 562nm Data ---
GEMINI_ZORRO_562 = os.path.join(gemini_dir, "TIC88297141_20260430_562.dat")
# NOTE: You will need to create/load a _load_zorro562_filter() and abs_mag_in_zorro562_blackbody() 
# function identical to your 832nm ones, just swapped to the 562nm transmission data.

gem_df_562 = np.loadtxt(GEMINI_ZORRO_562, comments="#", skiprows=32)
sep_au_562 = gem_df_562[:, 0] * DIST_PC
dmag_562 = gem_df_562[:, 1]

# Create the 562nm grid
zorro_iso_df['M_562'] = abs_mag_in_zorro562_blackbody(
    nparr(zorro_iso_df.lum), nparr(zorro_iso_df.teff))

M_562_TARGET_MODEL = float(np.interp(MSTAR_MSUN, zorro_iso_df.mass, zorro_iso_df.M_562))
zorro_iso_df['dmag_562'] = zorro_iso_df['M_562'] - M_562_TARGET_MODEL

# Inverse interpolator: 562 dmag -> Mass
zorro_by_dmag_562 = zorro_iso_df.sort_values('dmag_562')
fn_dmag562_to_mass = PchipInterpolator(
    zorro_by_dmag_562.dmag_562.values, zorro_by_dmag_562.mass.values, extrapolate=False)

# Convert 562 dmag -> Mass -> TESS dmag (using your new 562 functions)
mass_zorro_562 = fn_dmag562_to_mass(dmag_562) 
dmag_tess_zorro_562 = fn_mass_to_dmag_axis(mass_zorro_562)

# Plot the curve (dashed so it doesn't overpower the 832nm line)
ax.plot(sep_au_562, dmag_tess_zorro_562, color='cornflowerblue', lw=1.8, label='Gemini/Zorro 562nm', zorder=5)

# --- 562nm Inner Working Angle Wall ---
start_idx_562 = np.argmax(dmag_tess_zorro_562 > 0.1) 
true_iwa_au_562 = sep_au_562[start_idx_562]
true_start_dmag_562 = dmag_tess_zorro_562[start_idx_562]

ax.plot([true_iwa_au_562, true_iwa_au_562], [y_top, true_start_dmag_562], color='cornflowerblue', lw=1.8, zorder=5)
# (You usually don't need the outer wall for the shallower curve if the 832nm covers it)
# -----------------------------------------------------------------------
# C. STACKED OVERLAP SHADING (Multi-layered grey exclusion zones)
# -----------------------------------------------------------------------
EXCL_COLOR = 'black'
EXCL_ALPHA = 0.15

### Gaia contrast curve... interpolated w/ Web Plot Digitizer from Rizzutto et al. 2018

import pandas as pd 

gaia_path = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/gaia/gaia_contrast.csv"
gaia_df = pd.read_csv(gaia_path)

gaia_sep_arcsec = (gaia_df['sep_milliarcsec'].values) / 1000.
gaia_dmag = gaia_df['dmag'].values

# Convert separation to AU using your distance
gaia_sep_au = gaia_sep_arcsec * DIST_PC

# Plot the thick silver Gaia line (plotted directly on the TESS dmag axis 
# since the broad G and T bandpasses heavily overlap)
ax.plot(gaia_sep_au, gaia_dmag, color='brown', lw=2, label='Gaia', zorder=4)

# Draw the vertical wall to close the right side of the Gaia boundary
ax.plot([gaia_sep_au[-1], gaia_sep_au[-1]], [y_top, gaia_dmag[-1]], color='brown', lw=2, zorder=4)

# Shade the Gaia exclusion zone (using the same stacked grey effect!)
ax.fill_between(gaia_sep_au, gaia_dmag, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=1)


# 1. Transit Depth Exclusion (Inverted L-Shape)
# Fainter than depth limit (Bottom exclusion spanning the whole plot)
ax.fill_between([xmin, xmax], dmag_limit, y_bottom, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=1)
# Outside localization radius (Right exclusion from top down to dmag_limit)
ax.fill_between([transit_local_au, xmax], y_top, dmag_limit, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=1)

# 2. Speckle Imaging Exclusion (Shade from the blue curve UP to 0 mag)
ax.fill_between(sep_au, dmag_tess_zorro, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=2)

# Shade the 562nm exclusion zone
ax.fill_between(sep_au_562, dmag_tess_zorro_562, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=2)
# 3. RV Exclusion (Shade from the green curve UP to 0 mag)
ax.fill_between(rv_sma, rv_dmag, y_top, color=EXCL_COLOR, alpha=EXCL_ALPHA, zorder=3)


# -----------------------------------------------------------------------
# D. FORMATTING & SECONDARY AXIS
# -----------------------------------------------------------------------
# Lock in the axes (Note: setting ylim(y_bottom, y_top) automatically inverts the y-axis!)
ax.set_xlim(xmin, xmax)
ax.set_ylim(y_bottom, y_top)

ax.set_xlabel('Projected separation [AU]', fontsize=11, fontweight='medium')
ax.set_ylabel(r'Brightness contrast ($\Delta\mathrm{mag}$)', fontsize=11, fontweight='medium')

# Secondary Y-axis (Mass in M_sun)
secax = ax.secondary_yaxis(
    'right', functions=(fn_dmag_to_mass_calibrated, fn_mass_to_dmag_calibrated)
)
secax.set_ylabel(r'Companion mass [$M_\odot$]', fontsize=11, fontweight='medium')
secax.set_yticks(mass_ticks)
secax.set_yticklabels([f"{m:.2g}" for m in mass_ticks])

# Legend & Grid styling
ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='none', fontsize=9)
ax.xaxis.set_tick_params(which='both', direction='in', top=True)
ax.yaxis.set_tick_params(which='both', direction='in', right=True)

plt.title("Associated companions", fontsize=12, pad=10, fontweight='bold')
plt.tight_layout()

plt.savefig("dmag_vs_a.pdf", bbox_inches='tight')  # PDF vector output for the journal
print(f"Saved publication plot to pdf")