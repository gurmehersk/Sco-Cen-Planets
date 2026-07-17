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

RV_SENSITIVITY_CSV = "./alternate_analysis_approach/mass_vs_a.csv"

TESS_FILTER_PATH = os.path.join(DATADIR, "TESS_TESS.Red.dat")
# download: http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=TESS/TESS.Red ---> DONE! 
# (two whitespace-separated columns: wavelength [Angstrom], transmission [0-1], no header)
# only needed for the substellar (Baraffe/COND) blackbody calculation below.

### Why? Because the baraffe csv does not give u the tess magnitude.. So you need something that can do the conversion. Luckily we have tess filters that can do that for us!!  

MIST_ISOCMD_PATH = os.path.join(DATADIR, "MIST_iso_6a556a38f0b10/MIST_iso_6a556a38f0b10.iso.UBVRIplus")
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


# -----------------------------------------------------------------------
# Inverse mapping (dmag -> mass) for the secondary axis, built from the
# same merged isochrone grid used for fn_mass_to_dmag
# -----------------------------------------------------------------------

iso_df_inv = iso_df.dropna(subset=['dmag_TESS']).sort_values('dmag_TESS')

# -----------------------------------------------------------------------
# Diagnose + flatten any non-monotonicity in dmag_TESS vs mass before
# building the inverse (dmag -> mass) mapping for the secondary axis
# -----------------------------------------------------------------------

chk = iso_df.dropna(subset=['dmag_TESS']).sort_values('mass').reset_index(drop=True)
d = np.diff(nparr(chk['dmag_TESS']))

# dmag_TESS should be monotonically non-increasing as mass increases
bad = np.where(d > 0)[0]

if len(bad):
    idx = np.unique(np.concatenate([bad, bad + 1]))
    print(f"WARNING: {len(bad)} non-monotonic point(s) in dmag_TESS(mass); "
          f"flattening for the inverse (secondary-axis) mapping only. "
          f"Affected mass range: {chk['mass'].iloc[idx].min():.5f} - "
          f"{chk['mass'].iloc[idx].max():.5f} Msun")
    print(chk.iloc[idx][['mass', 'M_TESS', 'dmag_TESS']].to_string())

# enforce monotonic non-increasing sequence for inversion purposes
# (does NOT touch iso_df / fn_mass_to_dmag -- forward mapping stays exact)
dmag_monotonic = np.minimum.accumulate(nparr(chk['dmag_TESS']))
chk['dmag_TESS_monotonic'] = dmag_monotonic

iso_df_inv = chk.drop_duplicates(subset=['dmag_TESS_monotonic']) \
                .sort_values('dmag_TESS_monotonic')

fn_dmag_to_mass = interp1d(
    nparr(iso_df_inv['dmag_TESS_monotonic']), nparr(iso_df_inv['mass']),
    kind='linear', bounds_error=False,
    fill_value=(iso_df_inv['mass'].iloc[0], iso_df_inv['mass'].iloc[-1])
)
fn_dmag_to_mass = interp1d(
    nparr(iso_df_inv['dmag_TESS']), nparr(iso_df_inv['mass']),
    kind='quadratic', bounds_error=False,
    fill_value=(iso_df_inv['mass'].iloc[0], iso_df_inv['mass'].iloc[-1])  # clip at grid edges instead of NaN
)

# -----------------------------------------------------------------------
# 5. Plot
# -----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 4))

sel = ~rv_df['dmag_TESS'].isna()
ax.plot(rv_df.sma_au[sel], rv_df.dmag_TESS[sel], color='C0', lw=1.5)
ax.fill_between(rv_df.sma_au[sel], rv_df.dmag_TESS[sel], 0,
                 alpha=0.3, color='C0', label=r'excluded by RV (3$\sigma$)')

ax.set_xscale('log')
ax.set_xlabel('semi-major axis (AU)')
ax.set_ylabel(r'$\Delta T$ (mag)')
ax.invert_yaxis()   # brighter = up, standard convention for contrast plots
ax.legend()

# secondary axis: functions=(forward, inverse) where forward maps the
# PRIMARY axis (dmag) -> SECONDARY axis (mass), inverse goes back
secax = ax.secondary_yaxis('right', functions=(fn_dmag_to_mass, fn_mass_to_dmag))
secax.set_ylabel(r'companion mass ($M_\odot$)')


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