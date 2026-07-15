"""
make_latex_tables.py

Automates generation of AASTeX `deluxetable` parameter tables from a
juliet `posterior_samples` dictionary, for direct \\input into ms.tex.

Two ways to use it:

1) Called at the end of your fit script (recommended, see snippet in
   chat) -- runs right after `results = dataset.fit(...)`, no extra
   re-computation needed.

2) Standalone, from a previously saved run, WITHOUT re-running the
   (expensive) dynesty fit:

       python make_latex_tables.py --run-dir results/run_v1

   This requires that your fit script already saved
   `posterior_samples.pkl` and `run_metadata.pkl` inside that
   run directory (see snippet in chat).
"""

import os
import pickle
import argparse
from math import floor, log10

import numpy as np


# =========================================================
# EDIT THIS if your paper path ever changes
# =========================================================
PAPER_TABLES_DIR = "/home/gurmeher/gurmeher/Sco-Cen-Planets/tic88297141/paper/tables"


# ---------------------------------------------------------
# Rounding / formatting helpers
# ---------------------------------------------------------
def _decimals_for_sig_figs(x, sig=2):
    """
    Number of decimal places needed to show `x` (>0) to `sig`
    significant figures. Clipped at 0 -- for large uncertainties
    (>= 1) this just means "round to the nearest integer" rather than
    to the nearest ten/hundred. Good enough for most cases; if you
    have a jitter term with an uncertainty of e.g. 300, and you want
    it rounded to the nearest 100 rather than shown as an integer,
    swap the `round(...)` calls in format_med_err for
    `round(x, -k)` with a negative k instead.
    """
    if x == 0 or not np.isfinite(x):
        return 2
    return max(sig - int(floor(log10(abs(x)))) - 1, 0)


def format_med_err(med, hi_err, lo_err, sig=2):
    """
    Round both uncertainties to `sig` significant figures (using
    whichever of the two is smaller, so neither gets truncated),
    round the median to match, and return a LaTeX asymmetric-error
    string, e.g.

        format_med_err(4.644234, 0.000123, 0.000098)
        -> "$4.64423^{+0.00012}_{-0.00010}$"
    """
    positive_errs = [e for e in (hi_err, lo_err) if e > 0]
    smaller = min(positive_errs) if positive_errs else 0
    decimals = _decimals_for_sig_figs(smaller, sig=sig) if smaller > 0 else 2
    fmt = "{:." + str(decimals) + "f}"
    return (f"${fmt.format(round(med, decimals))}"
            f"^{{+{fmt.format(round(hi_err, decimals))}}}"
            f"_{{-{fmt.format(round(lo_err, decimals))}}}$")


def med_hi_lo(samples, key):
    med = np.median(samples[key])
    hi = np.percentile(samples[key], 84) - med
    lo = med - np.percentile(samples[key], 16)
    return med, hi, lo


def table_value(key, posterior_samples, priors, sig=2):
    """
    LaTeX-ready value string for `key`: posterior median +/- 1 sigma
    if it was a free parameter, or the fixed value from `priors` if
    it was held fixed in the fit.
    """
    if key not in priors:
        raise KeyError(
            f"'{key}' is not in `priors` -- check for a typo in the "
            f"parameter name, or that this run actually included it "
            f"(the parameter set changes between Model 1/2/3/4)."
        )
    dist = priors[key]['distribution']
    if dist == 'fixed' or key not in posterior_samples:
        fixed_val = priors[key]['hyperparameters']
        return f"{fixed_val} (fixed)"
    med, hi, lo = med_hi_lo(posterior_samples, key)
    return format_med_err(med, hi, lo, sig=sig)


# ---------------------------------------------------------
# Table builders (each returns a full deluxetable string)
# ---------------------------------------------------------
def build_planet_table(posterior_samples, priors, run_number, label="tab:planet"):
    rows = [
        (r"$P$ (days)",             "Orbital period",             'P_p1'),
        (r"$T_0$ (BTJD)",           "Transit center",             't0_p1'),
        (r"$R_p/R_*$",              "Planet-to-star radius ratio",'p_p1'),
        (r"$b$",                    "Impact parameter",           'b_p1'),
        (r"$\rho_*$ (kg m$^{-3}$)", "Stellar density",            'rho'),
        (r"$e$",                    "Eccentricity",                'ecc_p1'),
        (r"$\omega$ (deg)",         "Argument of periastron",     'omega_p1'),
        (r"$q_1$ (TESS)",           "Limb-darkening coefficient",  'q1_TESS'),
        (r"$q_2$ (TESS)",           "Limb-darkening coefficient",  'q2_TESS'),
    ]

    lines = [
        r"\begin{deluxetable}{lll}",
        r"\tablecaption{Planetary and Orbital Parameters (run v%d) \label{%s}}" % (run_number, label),
        r"\tablehead{\colhead{Parameter} & \colhead{Description} & \colhead{Value}}",
        r"\startdata",
    ]
    for name, desc, key in rows:
        lines.append(f"{name} & {desc} & {table_value(key, posterior_samples, priors)} \\\\")
    lines += [
        r"\enddata",
        r"\tablecomments{Values are the posterior median with 68\% (1$\sigma$) credible "
        r"intervals from the joint TESS + ground-based \texttt{juliet}/\texttt{dynesty} fit.}",
        r"\end{deluxetable}",
    ]
    return "\n".join(lines)


def build_gp_table(posterior_samples, priors, run_number, label="tab:gp"):
    rows = [
        (r"$B_{\rm GP}$ (TESS)",           "QP kernel amplitude",           'GP_B_TESS'),
        (r"$C_{\rm GP}$ (TESS)",           "QP kernel harmonic complexity", 'GP_C_TESS'),
        (r"$L_{\rm GP}$ (days, TESS)",     "QP kernel decay timescale",     'GP_L_TESS'),
        (r"$P_{\rm rot,GP}$ (days, TESS)", "QP kernel rotation period",     'GP_Prot_TESS'),
        (r"$\sigma_w$ (TESS)",             "White-noise jitter",            'sigma_w_TESS'),
        (r"$m_{\rm flux}$ (TESS)",         "Flux offset",                   'mflux_TESS'),
        (r"$m_{\rm dilution}$ (TESS)",     "Dilution factor",               'mdilution_TESS'),
    ]

    lines = [
        r"\begin{deluxetable}{lll}",
        r"\tablecaption{TESS GP Hyperparameters and Nuisance Parameters (run v%d) \label{%s}}" % (run_number, label),
        r"\tablehead{\colhead{Parameter} & \colhead{Description} & \colhead{Value}}",
        r"\startdata",
    ]
    for name, desc, key in rows:
        lines.append(f"{name} & {desc} & {table_value(key, posterior_samples, priors)} \\\\")
    lines += [r"\enddata", r"\end{deluxetable}"]
    return "\n".join(lines)


def build_instrument_table(posterior_samples, priors, ground_telescopes, run_number, label="tab:instruments"):
    lines = [
        r"\begin{deluxetable}{lccccc}",
        r"\tablecaption{Instrumental Nuisance Parameters (run v%d) \label{%s}}" % (run_number, label),
        r"\tablehead{\colhead{Instrument} & \colhead{$m_{\rm dilution}$} & \colhead{$m_{\rm flux}$} & "
        r"\colhead{$\sigma_w$} & \colhead{$\theta_0$} & \colhead{$\theta_1$}}",
        r"\startdata",
    ]
    for tel in ground_telescopes:
        row = [tel]
        for p in ['mdilution', 'mflux', 'sigma_w', 'theta0', 'theta1']:
            row.append(table_value(f'{p}_{tel}', posterior_samples, priors))
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\enddata",
        r"\tablecomments{$\theta_0$, $\theta_1$ are the linear and quadratic coefficients of the "
        r"second-order polynomial detrending term applied to each ground-based light curve.}",
        r"\end{deluxetable}",
    ]
    return "\n".join(lines)


def build_ld_table(posterior_samples, priors, solo_ld_telescopes, g_band_key, i_band_key,
                    run_number, label="tab:ld"):
    rows = [("TESS", 'q1_TESS', 'q2_TESS')]
    for tel in solo_ld_telescopes:
        rows.append((tel, f'q1_{tel}', f'q2_{tel}'))
    rows.append((f"g-band ({g_band_key})", f'q1_{g_band_key}', f'q2_{g_band_key}'))
    rows.append((f"i-band ({i_band_key})", f'q1_{i_band_key}', f'q2_{i_band_key}'))

    lines = [
        r"\begin{deluxetable}{lcc}",
        r"\tablecaption{Limb-Darkening Coefficients (Kipping 2013 parameterization, run v%d) \label{%s}}" % (run_number, label),
        r"\tablehead{\colhead{Instrument / Band} & \colhead{$q_1$} & \colhead{$q_2$}}",
        r"\startdata",
    ]
    for name, k1, k2 in rows:
        v1 = table_value(k1, posterior_samples, priors)
        v2 = table_value(k2, posterior_samples, priors)
        lines.append(f"{name} & {v1} & {v2} \\\\")
    lines += [r"\enddata", r"\end{deluxetable}"]
    return "\n".join(lines)


# ---------------------------------------------------------
# Top-level writer
# ---------------------------------------------------------
def write_tables(posterior_samples, priors, ground_telescopes, solo_ld_telescopes,
                  g_band_key, i_band_key, results_dir, run_number,
                  paper_tables_dir=PAPER_TABLES_DIR):
    """
    Writes 4 .tex files to TWO places:
      1) {results_dir}/tables/   -- versioned copy, tied to this run_number
      2) {paper_tables_dir}/     -- fixed-path copy, overwritten every run,
                                     so \\input paths in ms.tex never change

    Returns the dict {filename: latex_text} in case you want to log or
    inspect it (e.g. print into your .log file).
    """
    tables = {
        'tab_planet.tex':      build_planet_table(posterior_samples, priors, run_number),
        'tab_gp.tex':          build_gp_table(posterior_samples, priors, run_number),
        'tab_instruments.tex': build_instrument_table(posterior_samples, priors, ground_telescopes, run_number),
        'tab_ld.tex':          build_ld_table(posterior_samples, priors, solo_ld_telescopes,
                                               g_band_key, i_band_key, run_number),
    }

    versioned_dir = os.path.join(results_dir, 'tables')
    os.makedirs(versioned_dir, exist_ok=True)
    os.makedirs(paper_tables_dir, exist_ok=True)

    for fname, text in tables.items():
        for out_dir in (versioned_dir, paper_tables_dir):
            outpath = os.path.join(out_dir, fname)
            with open(outpath, 'w') as f:
                f.write(text + "\n")
            print(f"Wrote: {outpath}")

    return tables


# ---------------------------------------------------------
# Standalone entry point: regenerate tables from a saved run
# without re-running the (expensive) dynesty fit.
# ---------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Regenerate LaTeX parameter tables from a saved juliet run."
    )
    parser.add_argument('--run-dir', required=True,
                         help="e.g. results/run_v1 -- must contain "
                              "posterior_samples.pkl and run_metadata.pkl")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, 'posterior_samples.pkl'), 'rb') as f:
        posterior_samples = pickle.load(f)
    with open(os.path.join(args.run_dir, 'run_metadata.pkl'), 'rb') as f:
        meta = pickle.load(f)

    write_tables(
        posterior_samples=posterior_samples,
        priors=meta['priors'],
        ground_telescopes=meta['ground_telescopes'],
        solo_ld_telescopes=meta['solo_ld_telescopes'],
        g_band_key=meta['g_band_key'],
        i_band_key=meta['i_band_key'],
        results_dir=args.run_dir,
        run_number=meta['run_number'],
    )