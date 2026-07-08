"""
TIC 88297141b -- RV non-detection / upper-limit analysis

Key changes from the previous version:
  1. Epoch-bin the 12 raw points into their true 4 independent epochs
     (weighted mean + propagated error) before fitting. Fitting 12 points
     as if independent overweights the triplicate exposures and biases
     both K and sigma_w.
  2. Run K prior CASE 1 (U[0,1000], physical) and CASE 2 (U[-1000,1000],
     symmetric -- avoids the Lucy & Sweeney boundary bias at K=0) as two
     separate fits, both on the binned data.
  3. Run a flat model (K_p1 fixed = 0) and compare ln Z against both
     Keplerian fits -- this tells you whether a signal is statistically
     preferred at all, independent of what the K posterior looks like.
  4. Derive 1/2/3-sigma upper limits on K for both prior cases (folding
     to |K| for the symmetric case) and convert to Mp sin i / Mp limits.
  5. Posterior predictive check: draw ~100 random (K, mu) samples from
     the chains and overplot the resulting curves on the phase-folded
     data, per your own TODO comment.
"""

import numpy as np
import matplotlib.pyplot as plt
import juliet
import corner

# ---------------------------------------------------------------
# 0. Load raw data
# ---------------------------------------------------------------
rvfname = "TIC88297141.vels"

raw = np.loadtxt(rvfname, usecols=(0, 1, 2), unpack=False)
t_raw, rv_raw, err_raw = raw[:, 0], raw[:, 1], raw[:, 2]

offset = 2457000.
t_raw = t_raw - offset

t0 = 3803.2401428842
period = 4.6445890270

# Only rewrite the file (adding the instrument column) if it isn't
# already tagged -- avoids re-appending "PFS" on every rerun.
with open(rvfname) as f:
    first_line = f.readline().split()
if len(first_line) < 4:
    with open(rvfname, "w") as f:
        for ti, rvi, erri in zip(t_raw + offset, rv_raw, err_raw):
            f.write(f"{ti:.6f} {rvi:.3f} {erri:.3f} PFS\n")

print(f"Loaded {len(t_raw)} raw points")

# ---------------------------------------------------------------
# 1. Bin into true epochs (your 3-points-per-visit structure)
# ---------------------------------------------------------------
def bin_by_epoch(t, rv, err, gap_threshold=0.3):
    """
    Group points into epochs by sequential time gaps and return the
    inverse-variance-weighted mean RV per epoch. gap_threshold is in
    days -- tune this to your actual within-night cadence (default
    assumes points >~7 hr apart belong to different epochs).
    """
    order = np.argsort(t)
    t, rv, err = t[order], rv[order], err[order]

    groups, current = [], [0]
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > gap_threshold:
            groups.append(current)
            current = [i]
        else:
            current.append(i)
    groups.append(current)

    t_bin, rv_bin, err_bin = [], [], []
    print("\n--- Epoch binning diagnostics ---")
    for g in groups:
        w = 1.0 / err[g] ** 2
        rv_w = np.sum(rv[g] * w) / np.sum(w)
        err_w = np.sqrt(1.0 / np.sum(w))
        chi2_red = np.sum(w * (rv[g] - rv_w) ** 2) / (len(g) - 1)
        print(f"{chi2_red} = chi2, if >1, introduce fudge factor to err_w ")
        if chi2_red > 1:
            err_w *= np.sqrt(chi2_red)
        t_w = np.mean(t[g])
        t_bin.append(t_w)
        rv_bin.append(rv_w)
        err_bin.append(err_w)

        msg = (f"epoch t={t_w:9.3f}  N={len(g)}  "
               f"weighted_mean={rv_w:8.2f}  formal_err={err_w:6.2f}")
        if len(g) > 1:
            msg += f"  raw_std={np.std(rv[g]):8.2f}  raw_errs={np.round(err[g],1)}"
        print(msg)
    
    return np.array(t_bin), np.array(rv_bin), np.array(err_bin)


t_bin, rv_bin, err_bin = bin_by_epoch(t_raw, rv_raw, err_raw, gap_threshold=0.3)
print(f"\nBinned {len(t_raw)} raw points -> {len(t_bin)} independent epochs\n")

# check: if raw_std within an epoch is >> formal error, that scatter is
# short-timescale noise and should NOT be folded into sigma_w -- it's
# telling you something about intra-night systematics, not orbital motion.

binned_fname = "TIC88297141_binned.vels"
with open(binned_fname, "w") as f:
    for ti, rvi, erri in zip(t_bin + offset, rv_bin, err_bin):
        f.write(f"{ti:.6f} {rvi:.3f} {erri:.3f} PFS\n")

# ---------------------------------------------------------------
# 2. Fit helper
# ---------------------------------------------------------------
def run_fit(rvfilename, out_folder, K_dist, K_hyperp, n_live_points=1500):
    priors = {}
    params = ['P_p1', 't0_p1', 'K_p1', 'ecc_p1', 'omega_p1',
              'mu_PFS', 'sigma_w_PFS']
    dists = ['fixed', 'fixed', K_dist, 'fixed', 'fixed',
             'uniform', 'loguniform']
    hyperps = [period, t0, K_hyperp, 0., 90.,
               [-1000., 1000.], [1e-3, 1000.]]

    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {'distribution': dist, 'hyperparameters': hyperp}

    dataset = juliet.load(priors=priors, rvfilename=rvfilename,
                           out_folder=out_folder)
    results = dataset.fit(n_live_points=n_live_points)
    return dataset, results

run_number = 16
# CASE 1: physical, non-negative K
dataset_pos, results_pos = run_fit(
    binned_fname, f"rv_fit_v{run_number}_Kpos", 'uniform', [0., 1000.])

# CASE 2: symmetric K, avoids the K>=0 boundary bias
dataset_sym, results_sym = run_fit(
    binned_fname, f"rv_fit_v{run_number}_Ksym", 'uniform', [-1000., 1000.])

# CASE 3: flat model, no planet -- for the evidence comparison
dataset_flat, results_flat = run_fit(
    binned_fname, f"rv_fit_v{run_number}_flat", 'fixed', 0.0)

# ---------------------------------------------------------------
# 3. Evidence comparison (is a signal preferred at all?)
# ---------------------------------------------------------------
# NOTE: attribute name has varied slightly across juliet versions --
# if this KeyErrors, run print(results_pos.posteriors.keys()) to check.
lnZ_pos = results_pos.posteriors['lnZ']
lnZ_sym = results_sym.posteriors['lnZ']
lnZ_flat = results_flat.posteriors['lnZ']

print("\n--- Evidence comparison ---")
print(f"ln Z (flat, no planet)     = {lnZ_flat:.2f}")
print(f"ln Z (K in [0,1000])       = {lnZ_pos:.2f}   "
      f"Delta ln Z = {lnZ_pos - lnZ_flat:.2f}")
print(f"ln Z (K in [-1000,1000])   = {lnZ_sym:.2f}   "
      f"Delta ln Z = {lnZ_sym - lnZ_flat:.2f}")

print("Rule of thumb: Delta ln Z >~ 3-5 needed to call this a real detection.\n")

# ---------------------------------------------------------------
# 4. Upper limits on K (1/2/3 sigma)
# ---------------------------------------------------------------
sigma_pctiles = {'1sigma': 84.13, '2sigma': 97.72, '3sigma': 99.87}

K_pos_samples = results_pos.posteriors['posterior_samples']['K_p1']
K_sym_samples = results_sym.posteriors['posterior_samples']['K_p1']
K_sym_abs = np.abs(K_sym_samples)  # fold: symmetric K is a phase ambiguity,
                                    # physical amplitude is |K|

print("--- K upper limits [m/s] ---")
K_limits = {}
for label, samples in [('U[0,1000]', K_pos_samples),
                        ('|U[-1000,1000]|', K_sym_abs)]:
    K_limits[label] = {}
    for sig, pct in sigma_pctiles.items():
        val = np.percentile(samples, pct)
        K_limits[label][sig] = val
        print(f"{label:>18s}  {sig}: K < {val:7.2f} m/s")

# ---------------------------------------------------------------
# 5. Convert K upper limits to planet mass upper limits
# ---------------------------------------------------------------
MJUP_TO_MEARTH = 317.83

def semiamp_to_mass(K, P_days, Mstar_msun, ecc=0.0, incl_deg=90.0):
    """
    K [m/s], P [days], Mstar [Msun] -> companion mass.
    incl_deg=90 assumes sin(i)=1 (Mp sin i == Mp). Pass your transit-fit
    inclination (i = arccos(b * Rstar/a)) here for the strict Mp.
    """
    P_yr = P_days / 365.25
    Mp_sini_mjup = K * np.sqrt(1 - ecc ** 2) * P_yr ** (1 / 3) * Mstar_msun ** (2 / 3) / 28.4329
    sin_i = np.sin(np.radians(incl_deg))
    Mp_true_mjup = Mp_sini_mjup / sin_i
    return {
        'Mp_sini_mjup': Mp_sini_mjup,
        'Mp_sini_mearth': Mp_sini_mjup * MJUP_TO_MEARTH,
        'Mp_true_mjup': Mp_true_mjup,
        'Mp_true_mearth': Mp_true_mjup * MJUP_TO_MEARTH,
    }

Mstar_msun = 0.3          
incl_deg = 90.0           
# -----------------------------------------------------

print("\n--- Mass upper limits ---")
for label, lims in K_limits.items():
    print(f"\n{label}:")
    for sig, K_val in lims.items():
        m = semiamp_to_mass(K_val, period, Mstar_msun, incl_deg=incl_deg)
        print(f"  {sig}: Mp sin i < {m['Mp_sini_mearth']:6.2f} Mearth "
              f"({m['Mp_sini_mjup']:.3f} Mjup)   "
              f"| Mp (incl-corrected) < {m['Mp_true_mearth']:6.2f} Mearth")

# ---------------------------------------------------------------
# 6. Phase-folded data + posterior predictive check (100 draws)
# ---------------------------------------------------------------
def circular_rv_model(t, P, t0, K, mu):
    # Exact for ecc=0, omega=90 deg (matches your fixed priors above)
    return -K * np.sin(2 * np.pi * (t - t0) / P) + mu


def phase_fold(t, P, t0):
    ph = ((t - t0) % P) / P
    ph[ph > 0.5] -= 1.0
    return ph


for label, results, samples_K in [
    ("Kpos", results_pos, K_pos_samples),
    ("Ksym", results_sym, K_sym_samples),
]:
    mu_samples = results.posteriors['posterior_samples']['mu_PFS']

    n_draws = 100
    idx = np.random.choice(len(samples_K), size=n_draws, replace=False)

    t_grid = np.linspace(t0 - period / 2, t0 + period / 2, 500)
    phase_grid = phase_fold(t_grid, period, t0)
    order = np.argsort(phase_grid)

    plt.figure(figsize=(8, 5))
    for i in idx:
        curve = circular_rv_model(t_grid, period, t0, samples_K[i], mu_samples[i])
        plt.plot(phase_grid[order], curve[order], color='gray', alpha=0.08, lw=1)

    phase_data = phase_fold(t_bin, period, t0)
    plt.errorbar(phase_data, rv_bin, yerr=err_bin, fmt='o', color='C0',
                 zorder=10, label='binned PFS (4 epochs)')

    plt.xlabel('Phase')
    plt.ylabel('RV [m/s]')
    plt.title(f'Posterior predictive check -- {label}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'posterior_draws_{label}_binned.png', dpi=150)
    plt.close()

    # corner plot
    params_to_plot = ['K_p1', 'mu_PFS', 'sigma_w_PFS']
    corner_samples = np.vstack(
        [results.posteriors['posterior_samples'][p] for p in params_to_plot]).T
    fig = corner.corner(corner_samples, labels=params_to_plot,
                         quantiles=[0.16, 0.5, 0.84], show_titles=True)
    fig.savefig(f'corner_rv_{label}_binned.png', dpi=150)
    plt.close(fig)

print("\nSaved: posterior_draws_Kpos.png, posterior_draws_Ksym.png, "
      "corner_rv_Kpos.png, corner_rv_Ksym.png")