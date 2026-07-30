"""
TIC 88297141 -- Radial Velocity Linear Trend Analysis (Model 3)
==================================================================

PURPOSE
-------
Given poor MCMC convergence in the free-period Keplerian search
(Model 2), this script derives an ANALYTIC, non-MCMC upper limit on
any long-term RV drift by fitting a straight line to the binned PFS
epochs. This replaces the previously eyeballed "1 km/s" input to
rv_limit_contour.py with an actual, data-derived number.

IMPORTANT CAVEAT: a linear trend is only a sensible proxy for a
companion whose orbital period is much longer than our observing
baseline (~43 days) -- in that regime, the true (sinusoidal) RV curve
looks locally linear over our baseline. This method does NOT test the
transiting planet's own P_orb = 4.6446-day signal (the baseline spans
~9 of its periods, so a real signal at that period would look like
scatter, not a slope); that specific test is what the fixed-period
Model 1 fit is for.

METHOD
------
1. Fit a weighted straight line RV(t) = a + b*(t - t_ref) to the 4
   binned epochs, using inverse-variance weights (1/sigma^2) -- done
   by hand below (normal equations), so every step is visible rather
   than hidden inside a library call.
2. Check whether the formal per-epoch errors actually explain the
   point-to-point scatter (they don't, here -- reduced chi^2 is far
   above 1). This is the SAME phenomenon your epoch-binning function
   already corrects for one level down (intra-night scatter); here
   we apply the identical "fudge factor" logic one level up
   (inter-epoch scatter), rather than trusting the raw fit
   uncertainty at face value.
3. Take an n-sigma upper limit on the resulting (rescaled) slope
   uncertainty as the bound on |dv/dt|.
4. For an assumed CIRCULAR orbit with period P much longer than the
   baseline, the maximum possible instantaneous slope of the RV curve
   is K*(2*pi/P) (reached at the RV=0 crossing). Rearranging:
       K < (dv/dt limit) * P / (2*pi)
   This turns the single dv/dt limit into a full (K, a) exclusion
   curve as a function of assumed outer period / semi-major axis,
   directly comparable to the other curves in the associated-
   companions plot.

USAGE
-----
Edit the DATA and CONFIG blocks below to match your table / adopted
significance level, then run:
    python m3_rv_trend_limit.py
"""

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. DATA -- Table 1: binned PFS epochs (inverse-variance-weighted
#    mean of 3 consecutive same-night exposures per epoch)
# =====================================================================
t_bjd = np.array([2461156.823, 2461157.838, 2461159.813, 2461199.581])
rv    = np.array([117.74, 72.52, -527.75, -363.77])   # m/s
err   = np.array([18.38, 16.51, 17.48, 20.56])        # m/s, formal per-epoch error

# =====================================================================
# 2. CONFIG
# =====================================================================
P_orb_days = 4.6445890270   # transiting planet's period -- NOT what this
                            # script constrains, see caveat above
Mstar_msun = 0.30
N_SIGMA    = 3              # confidence level adopted for the final dv/dt limit
OUTER_PERIODS_DAYS = np.logspace(np.log10(50), np.log10(3650), 200)  # ~50 d to 10 yr

# =====================================================================
# 3. WEIGHTED LINEAR LEAST-SQUARES FIT (by hand -- normal equations)
# =====================================================================
# We're fitting  rv_i = a + b*x_i,  where x_i = t_i - t_ref,
# with weights w_i = 1/err_i^2 (inverse-variance weighting).
#
# The weighted least-squares solution minimizes:
#     chi^2 = sum_i  w_i * (rv_i - a - b*x_i)^2
#
# Setting d(chi^2)/da = 0 and d(chi^2)/db = 0 gives two linear
# ("normal") equations in a and b, solved in closed form using these
# weighted sums:
t_ref = t_bjd[0]
x = t_bjd - t_ref
w = 1.0 / err**2

S   = np.sum(w)           # sum of weights
Sx  = np.sum(w * x)       # weighted sum of x
Sy  = np.sum(w * rv)      # weighted sum of y
Sxx = np.sum(w * x**2)    # weighted sum of x^2
Sxy = np.sum(w * x * rv)  # weighted sum of x*y

Delta = S * Sxx - Sx**2   # determinant of the normal-equations matrix

b = (S * Sxy - Sx * Sy) / Delta     # slope      [m/s per day]
a = (Sxx * Sy - Sx * Sxy) / Delta   # intercept  [m/s]

# Formal (pre-rescaling) uncertainties on a and b, from standard error
# propagation through the normal equations:
sigma_b_formal = np.sqrt(S / Delta)
sigma_a_formal = np.sqrt(Sxx / Delta)

print("=" * 70)
print("STEP 1: raw weighted linear fit (formal errors only)")
print("=" * 70)
print(f"  slope     b = {b:+.3f} +/- {sigma_b_formal:.3f}  m/s/day")
print(f"  intercept a = {a:+.3f} +/- {sigma_a_formal:.3f}  m/s")

# =====================================================================
# 4. CHECK WHETHER THE FORMAL ERRORS ACTUALLY EXPLAIN THE SCATTER
# =====================================================================
# If chi^2/dof >> 1, there's real point-to-point scatter (jitter) the
# quoted per-epoch errors don't capture -- exactly the situation your
# juliet fit handles with a free sigma_w_PFS term. We check this for
# the flat/no-trend model too, since with only 4 points and 2 line
# parameters, dof=2 alone is barely informative.
model_line = a + b * x
chi2_line  = np.sum(w * (rv - model_line) ** 2)
dof_line   = len(t_bjd) - 2   # N points minus (slope, intercept)

mu_flat   = Sy / S                        # weighted mean = best flat-model fit
chi2_flat = np.sum(w * (rv - mu_flat) ** 2)
dof_flat  = len(t_bjd) - 1

print()
print("=" * 70)
print("STEP 2: is there excess scatter beyond the formal errors?")
print("=" * 70)
print(f"  Line fit:  chi2 = {chi2_line:.1f}, dof = {dof_line}, "
      f"reduced chi2 = {chi2_line / dof_line:.1f}")
print(f"  Flat fit:  chi2 = {chi2_flat:.1f}, dof = {dof_flat}, "
      f"reduced chi2 = {chi2_flat / dof_flat:.1f}")
print("  (Both >> 1: real epoch-to-epoch jitter, not just measurement noise.)")

# =====================================================================
# 5. RESCALE THE SLOPE UNCERTAINTY TO ACCOUNT FOR THIS EXCESS SCATTER
# =====================================================================
# Standard practice when reduced chi^2 > 1 and you have no independent
# handle on the jitter: inflate the parameter uncertainty by
# sqrt(reduced chi^2) from the fit itself. This is the identical
# "fudge factor" your epoch-binning function already applies one
# level down (intra-night -> inter-epoch); here it's applied between
# epochs instead.
fudge = np.sqrt(chi2_line / dof_line)
sigma_b_rescaled = sigma_b_formal * fudge

print()
print("=" * 70)
print("STEP 3: rescaled (honest) slope uncertainty")
print("=" * 70)
print(f"  fudge factor = sqrt(reduced chi2) = {fudge:.3f}")
print(f"  slope b = {b:+.3f} +/- {sigma_b_rescaled:.3f}  m/s/day  (rescaled)")

for n in (1, 2, 3):
    print(f"  {n}-sigma upper limit on |dv/dt|: "
          f"{abs(b) + n * sigma_b_rescaled:.2f} m/s/day")

baseline_days = x[-1]
dvdt_limit = abs(b) + N_SIGMA * sigma_b_rescaled
print()
print(f"  ==> Adopting {N_SIGMA}-sigma limit: |dv/dt| < {dvdt_limit:.2f} m/s/day")
print(f"      (baseline = {baseline_days:.1f} days; total Delta-RV bound "
      f"= {dvdt_limit * baseline_days / 1000:.3f} km/s over the full baseline)")

# =====================================================================
# 6. CONVERT dv/dt LIMIT -> (K, a) EXCLUSION CURVE FOR AN OUTER,
#    CIRCULAR-ORBIT COMPANION
# =====================================================================
# For a circular orbit, RV(t) = K * sin(2*pi*(t-t0)/P).
# Its time derivative is d(RV)/dt = K*(2*pi/P)*cos(2*pi*(t-t0)/P),
# which is MAXIMIZED (at the RV=0 crossing) at K*(2*pi/P).
#
# So if we observe |dv/dt| < dvdt_limit, and the companion's period P
# is much longer than our ~43-day baseline (so the sinusoid really
# does look locally linear over it), the largest K consistent with
# that limit is:
#       K_limit(P) = dvdt_limit * P / (2*pi)
#
# Each assumed period P is converted to a semi-major axis via
# Kepler's third law (a[AU]^3 = Mstar[Msun] * P[yr]^2), so the result
# is directly comparable to the AU-separation x-axis used elsewhere
# in the false-positive plot.
def K_limit_from_slope(dvdt_limit_ms_per_day, P_days):
    return dvdt_limit_ms_per_day * P_days / (2 * np.pi)


def semimajor_axis_au(P_days, Mstar_msun):
    P_yr = P_days / 365.25
    return (Mstar_msun * P_yr ** 2) ** (1 / 3)


K_curve = K_limit_from_slope(dvdt_limit, OUTER_PERIODS_DAYS)
a_curve = semimajor_axis_au(OUTER_PERIODS_DAYS, Mstar_msun)

# Save the curve for use in your existing associated-companions plot,
# in place of the old eyeballed-1-km/s version
np.savetxt(
    "rv_trend_limit_curve.csv",
    np.column_stack([OUTER_PERIODS_DAYS, a_curve, K_curve]),
    header="period_days,sma_au,K_limit_ms",
    delimiter=",",
    comments="",
)
print(f"\nSaved (period, a, K_limit) curve to rv_trend_limit_curve.csv "
      f"({len(OUTER_PERIODS_DAYS)} points)")

# =====================================================================
# 7. SANITY-CHECK PLOTS
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: data + best-fit line, so you can SEE what "reduced chi2 ~ 400"
# actually looks like -- a nearly flat/shallow line through wildly
# scattered points, not a real trend.
ax = axes[0]
ax.errorbar(x, rv, yerr=err, fmt='o', color='C0', label='binned PFS epochs')
xx = np.linspace(x.min() - 2, x.max() + 2, 100)
ax.plot(xx, a + b * xx, color='C1', label=f'weighted fit (b={b:.1f} m/s/day)')
ax.set_xlabel(f'days since BJD {t_ref:.3f}')
ax.set_ylabel('RV [m/s]')
ax.set_title('Weighted linear fit to binned RVs')
ax.legend()

# Right: resulting (a, K) exclusion curve for outer companions
ax = axes[1]
ax.plot(a_curve, K_curve, color='C2')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('assumed semi-major axis [AU]')
ax.set_ylabel(f'K limit [m/s] ({N_SIGMA}-sigma)')
ax.set_title('Analytic outer-companion K(a) bound')

fig.tight_layout()
fig.savefig("m3_rv_trend_diagnostic.png", dpi=150)
print("Saved diagnostic plot to m3_rv_trend_diagnostic.png")