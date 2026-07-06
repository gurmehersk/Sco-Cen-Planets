import juliet 

import numpy as np


priors = {}
rvfname = "TIC88297141.vels"
t, rv, rv_err = np.loadtxt(
    rvfname,
    usecols=(0,1,2),
    unpack=True
)
import numpy as np

t, rv, err = np.loadtxt(rvfname, usecols=(0, 1, 2), unpack=True)

with open(rvfname, "w") as f:
    for ti, rvi, erri in zip(t, rv, err):
        f.write(f"{ti:.6f} {rvi:.3f} {erri:.3f} PFS\n")
print(len(t))
print(rv)

offset = 2457000.

t = t - offset

t_0 = 3803.2401428842 
period = 4.6445890270

# we will fix P and t0, not fit for them.
params = ['P_p1', 't0_p1', 'K_p1', 'ecc_p1', 'omega_p1',
          'mu_PFS', 'sigma_w_PFS']

dists  = ['fixed', 'fixed', 'uniform', 'fixed', 'fixed',
          'uniform', 'loguniform']

hyperps = [period, t_0, [0., 100.], 0., 90.,
           [-100., 100.], [1e-3, 1000.]]

for param, dist, hyperp in zip(params, dists, hyperps):
    priors[param] = {}
    priors[param]['distribution'] = dist
    priors[param]['hyperparameters'] = hyperp

dataset = juliet.load(priors=priors, rvfilename= rvfname, out_folder='rv_fit')
results = dataset.fit(n_live_points=300)

# Posterior samples for every fitted parameter
posteriors = results.posteriors['posterior_samples']

# e.g. grab K posterior
K_samples = posteriors['K_p1']
print(np.median(K_samples), np.std(K_samples))


import matplotlib.pyplot as plt
import numpy as np

P  = period  
t0 = t_0  

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True,
                                gridspec_kw={'height_ratios':[3,1]})

for instrument in dataset.inames_rv:
    t   = dataset.times_rv[instrument]
    rv  = dataset.data_rv[instrument]
    err = dataset.errors_rv[instrument]

    # phase fold
    phase = ((t - t0) % P) / P
    phase[phase > 0.5] -= 1.0

    # model evaluated at data times (includes jitter added in quadrature)
    model = results.rv.evaluate(instrument)

    ax1.errorbar(phase, rv, yerr=err, fmt='o', label=instrument)
    ax2.errorbar(phase, rv - model, yerr=err, fmt='o')

# overplot the best-fit curve smoothly
t_model = np.linspace(t0, t0+P, 1000)
phase_model = ((t_model - t0) % P) / P
phase_model[phase_model > 0.5] -= 1.0
sort = np.argsort(phase_model)

full_model = results.rv.evaluate(instrument, t=t_model)  # check API below
ax1.plot(phase_model[sort], full_model[sort], color='k', zorder=10)

ax1.set_ylabel('RV [m/s]')
ax2.set_ylabel('O-C [m/s]')
ax2.set_xlabel('Phase')
ax2.axhline(0, color='k', ls='--', lw=1)
ax1.legend()
plt.tight_layout()
plt.savefig('rv_phase_folded.png', dpi=150)


import corner

params_to_plot = ['K_p1', 'mu_PFS', 'sigma_w_PFS']
samples = np.vstack([posteriors[p] for p in params_to_plot]).T

fig = corner.corner(samples, labels=params_to_plot,
                     quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.savefig('corner_rv.png', dpi=150)


