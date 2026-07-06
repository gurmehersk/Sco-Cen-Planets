import numpy as np
import radvel
from scipy import optimize
rvfname = "TIC88297141.vels"
t, rv, rv_err = np.loadtxt("TIC88297141.vels", usecols=(0,1,2), unpack=True)

t0_full_bjd = 3803.2401428842       
known_period  = 4.6445890270   

t, rv, err = np.loadtxt(rvfname, usecols=(0, 1, 2), unpack=True)

with open(rvfname, "w") as f:
    for ti, rvi, erri in zip(t, rv, err):
        f.write(f"{ti:.6f} {rvi:.3f} {erri:.3f} PFS\n")
print(len(t))
print(rv)


### everything pertaining to the radvel fit is taken from the K2-24 Fitting & MCMC tutorial on the following website:

### https://radvel.readthedocs.io/en/latest/tutorials/K2-24_Fitting%2BMCMC.html


params = radvel.Parameters(1, basis='per tc e w k')  # 1 planet

params['per1'] = radvel.Parameter(value=known_period, vary=False)   # fixed
params['tc1']  = radvel.Parameter(value=t0_full_bjd,  vary=False)   # fixed
params['e1']   = radvel.Parameter(value=0.0,          vary=False)   # circular
params['w1']   = radvel.Parameter(value=np.pi/2,      vary=False)   # NOTE: radians, not degrees!
params['k1']   = radvel.Parameter(value=100.0,        vary=True)    # initial guess, will be fit


params['dvdt'] = radvel.Parameter(value=0.0, vary=False)
params['curv'] = radvel.Parameter(value=0.0, vary=False)


time_base = np.median(t)
mod = radvel.RVModel(params, time_base=time_base)

# --- likelihood for PFS ---
like = radvel.likelihood.RVLikelihood(mod, t, rv, rv_err, suffix='_PFS') ### from radvel tut... [5] cell... 
like.params['gamma_PFS'] = radvel.Parameter(value=np.median(rv), vary=True)
like.params['jit_PFS']   = radvel.Parameter(value=5.0, vary=True)

# --- posterior + priors ---
post = radvel.posterior.Posterior(like)
post.priors += [radvel.prior.HardBounds('jit_PFS', 0.0, 1000.0)]

# --- MAP fit first ---
res = optimize.minimize(post.neglogprob_array, post.get_vary_params(), method='Powell')
print(post)

# --- then MCMC for uncertainties ---
chains = radvel.mcmc(post, nwalkers=50, nrun=10000)

import corner

# 'chains' is the pandas DataFrame returned by radvel.mcmc()
fit_params = post.name_vary_params()   # e.g. ['k1', 'gamma_PFS', 'jit_PFS']

label_map = {
    'k1':        r'$K$ (m/s)',
    'gamma_PFS': r'$\gamma_{\rm PFS}$ (m/s)',
    'jit_PFS':   r'$\sigma_{\rm jit,PFS}$ (m/s)',
    'per1':      r'$P$ (days)',
    'tc1':       r'$t_0$ (BJD)',
}
plot_labels = [label_map.get(p, p) for p in fit_params]

fig = corner.corner(
    chains[fit_params],
    labels=plot_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.3f'
)
fig.savefig('corner_rv_radvel.png', dpi=150)