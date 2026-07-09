'''

radvel_rv.py 

We are going to try and use Luke's timmy radvel code 
as a template to find the radial velocity contrast 
curve.

If you are interested in how we can find the mass 
upper limits for the known period and t0, check
the juliet version named "juliet_fitting.py"

'''

### Note: we might have to make the semi amplitude K to be positive in this 
# and not the symmetric case.

starname = "TIC88297141"
nplanets = 1
instnames = ["PFS"]
ntels = len(instnames)                         # number of instruments with unique velocity zero-points
fitting_basis = 'logper tc secosw sesinw logk'   
bjd0 = 0                                       # reference epoch for RV timestamps (i.e. this number has been subtracted off your timestamps)
planet_letters = {1: 'b'}                      # map the numbers in the Parameters keys to planet letters (for plotting and tables)

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


# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(nplanets,basis='per tc e w k', planet_letters=planet_letters)    # initialize Parameters object

### everything pertaining to the radvel fit is taken from the K2-24 Fitting & MCMC tutorial on the following website:

### https://radvel.readthedocs.io/en/latest/tutorials/K2-24_Fitting%2BMCMC.html



'''
A minute to set things straight and make something that seems unnervy clear:

The value=... you set on the Parameter object is the starting point for the search — nothing more, its not the prior and it doesnt influence your find in any way. 
It's used to seed the local optimizer (radvel's MLE step is typically Nelder-Mead/Powell-style — a local optimizer, not a global one) — it walks downhill from wherever you put it, 
so if you start it in a nonsense part of parameter space, it can get stuck at some spurious local optimum instead of the real signal.

It is essentially used to initialize the MCMC walkers: they're seeded as a small perturbed ball around the starting/MLE point, not drawn uniformly from across the whole prior range. 
If you seed the walkers near a random period with no relation to reality, they can spend the entire chain stuck in an aliasing peak and never find the right region at all, 
regardless of how wide the prior technically permits them to roam.

So Luke sets per1/tc1 to the actual known transit values because RV likelihood surfaces are brutally multimodal (aliases everywhere), and giving the search its best real-world 
starting guess makes it converge efficiently and reliably — while the wide prior is what keeps the fit honest: it doesn't force the answer to be the transit period, 
it just gives the walkers a sensible place to start exploring from. If the RVs actually prefer a totally different period/phase, the wide prior lets the chain wander
there and you'd see the posterior pull away from your starting guess — which is exactly the false-positive signature you're looking for.

'''

anybasis_params['per1'] = radvel.Parameter(value=known_period)   
anybasis_params['tc1']  = radvel.Parameter(value=t0_full_bjd+2457000)   
anybasis_params['e1']   = radvel.Parameter(value=0.0)   # circular
anybasis_params['w1']   = radvel.Parameter(value=np.pi/2)   # NOTE: radians, not degrees!
anybasis_params['k1']   = radvel.Parameter(value=200.0)    # initial guess, will be fit

time_base = 2461220.5 # somewhere near the midpoint of the time baseline... GK calculated this using the epochs we got from PFS
anybasis_params['dvdt'] = radvel.Parameter(value=0.0)
anybasis_params['curv'] = radvel.Parameter(value=0.0)

anybasis_params['gamma_PFS'] = radvel.Parameter(value = -227)
anybasis_params['jit_PFS']   = radvel.Parameter(value=282) 

params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)

# Set the 'vary' attributes of each of the parameters in the fitting basis. A parameter's 'vary' attribute should
# be set to False if you wish to hold it fixed during the fitting process. By default, all 'vary' parameters
# are set to True.

params['secosw1'].vary = False #True
params['sesinw1'].vary = False #True
params['logper1'].vary = True
params['tc1'].vary = True

params['curv'].vary = False
params['dvdt'].vary = False

# -----------------------------------------------------------------------------
# Priors
# Bounds kept wide on logk1/logper1 so a real signal isn't artificially
# excluded by the prior -- you want the fit to be free to find a false-
# positive signal wherever it lives, not boxed in near the transit period.
# -----------------------------------------------------------------------------
priors = [
    radvel.prior.EccentricityPrior(nplanets),
    radvel.prior.HardBounds('logk1', np.log(0.01), np.log(1e5)),
    radvel.prior.HardBounds('logper1', np.log(0.1), np.log(1e14)),
    radvel.prior.HardBounds('jit_PFS', 0.0, 400),   # matches Luke's TOI-837 pattern
]
 

stellar = dict(mstar=0.3005, mstar_err=0.09)       # placeholder -- replace with your actual value
planet = dict(rp1=0.095, rperr1=0.003)            # Rp/R* from the juliet joint fit


'''
#time_base = np.median(t) # let's comment this out for now... though i feel like it wouldve given the same answer as done above lol, let's try to be as consistent as possible...
mod = radvel.RVModel(params, time_base=time_base)

# --- likelihood for PFS ---
like = radvel.likelihood.RVLikelihood(mod, t, rv, rv_err, suffix='_PFS') ### from radvel tut... [5] cell... 


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
fig.savefig('corner_rv_radvel.png', dpi=150)'''

