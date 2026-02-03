import juliet 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
from matplotlib import gridspec

number_of_cores = 24 # Number of cores to use
run_number = 7

def get_data(data, hdr):
    ''' GETTING DATA FROM THE FITS FILES '''
    time = data['TIME']
    tessmag = hdr.get('TESSMAG', 'N/A')
    tempeff = hdr.get('TEFF', 'N/A')
    sap_flux = data['SAP_FLUX']
    pdcsap_flux = data['PDCSAP_FLUX']
    qual = data['QUALITY']
    bkgd = data['SAP_BKG'] # TODO : PLOT ME!
    return time, tessmag, tempeff, sap_flux, pdcsap_flux, qual, bkgd

tic_id = 88297141

lcpath = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025127075000-s0092-0000000088297141-0289-s/tess2025127075000-s0092-0000000088297141-0289-s_lc.fits"

hdu_list = fits.open(lcpath)
hdr = hdu_list[0].header
data = hdu_list[1].data
hdu_list.close()


time, tessmag, tempeff, sap_flux, pdcsap_flux, qual, bkgd = get_data(data, hdr)

mask = (qual == 0)

t = time[mask]
f = pdcsap_flux[mask]
ferr = data['PDCSAP_FLUX_ERR'][mask]


# Normalizing the fluxes!! 
median_flux = np.nanmedian(f)
f = f / median_flux
ferr = ferr / median_flux


# the naming convention adopted above is to remain consistent with the juliet docs 

# Now, to detrend using gaussian processes in Juliet, we need to know the T0 and period 
# of the transit! Luckily we have a good idea of this.
p, t0 = 4.64423, 3803.24126

# using juliet documentation
# Get phases --- identify out-of-transit (oot) times by phasing the data
# and selecting all points at absolute phases larger than 0.02:
phases = juliet.utils.get_phases(t, p, t0)
idx_oot = np.where(np.abs(phases)>0.02)[0]

# Save the out-of-transit data into dictionaries so we can feed them to juliet:
times, fluxes, fluxes_error = {},{},{}
#times['TESS'], fluxes['TESS'], fluxes_error['TESS'] = t[idx_oot],f[idx_oot],ferr[idx_oot]
#### For multisector stuff 
'''
r = lk.search_lightcurve(
    f"TIC {tic_id}",
    mission="TESS",
    author="SPOC",
    cadence="long"
)

lc = (
    r.download_all()
     .stitch()
     .remove_nans()
     .remove_outliers(sigma=5)
     .normalize()
)
'''

# https://juliet.readthedocs.io/en/latest/tutorials/gps.html#detrending-lightcurves-with-gps

# First define the priors:
priors = {}

# Same priors as for the transit-only fit, but we now add the GP priors:

## P_p1 : period of planet
## t0_p1 : time of transit center
## r1_p1, r2_p1 : parametrization of planet-to-star radius ratio, important for impact parameter
# q1_TESS, q2_TESS : quadratic limb-darkening coefficients
## ecc_p1, omega_p1 : eccentricity and argument of periastron
## rho : stellar density in cgs units
## mdilution_TESS : dilution factor for TESS --> constant at 1 for our case because
# we assume no contamination from other stars
## mflux_TESS : flux offset for TESS data
## sigma_w_TESS : white noise term for TESS data
## GP_sigma_TESS : amplitude of the GP for TESS data
## GP_rho_TESS : length scale of the GP for TESS data


### For the exact hyperparameters, i am going to try and emulate the prior values i had inputted in the previous transit fit
### when i wasnt doing simultaneous GP fitting. Back then the t0 sigma was 0.01, while the period prior was 0.1 
params = ['P_p1','t0_p1','r1_p1','r2_p1','q1_TESS','q2_TESS','ecc_p1','omega_p1',\
          'rho', 'mdilution_TESS', 'mflux_TESS', 'sigma_w_TESS', \
          'GP_sigma_TESS', 'GP_rho_TESS']

dists = ['normal','normal','uniform','uniform','uniform','uniform','fixed','fixed',\
         'normal', 'fixed', 'normal', 'loguniform', \
         'loguniform', 'loguniform']

#Juliet hyperparams
hyperps = [
    [p, 0.01],           
    [t0, 0.01],          
    [0., 1.],            
    [0., 1.],            
    [0., 1.],            
    [0., 1.],            
    0.0,                 
    90.,                 
    [0.1, 10],          
    1.0,                 
    [0., 0.1],           
    [1e-6, 1e-3],      
    [1e-6, 1e-1],        
    [0.1, 10.]           
]

### I think this is right (?) i made changes from the template. For instance, i changed the rho from loguniform to 
# normal since i had done that in the original transit fit (?) maybe log uniform is better (?) should i just follow the 
## template?

# Populate the priors dictionary:
for param, dist, hyperp in zip(params, dists, hyperps):
    priors[param] = {}
    priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

times['TESS'], fluxes['TESS'], fluxes_error['TESS'] = t,f,ferr
dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, \
                      yerr_lc = fluxes_error, GP_regressors_lc = times, out_folder = f'88297141_transitGP_v{run_number}', verbose = True)




print(f"Fitting the dataset with Juliet, using {number_of_cores} cores...")


# After results = dataset.fit()

print("\n" + "="*60)
print("BEST-FIT PARAMETERS (Median ± 1σ)")
print("="*60)

results = dataset.fit(use_dynesty=True, dynesty_nthreads = number_of_cores)

posterior_samples = results.posteriors['posterior_samples']

# Parameters to report
params_to_report = {
    'P_p1': 'Period (days)',
    't0_p1': 'T0 (BJD-2457000)',
    'r1_p1': ' r1 ',        
    'r2_p1': ' r2 ',        
    'rho': 'Stellar density (g/cm³)',
    'GP_sigma_TESS': 'GP amplitude',
    'GP_rho_TESS': 'GP timescale (days)',
    'sigma_w_TESS': 'Jitter',
}

for param, label in params_to_report.items():
    median = np.median(posterior_samples[param])
    lower = np.percentile(posterior_samples[param], 16)
    upper = np.percentile(posterior_samples[param], 84)
    err_low = median - lower
    err_high = upper - median
    print(f"{label:30s}: {median:.6f} +{err_high:.6f} -{err_low:.6f}")

print("="*60 + "\n")

# Also get the derived physical parameters
# Juliet computes these automatically --> [2nd Feb 2026: apparently not??]

### Alright so apparently JULIET DOES NOT COMPUTE THESE AUTOMATICALLY, SO BACK TO THE DRAWING BOARD OF MANUALLY CALCULATING THEM

if 'p_p1' not in posterior_samples:
    # We use the Espinoza (2018) parametrization for p and b
    # p = r1 (when r2 < 0.5) or p = 1 - r1 (when r2 >= 0.5) 
    # b = r2 * (1 + p) (when r2 < 0.5) or b = (1 - r2) * (1 + p) (when r2 >= 0.5)
    r1 = posterior_samples['r1_p1']
    r2 = posterior_samples['r2_p1']
    
    
    # Calculate p (planet-to-star radius ratio)
    p_p1 = np.where(r2 < 0.5, r1, 1 - r1)
    
    # Calculate b (impact parameter)
    b_p1 = np.where(r2 < 0.5, r2 * (1 + p_p1), (1 - r2) * (1 + p_p1))
    
    # Append this to the posterior samples dictionary 
    posterior_samples['p_p1'] = p_p1
    posterior_samples['b_p1'] = b_p1

if 'p_p1' in posterior_samples:  # planet-to-star radius ratio
    p_median = np.median(posterior_samples['p_p1'])
    p_lower = np.percentile(posterior_samples['p_p1'], 16)
    p_upper = np.percentile(posterior_samples['p_p1'], 84)
    print(f"Rp/Rs (derived):               {p_median:.6f} +{p_upper-p_median:.6f} -{p_median-p_lower:.6f}")

if 'b_p1' in posterior_samples:  # impact parameter
    b_median = np.median(posterior_samples['b_p1'])
    b_lower = np.percentile(posterior_samples['b_p1'], 16)
    b_upper = np.percentile(posterior_samples['b_p1'], 84)
    print(f"Impact parameter b:            {b_median:.6f} +{b_upper-b_median:.6f} -{b_median-b_lower:.6f}")

print("="*60)

# Extract full model:
transit_plus_GP_model = results.lc.evaluate('TESS')

# Deterministic part of the model (transit divided by mflux):
transit_model = results.lc.model['TESS']['deterministic']

# GP part of the model:
gp_model = results.lc.model['TESS']['GP']

# Recalculate phases for all data points
phases = juliet.utils.get_phases(dataset.times_lc['TESS'], 
                                  results.posteriors['posterior_samples']['P_p1'].mean(),
                                  results.posteriors['posterior_samples']['t0_p1'].mean())

# Now plot
fig = plt.figure(figsize=(14,5))
gs = gridspec.GridSpec(1, 2, width_ratios=[2,1])
ax1 = plt.subplot(gs[0])

# Plot data - USE FULL TIME RANGE FIRST
ax1.errorbar(dataset.times_lc['TESS'], dataset.data_lc['TESS'], \
             yerr = dataset.errors_lc['TESS'], fmt = '.', alpha = 0.1, label='Data')

# Plot the (full, transit + GP) model:
ax1.plot(dataset.times_lc['TESS'], transit_plus_GP_model, color='black', 
         linewidth=2, zorder=10, label='Transit + GP model')

# Don't set xlim initially to see all data
# ax1.set_xlim([1328,1350])
ax1.set_xlabel('Time (BJD - 2457000)', fontsize=12)
ax1.set_ylabel('Relative flux', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = plt.subplot(gs[1])

# Now plot phase-folded lightcurve with GP removed:
ax2.errorbar(phases, dataset.data_lc['TESS'] - gp_model, \
             yerr = dataset.errors_lc['TESS'], fmt = '.', alpha = 0.3, 
             label='GP-corrected data')

# Plot transit-only model:
idx = np.argsort(phases)
ax2.plot(phases[idx], transit_model[idx], color='black', linewidth=2,
         zorder=10, label='Transit model')

ax2.set_xlabel('Phases', fontsize=12)
ax2.set_ylabel('Relative flux', fontsize=12)
ax2.set_xlim([-0.03, 0.03])
# ax2.set_ylim([0.96, 1.04])  # Remove if you want auto-scale
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('88297141_transit_GP_fit.png', dpi=300, bbox_inches='tight')
print(f"Saved transit + GP plot to 88297141_transit_GP_fit_v{run_number}.png")


import corner

posterior_samples = results.posteriors['posterior_samples']
params_to_plot = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'rho', 'GP_sigma_TESS', 'GP_rho_TESS']
labels = ['Period (d)', 'T0 (BJD-2457000)', 'Rp/Rs', 'b', 'ρ (g/cm³)', 
          'GP σ', 'GP ρ (d)']

samples = np.array([posterior_samples[param] for param in params_to_plot]).T

fig_corner = corner.corner(samples, labels=labels, 
                           label_kwargs={'fontsize': 12},
                           title_kwargs={'fontsize': 10})
fig_corner.savefig('88297141_corner_plot_v4.png', dpi=300, bbox_inches='tight')
print(f"Saved corner plot to 88297141_corner_plot_v{run_number}.png")

plt.close('all')