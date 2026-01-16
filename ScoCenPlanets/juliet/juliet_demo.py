import juliet 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
from matplotlib import gridspec

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

hyperps = [[p, 0.1], [t0, 0.01], [0, 1.], [0., 1.], [0., 1.], [0., 1.], 0.0, 90.,\
           [1.0, 0.5], 1.0, [0., 0.1], [0.1, 1000.], \
           [1e-6, 1e6], [1e-3, 1e3]]

### I think this is right (?) i made changes from the template. For instance, i changed the rho from loguniform to 
# normal since i had done that in the original transit fit (?) maybe log uniform is better (?) should i just follow the 
## template?

# Populate the priors dictionary:
for param, dist, hyperp in zip(params, dists, hyperps):
    priors[param] = {}
    priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

times['TESS'], fluxes['TESS'], fluxes_error['TESS'] = t,f,ferr
dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, \
                      yerr_lc = fluxes_error, GP_regressors_lc = times, out_folder = '88297141_transitGP', verbose = True)

results = dataset.fit()

# Extract full model:
transit_plus_GP_model = results.lc.evaluate('TESS')

# Deterministic part of the model (in our case transit divided by mflux):
transit_model = results.lc.model['TESS']['deterministic']

# GP part of the model:
gp_model = results.lc.model['TESS']['GP']

# Now plot. First preambles:
fig = plt.figure(figsize=(12,4))
gs = gridspec.GridSpec(1, 2, width_ratios=[2,1])
ax1 = plt.subplot(gs[0])

# Plot data
ax1.errorbar(dataset.times_lc['TESS'], dataset.data_lc['TESS'], \
             yerr = dataset.errors_lc['TESS'], fmt = '.', alpha = 0.1)

# Plot the (full, transit + GP) model:
ax1.plot(dataset.times_lc['TESS'], transit_plus_GP_model, color='black',zorder=10)

ax1.set_xlim([t0-10,t0+10])
ax1.set_ylim([0.96,1.04])
ax1.set_xlabel('Time (BJD - 2457000)')
ax1.set_ylabel('Relative flux')

ax2 = plt.subplot(gs[1])

# Now plot phase-folded lightcurve but with the GP part removed:
ax2.errorbar(phases, dataset.data_lc['TESS'] - gp_model, \
             yerr = dataset.errors_lc['TESS'], fmt = '.', alpha = 0.3)

# Plot transit-only (divided by mflux) model:
idx = np.argsort(phases)
ax2.plot(phases[idx],transit_model[idx], color='black',zorder=10)
ax2.yaxis.set_major_formatter(plt.NullFormatter())
ax2.set_xlabel('Phases')
ax2.set_xlim([-0.03,0.03])
ax2.set_ylim([0.96,1.04])


# SAVE THE PLOT
plt.tight_layout()
plt.savefig('88297141_transit_GP_fit.png', dpi=300, bbox_inches='tight')
print("Saved transit + GP plot to 88297141_transit_GP_fit.png")

# Also save the corner plot if you want to see parameter correlations
# For corner plot, use juliet.utils or results.posteriors
# Option 1: Use juliet's built-in corner plot
import corner

posterior_samples = results.posteriors['posterior_samples']
params_to_plot = ['P_p1', 't0_p1', 'r2_p1', 'rho', 'GP_sigma_TESS', 'GP_rho_TESS']

# Get the samples for these parameters
samples = np.array([posterior_samples[param] for param in params_to_plot]).T

# Create corner plot
fig_corner = corner.corner(samples, labels=params_to_plot)
fig_corner.savefig('88297141_corner_plot.png', dpi=300, bbox_inches='tight')
print("Saved corner plot to 88297141_corner_plot.png")

plt.close('all')