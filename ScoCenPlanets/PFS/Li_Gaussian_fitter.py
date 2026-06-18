from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt
'''
[GURMEHER] My understanding of everything that is going on and what we do:

rn71.XXXX files are all exposure of the same star, here TIC88297141.
At PFS, we took 6 back to back observations of the star. These exposures,
if we read the readme.txt, is *without* the iodine cell in the lightpath.

Same star
Same spectrograph setup
Just taken at slightly different times that night
No iodine absorption contaminating the stellar spectrum

number of dot is just the exposure order, so should not be too concerning

In the PFS directory, we might see some other files, rn71.59XX, these are the
exposures/spectra *with* iodine cell. These exposures are used for precise RV 
work. We do not want these for raw Lithium measurements. 
'''

templates = ['rn71.6521','rn71.6522','rn71.6523',
             'rn71.6524','rn71.6525','rn71.6526']


'''
Our aim here, is to add/stack all the 6 exposures on top of each other.
Individually, the spectra have extremely low SNR, and are noisy and look 
bad. Ideally, we want to maximize SNR --> this is done by stacking. 
When you add N identical exposures, signal adds linearly but noise adds in 
quadrature --> Say one exposure has signal S and noise N (so SNR = S/N).
After adding 6 identical exposures:

Signal becomes 6S —> every real photon from the star adds up every time
Noise becomes √6 · N —> random noise doesn't all point the same direction, 
so it partially cancels. 

Noise grows slower than signal, so we can drown it by stacking 
'''

'''
just confirming, using "more data" i.e., from a different time in the night does  **not** make sense right. That doesn't give us higher exposure time...
'''



# create a blank stack array which will contain the stacked flux measurements
## will behave as our accumulator 
stack = np.zeros((73, 3520)) 

# co-add all 6 template spectra
for f in templates:
    stack += readsav(f)['sp']
    ## Loops over each filename, reads its flux array (shape 73×3520), and adds 
    # it into stack. After 6 iterations, each pixel in stack holds the sum of that 
    # pixel's counts across all 6 exposures --> simple addition.

w = readsav('w_n71_22.dat')['w'] # the .dat file contains the wavelength solution
## without this file,the x-axis would just be pixel number 0–3519 (uncallibrated), which is 
# meaningless for our purposes .

li_order = 65 # just manually, we saw what order corresponded to the Lithium wavelength

## You go from a 2D (73, 3520) array down to a 1D (3520,) array --> one specific 
# echelle order covering ~6668–6779 Å, the one that contains Li.
wave = w[li_order]
flux = stack[li_order]

# mask edges and cosmic ray at ~6740 --> not necessary, but was recommended so added..
mask = (wave > 6685) & (wave < 6770) & ~((wave > 6737) & (wave < 6743))
wm = wave[mask]
fm = flux[mask]

# smooth with a small Gaussian kernel for visualization, 
# I don't think we need this --> optional if we'd like to add

'''

Visualization tool --> we use a sliding window that moves pixel by pixel 
across the data, computing a local weighted average of neighbors at each position. 
The Gaussian kernel is essentially the same idea as a weighted rolling window used 
in the wotan detrending that we did for our original pipeline.

smoothening kernel is 2 pixels wide. the higher the pixel number in the gaussian width,
the more hard the smoothening, think of it like the window length in wotan. 
'''
from scipy.ndimage import gaussian_filter1d
flux_smooth = gaussian_filter1d(fm, sigma=2)

plt.figure(figsize=(13, 4))
plt.plot(wm, fm, lw=0.6, color='dodgerblue', alpha=0.5, label='co-added raw')
plt.plot(wm, flux_smooth, lw=1.5, color='navy', label='smoothed')
plt.axvline(6707.8, color='red', ls='--', lw=1, label='Li I 6707.8 Å')
plt.axvline(6717.7, color='orange', ls='--', lw=1, label='Ca I 6717.7 Å')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Counts')
plt.legend()
plt.xticks(np.arange(6688, 6730, 2), rotation=90)
plt.yticks(np.arange(300, 800, 10), rotation=30)
plt.title('Order 65 — co-added 6 templates')
plt.tight_layout()
plt.show()

plt.savefig('templates_overplot.pdf', dpi=150, bbox_inches='tight')
plt.close()


# find the minimum of the smoothed flux in a narrow window around where we see the dip
window = (wm > 6707) & (wm < 6712)
li_center = wm[window][np.argmin(flux_smooth[window])]
print(f"Li dip center: {li_center:.3f} Å")
print(f"Shift from rest: {li_center - 6707.835:.3f} Å")

#### now we start the process of finding the lithium ew using Luke's cdips code as motivation
#### note: our code will not be as complicated as Luke's as his was a pipeline implementation 
### to account for various instruments.. edge case triggers, etc. We can be more loose


# from cdips
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from astropy import units as u

# trim to a narrow region around the Li line --> local continuum normalization rather than global
# this is because we are solely focused on discerning the equivalent width of our lithium line.
li_mask = (wm > 6700) & (wm < 6720)
wm_li = wm[li_mask]
fm_li = fm[li_mask]

# build spectrum1D on trimmed region --> same as cdips code
spec = Spectrum1D(spectral_axis=wm_li*u.AA, flux=fm_li*u.dimensionless_unscaled)


exclude_regions = [
    SpectralRegion(6707*u.AA, 6712*u.AA),  
    SpectralRegion(6716*u.AA, 6720*u.AA),  
]

# fit continuum
continuum = fit_generic_continuum(spec, exclude_regions=exclude_regions)(spec.spectral_axis)

# normalize
cont_norm_spec = spec / continuum

plt.figure(figsize=(10,4))
plt.plot(cont_norm_spec.spectral_axis, cont_norm_spec.flux, lw=0.8, color='navy')
plt.axhline(1.0, color='gray', ls='--', lw=0.8)
plt.axvline(li_center, color='red', ls='--', lw=1, label=f'Li center {li_center:.2f} Å')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Normalised flux')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("checker.png")


# next step: Gaussian fitting

### note the cdips code flips the spectrum (1 - flux) to turn the absorption dip into a peak, fits a Gaussian to that peak, then flips back. 
# We shall do the same:

from astropy.modeling import models, fitting

# flip so absorption becomes emission peak
full_spec = Spectrum1D(
    spectral_axis=cont_norm_spec.spectral_axis,
    flux=(1 - cont_norm_spec.flux)
)

# define the region to fit over (±1Å around our empirical li_center)
region = SpectralRegion((li_center - 1.0)*u.AA, (li_center + 1.0)*u.AA)

# initial guess for the Gaussian
g_init = models.Gaussian1D(
    amplitude=0.5*u.dimensionless_unscaled,
    mean=li_center*u.AA,
    stddev=0.5*u.AA
)

# fit
from specutils.fitting import fit_lines
g_fit = fit_lines(full_spec, g_init, window=(region.lower, region.upper))
print(g_fit)


from specutils.analysis import equivalent_width

# define the integration region (±1Å around our empirical li center)
region = SpectralRegion((li_center - 1.0)*u.AA, (li_center + 1.0)*u.AA)

# EW from the continuum normalised spectrum directly
li_ew = equivalent_width(cont_norm_spec, regions=region)
print(f"Li EW (direct): {li_ew:.3f}")
print(f"Li EW (direct): {(li_ew.to(u.AA)*1000):.1f} mA")


# evaluate the gaussian fit on a fine grid
x_fit = np.linspace(
    cont_norm_spec.spectral_axis.min(),
    cont_norm_spec.spectral_axis.max(),
    10000
)
y_fit = g_fit(x_fit)

# build a spectrum from the gaussian fit
fitted_spec = Spectrum1D(
    spectral_axis=x_fit,
    flux=(1 - y_fit)*u.dimensionless_unscaled
)

# EW from the gaussian fit
fitted_li_ew = equivalent_width(fitted_spec, regions=region)
print(f"Li EW (Gaussian fit): {fitted_li_ew.to(u.AA)*1000:.1f} mA")


x_fit_plot = np.linspace(li_center - 2.0, li_center + 2.0, 10000) * u.AA
y_fit_plot = g_fit(x_fit_plot)

plt.figure(figsize=(10, 5))

# normalised spectrum
plt.plot(cont_norm_spec.spectral_axis, cont_norm_spec.flux,
         color='navy', lw=0.8, label='continuum normalised')

# gaussian fit
plt.plot(x_fit_plot, 1 - y_fit_plot,
         color='red', lw=1.5, label=f'Gaussian fit')

# integration region
plt.axvspan(li_center - 1.0, li_center + 1.0,
            alpha=0.1, color='green', label='integration region (±1Å)')

# reference lines
plt.axhline(1.0, color='gray', ls='--', lw=0.8)
plt.axvline(li_center, color='red', ls=':', lw=0.8)

# annotate EW result
plt.text(0.97, 0.05,
         f'EW (direct) = 353.8 mÅ\nEW (Gaussian) = 349.7 mÅ',
         transform=plt.gca().transAxes,
         ha='right', va='bottom', fontsize=10,
         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

plt.xlabel('Wavelength (Å)')
plt.ylabel('Normalised flux')
plt.title('Li I 6708Å — TIC88297141')
plt.legend(loc='upper left')
plt.xlim(li_center - 2.5, li_center + 2.5)
plt.tight_layout()
plt.savefig('Li_EW_result.pdf', dpi=150, bbox_inches='tight')
plt.show()


N_mc = 20

# estimate noise from the continuum (away from the Li line)
continuum_mask = (wm_li < li_center - 1.5) | (wm_li > li_center + 1.5)
continuum_rms = np.nanstd(cont_norm_spec.flux.value[continuum_mask])
print(f"Continuum RMS (noise estimate): {continuum_rms:.4f}")

mc_ews = []

for i in range(N_mc):
    np.random.seed(i)
    
    # add random noise to the flux
    noise = np.random.normal(loc=0, scale=continuum_rms, size=len(fm_li))
    
    # rebuild spectrum with noise
    spec_mc = Spectrum1D(
        spectral_axis=wm_li*u.AA,
        flux=(fm_li + noise)*u.dimensionless_unscaled
    )
    
    # refit continuum
    continuum_mc = fit_generic_continuum(
        spec_mc, exclude_regions=exclude_regions
    )(spec_mc.spectral_axis)
    cont_norm_mc = spec_mc / continuum_mc
    
    # flip and refit gaussian
    full_spec_mc = Spectrum1D(
        spectral_axis=cont_norm_mc.spectral_axis,
        flux=(1 - cont_norm_mc.flux)
    )
    g_init_mc = models.Gaussian1D(
        amplitude=0.5*u.dimensionless_unscaled,
        mean=li_center*u.AA,
        stddev=0.5*u.AA
    )
    g_fit_mc = fit_lines(
        full_spec_mc, g_init_mc,
        window=(region.lower, region.upper)
    )
    
    # evaluate on fine grid and get EW
    y_fit_mc = g_fit_mc(x_fit)
    fitted_spec_mc = Spectrum1D(
        spectral_axis=x_fit,
        flux=(1 - y_fit_mc)*u.dimensionless_unscaled
    )
    ew_mc = equivalent_width(fitted_spec_mc, regions=region)
    mc_ews.append(ew_mc.to(u.AA).value * 1000)  # in mA
    print(f"  iteration {i+1}/{N_mc}: EW = {mc_ews[-1]:.1f} mA")

mc_ews = np.array(mc_ews)
mc_ews = mc_ews[np.isfinite(mc_ews)]  # drop any failed fits

p16, p50, p84 = np.percentile(mc_ews, [16, 50, 84])
perr = p84 - p50
merr = p50 - p16

print(f"\nMonte Carlo result:")
print(f"EW = {p50:.1f} +{perr:.1f} -{merr:.1f} mA")

