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
plt.title('Order 65 — co-added 6 templates')
plt.tight_layout()
plt.show()

plt.savefig('templates_overplot.pdf', dpi=150, bbox_inches='tight')
plt.close()