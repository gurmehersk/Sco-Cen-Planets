from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt

templates = ['rn71.6521','rn71.6522','rn71.6523',
             'rn71.6524','rn71.6525','rn71.6526']

# co-add all 6 template spectra
stack = np.zeros((73, 3520))
for f in templates:
    stack += readsav(f)['sp']

w = readsav('w_n71_22.dat')['w']

li_order = 65
wave = w[li_order]
flux = stack[li_order]

# mask edges and cosmic ray at ~6740
mask = (wave > 6685) & (wave < 6770) & ~((wave > 6737) & (wave < 6743))
wm = wave[mask]
fm = flux[mask]

# smooth with a small Gaussian kernel for visualization
from scipy.ndimage import gaussian_filter1d
flux_smooth = gaussian_filter1d(fm, sigma=2)

plt.figure(figsize=(13, 4))
plt.plot(wm, fm, lw=0.6, color='steelblue', alpha=0.5, label='co-added raw')
plt.plot(wm, flux_smooth, lw=1.5, color='navy', label='smoothed')
plt.axvline(6707.8, color='red', ls='--', lw=1, label='Li I 6707.8 Å')
plt.axvline(6717.7, color='orange', ls='--', lw=1, label='Ca I 6717.7 Å')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Counts')
plt.legend()
plt.title('Order 65 — co-added 6 templates')
plt.tight_layout()
plt.show()
