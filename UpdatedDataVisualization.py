import os
import numpy as np
from glob import glob
from astropy.io import fits
import matplotlib.pyplot as plt

lcpaths = glob("*/*.fits")

for lcpath in lcpaths:

    print(lcpath)

    hdu_list = fits.open(lcpath)
    hdr = hdu_list[0].header
    data = hdu_list[1].data
    hdu_list.close()

    time = data['TIME']
    sap_flux = data['SAP_FLUX']
    pdcsap_flux = data['PDCSAP_FLUX']
    qual = data['QUALITY']
    bkgd = data['SAP_BKG'] # TODO : PLOT ME!

    sel = (qual == 0)
    time = time[sel]
    sap_flux = sap_flux[sel]
    pdcsap_flux = pdcsap_flux[sel]

    fig, axs = plt.subplots(nrows=2, figsize=(12,3))
    axs[0].scatter(time, sap_flux/np.nanmedian(sap_flux), c='k', s=0.5)
    axs[1].scatter(time, pdcsap_flux/np.nanmedian(pdcsap_flux), c='k', s=0.5)

    for ax in axs:
        ax.set_xlim([3665, 3690])
        ax.set_ylim([0.95, 1.05])

    savpath = os.path.basename(lcpath).replace(".fits",".png")
    fig.savefig(savpath, bbox_inches='tight')

    assert 0