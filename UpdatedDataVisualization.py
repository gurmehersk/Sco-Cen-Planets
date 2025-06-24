import os
import numpy as np
from glob import glob
from astropy.io import fits
import matplotlib.pyplot as plt

# creating a function that can bin the lightcurve for us
def bin_lightcurve(time, flux, bin_minutes=30):
    bin_size = bin_minutes / (24 * 60)  # convert minutes to days
    bins = np.arange(time.min(), time.max() + bin_size, bin_size)
    digitized = np.digitize(time, bins)

    binned_time = np.array([np.nanmean(time[digitized == i]) for i in range(1, len(bins))])
    binned_flux = np.array([np.nanmean(flux[digitized == i]) for i in range(1, len(bins))])

    return binned_time, binned_flux

sector_number = 87
lcpaths = glob(f"/ar1/TESS/SPOC/s00{sector_number}/*.fits")

for lcpath in lcpaths:

    print(lcpath)

    hdu_list = fits.open(lcpath)
    hdr = hdu_list[0].header
    data = hdu_list[1].data
    hdu_list.close()
    tic_id = hdr.get('TICID', 'unknown')
    time = data['TIME']
    tessmag = hdr.get('TESSMAG', 'N/A')
    tempeff = hdr.get('TEFF', 'N/A')
    sap_flux = data['SAP_FLUX']
    pdcsap_flux = data['PDCSAP_FLUX']
    qual = data['QUALITY']
    bkgd = data['SAP_BKG'] # TODO : PLOT ME!
    savpath = f"/home/gurmeher/gurmeher/lightcurves/sector_{sector_number}/TIC_{tic_id}.pdf"
    # savpath2 = f"/home/gurmeher/gurmeher/lightcurves/sector_{sector_number}/bkgd/TIC_{tic_id}.pdf"

    if os.path.exists(savpath):
            print(f"Skipping TIC {tic_id} — already cached.")
            continue

    sel = (qual == 0)
    time = time[sel]
    sap_flux = sap_flux[sel]
    pdcsap_flux = pdcsap_flux[sel]
    bkgd = bkgd[sel]


    q1 = np.nanpercentile(sap_flux, 25)
    q3 = np.nanpercentile(sap_flux, 75)
    iqr = q3 - q1

    # Define upper limit to clip flares
    upper_bound = q3 + 1.5 * iqr
    mask = sap_flux < upper_bound

    sap_time_binned, sap_flux_binned = bin_lightcurve(time[mask], sap_flux[mask]/np.nanmedian(sap_flux[mask]))
    pdc_time_binned, pdc_flux_binned = bin_lightcurve(time[mask], pdcsap_flux[mask]/np.nanmedian(pdcsap_flux[mask]))
    bkg_time_binned, bkg_flux_binned = bin_lightcurve(time[mask], bkgd[mask]/np.nanmedian(bkgd[mask]))

    fig, axs = plt.subplots(nrows=3, figsize=(10,12), sharex = True)
    plt.subplots_adjust(hspace=0.3)
    axs[0].scatter(sap_time_binned, sap_flux_binned, c='k', s=0.8, label = 'SAP')
    axs[0].set_ylabel("SAP", fontsize = 8 )
    axs[0].set_title(f"TIC {tic_id} — TESS mag = {tessmag} at Temp = {tempeff} Binned to 30 minutes", fontsize=8)

    axs[1].scatter(pdc_time_binned, pdc_flux_binned, c='k', s=0.8, label = 'PDCSAP')
    axs[1].set_ylabel("PDCSAP", fontsize = 8)
    axs[2].scatter(bkg_time_binned, bkg_flux_binned, c= 'k', s = 0.8, label = 'Background')
    axs[2].set_ylabel("Bkg", fontsize = 8)
    axs[2].set_xlabel('Time (BTJD)', fontsize = 8)
# added new folder inside each sector which has "fluxes" and "bkgd" so I can separately save them
# keeping the sap cutoff for the pdc_sap cutoff since they should be about the same when considering flares, wouldn't change much i believe
    for ax in axs:
        
       # initially had put the q1, q3 and everything here, caused such problems
        ax.set_xlim(time.min()-2, time.max())
        #ax.set_ylim([0, upper_bound])


    fig.savefig(savpath, bbox_inches='tight', format = "pdf")
    plt.close(fig)
   # assert 0 # remove this when not testing and actually running code




