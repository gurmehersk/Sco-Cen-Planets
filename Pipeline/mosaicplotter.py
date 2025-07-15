import os
import numpy as np
from glob import glob
from astropy.io import fits
from lightkurve import search_lightcurve
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from wotan import flatten
from transitleastsquares import transitleastsquares
from matplotlib.backends.backend_pdf import PdfPages

# using subplots
mosaic = (
    """
    AAAAAAAA
    AAAAAAAA
    AAAAAAAA
    BBBBBBBB
    BBBBBBBB
    BBBBBBBB
    CCCCCCCC
    CCCCCCCC
    CCCCCCCC
    DDDDDDDD
    DDDDDDDD
    DDDDDDDD
    EEEEEEEE
    EEEEEEEE
    EEEEEEEE
    FFFFGGGG
    """)
# Let's only plot PDCSAP!!
subplotter, plaxes = plt.subplot_mosaic(mosaic, figsize=(16, 12))


'''BINNING TO 30 MINUTES'''

def bin_lightcurve(time, flux, bin_minutes=30):
    bin_size = bin_minutes / (24 * 60)  # minutes to days
    bins = np.arange(time.min(), time.max() + bin_size, bin_size)
    digitized = np.digitize(time, bins)

    binned_time = []
    binned_flux = []

    for i in range(1, len(bins)):
        bin_time = time[digitized == i]
        bin_flux = flux[digitized == i]
        if len(bin_time) > 0:
            binned_time.append(np.nanmean(bin_time))
            binned_flux.append(np.nanmean(bin_flux))

    return np.array(binned_time), np.array(binned_flux)


''' RETREIVING THE DATA FOR A PARTICULAR TICID AND GETTING ALL ITS INFORMATION '''

#sector_number = 91
tic_id = 81777149
#lcpath = (f"/ar1/TESS/SPOC/s0091/tess2025099153000-s0091-0000000009676822-0288-s_lc.fits")
lcpath = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025071122000-s0090-0000000081777149-0287-s/tess2025071122000-s0090-0000000081777149-0287-s_lc.fits"
pdf_path = (f"/home/gurmeher/gurmeher/detrending/TIC_COMBINED_{tic_id}mosaicplotter.pdf")
hdu_list = fits.open(lcpath)
hdr = hdu_list[0].header
data = hdu_list[1].data
time = data['TIME']
tessmag = hdr.get('TESSMAG', 'N/A')
tempeff = hdr.get('TEFF', 'N/A')
sap_flux = data['SAP_FLUX']
pdcsap_flux = data['PDCSAP_FLUX']
qual = data['QUALITY']
bkgd = data['SAP_BKG'] # TODO : PLOT ME!
#savpath = f"/home/gurmeher/gurmeher/detrending/TIC_{tic_id}.pdf"

''' PLOTTING RAW DATA FIRST '''

#fig3, axs3 = plt.subplots(nrows =3 , figsize = (6,10), sharex = True)
#plt.subplots_adjust(hspace = 0.3)
#plaxes['A'].scatter(time, sap_flux, c='k', s= 0.8, label = 'SAP')
plaxes['A'].scatter(time, pdcsap_flux, c = 'k', s= 0.8, label = 'PDCSAP')
plaxes['A'].set_title(f" RAW DATA FOR TIC {tic_id}")

''' CLEANING IT '''

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

fig, axs = plt.subplots(nrows=6, figsize=(12,18))
plt.subplots_adjust(hspace=0.3)
#plaxes['B'].scatter(sap_time_binned, sap_flux_binned, zorder = 2, c='k', s=0.8, label = 'SAP')
#plaxes['B'].set_ylabel("SAP", fontsize = 8 )


plaxes['C'].scatter(pdc_time_binned, pdc_flux_binned, zorder =2,  c='k', s=0.8, label = 'PDCSAP')
plaxes['C'].set_ylabel("PDCSAP", fontsize = 8)
plaxes['C'].set_title(f"TIC {tic_id} — TESS mag = {tessmag} at Temp = {tempeff} Binned to 30 minutes", fontsize=8)
'''LOMBSCARGLE PERIODOGRAM '''

frequency_SAP, power_SAP = LombScargle(sap_time_binned, sap_flux_binned).autopower()
frequency_PDCSAP, power_PDCSAP = LombScargle(pdc_time_binned, pdc_flux_binned).autopower()
mask = frequency_PDCSAP < 20
frequency_PDCSAP = frequency_PDCSAP[mask]
power_PDCSAP = power_PDCSAP[mask]
frequency_SAP = frequency_SAP[mask]
power_SAP = power_SAP[mask]
best_frequency_SAP = frequency_SAP[np.argmax(power_SAP)]
best_period_SAP = 1 / best_frequency_SAP 
best_frequency_PDCSAP = frequency_PDCSAP[np.argmax(power_PDCSAP)]
best_period_PDCSAP = 1 / best_frequency_PDCSAP

print(f'Best period : {best_period_PDCSAP}')
print(f'Best frequency: {best_frequency_PDCSAP}')

'''plaxes['E'].plot(frequency_SAP, power_SAP, label = 'SAP LS')
plaxes['E'].set_ylabel("Power", fontsize = 8)
plaxes['E'].set_xlabel('Frequency', fontsize = 8)'''

plaxes['F'].plot(frequency_PDCSAP, power_PDCSAP, label = 'PDC LS')
plaxes['F'].set_ylabel("Power", fontsize = 8)
plaxes['F'].set_xlabel('Frequency', fontsize = 8)

#axs[2].set_xscale('log')  # For SAP
plaxes['F'].set_xscale('log')  # For PDCSAP

# keeping the sap cutoff for the pdc_sap cutoff since they should be about the same when considering flares, wouldn't change much i believe
'''for i in range(len(axs)):
    if i < 2 or i > 3:
    # initially had put the q1, q3 and everything here, caused such problems
        axs[i].set_xlim(time.min()-2, time.max())
    else:
        axs[i].set_xlim(0,20)
        # I set the axis limit at 20 because it was catching another rotation period at about 0.02 days, which was a little crazy for the 
        # that 0.02 thing I saw was the binning cadence signal which is okay to mask
        # lightcurve
        continue'''


    #ax.set_ylim([0, upper_bound])

'''PHASE FOLDING TO ONE TIME PERIOD '''

sap_phase = (sap_time_binned % best_period_SAP)/best_period_SAP
pdcsap_phase = (pdc_time_binned % best_period_PDCSAP)/best_period_PDCSAP
fig_phase, axs_phase = plt.subplots(2, figsize=(10, 8), sharex=True)
plt.subplots_adjust(hspace=0.3)
axs_phase[0].scatter(sap_phase, sap_flux_binned, s=0.5, c='black', label='SAP Phase Folded')
axs_phase[0].set_ylabel("Flux")
axs_phase[0].set_title(f"TIC {tic_id} — SAP Phase Folded at {best_period_SAP:.4f} d") # four decimal places rounded 
axs_phase[0].legend()

plaxes['B'].scatter(pdcsap_phase, pdc_flux_binned, s=0.5, c='black', label='PDCSAP Phase Folded')
plaxes['B'].set_xlabel("Phase")
plaxes['B'].set_ylabel("Flux")
plaxes['B'].set_title(f"TIC {tic_id} — PDCSAP Phase Folded at {best_period_PDCSAP:.4f} d") # 4 decimal places round
plaxes['B'].legend()


''' NOW WE DO THE WOTAN FLATTENING OF THE LIGHT CURVE '''

wdwl = 0.3 * best_period_PDCSAP

flatten_lc1, trend_lc1 = flatten(sap_time_binned, sap_flux_binned, window_length = wdwl, return_trend = True, method = 'biweight')
flatten_lc2, trend_lc2 = flatten(pdc_time_binned, pdc_flux_binned, window_length = wdwl, return_trend = True, method = 'biweight')
#plaxes['B'].scatter(sap_time_binned, trend_lc1, s = 1.5, zorder = 1, color = 'red')
plaxes['C'].scatter(pdc_time_binned, trend_lc2, s = 1.5, zorder = 1, color = 'red')

#axs[4].scatter(sap_time_binned, flatten_lc1, s=1, color='black', label = 'Flattened SAP')
plaxes['D'].scatter(pdc_time_binned, flatten_lc2, s = 1, color = 'black', label = 'Flattened PDCSAP')

for ax in axs:
    ax.legend()

       
#fig.savefig(savpath, bbox_inches='tight', format = "pdf")
#plt.close(fig)

''' NOW WE DO THE TLS PHASE FOLDING AND PLOTTING ON A NEW GRAPH '''

def clean_arrays(time, flux):
    mask = (~np.isnan(time)) & (~np.isnan(flux))
    return time[mask], flux[mask]

sap_time_clean, flatten_lc1_clean = clean_arrays(sap_time_binned, flatten_lc1)
pdc_time_clean, flatten_lc2_clean = clean_arrays(pdc_time_binned, flatten_lc2)

print(f"TIC {tic_id}: SAP clean time length = {len(sap_time_clean)}, flatten length = {len(flatten_lc1_clean)}")
print(f"TIC {tic_id}: PDCSAP clean time length = {len(pdc_time_clean)}, flatten length = {len(flatten_lc2_clean)}")

print(f"TIC {tic_id}: SAP clean time NaNs = {np.isnan(sap_time_clean).sum()}, flatten NaNs = {np.isnan(flatten_lc1_clean).sum()}")
print(f"TIC {tic_id}: PDCSAP clean time NaNs = {np.isnan(pdc_time_clean).sum()}, flatten NaNs = {np.isnan(flatten_lc2_clean).sum()}")

if len(sap_time_clean) == 0 or len(flatten_lc1_clean) == 0:
    print(f"TIC {tic_id}: Empty SAP arrays for TLS, skipping.")
   

if len(pdc_time_clean) == 0 or len(flatten_lc2_clean) == 0:
    print(f"TIC {tic_id}: Empty PDCSAP arrays for TLS, skipping.")


if len(sap_time_clean) < 50 or (sap_time_clean.max() - sap_time_clean.min()) < 5:
    print(f"TIC {tic_id}: Not enough data points or too short time span for TLS. Skipping.")
  
model1 = transitleastsquares(sap_time_clean, flatten_lc1_clean)
model2 = transitleastsquares(pdc_time_clean, flatten_lc2_clean)

min_period = 0.5  # days, or a bit more than your cadence
max_period = (sap_time_clean.max() - sap_time_clean.min()) / 2  # maximum orbtial period is half baseline
print(max_period)


#import IPython; IPython.embed() # --> to mess around and investigate inside the code

# getting an error on the results module here 

results1 = model1.power(period_min = min_period, period_max = max_period) # now inputting minimum and maximum period to try and fix valueError of empty TLS
results2 = model2.power(period_min = min_period, period_max = max_period)

#figure2, axs2 = plt.subplots(nrows = 2, figsize = (10,12))
#plt.subplots_adjust(hspace=0.3)

# whats the TLS found orbiital period 
    
period1 = results1.period
sde1 = results1.SDE

period2 = results2.period
sde2 = results2.SDE

periods = results2.periods
power = results2.power


plaxes['G'].plot(periods, power, color='black')
plaxes['G'].set_xlabel("Trial Period (days)")
plaxes['G'].set_ylabel("TLS Power (SDE)")
plaxes['G'].set_title("TLS Detection Spectrum")
plaxes['G'].grid(True)
    
#axs2[0].scatter(results1.folded_phase, results1.folded_y, marker = 'o', s = 0.25, color = 'black', label = f'SAP phase-folded\nTLS Period = {period1:.4f} d\nSDE = {sde1:.2f}')
#axs2[0].scatter(results1.model_folded_phase, results1.model_folded_model, s = 1, color = 'red', label = 'TLS MODEL for SAP Flux')

plaxes['E'].scatter(results2.folded_phase, results2.folded_y, marker = 'o', s = 0.25, color = 'black', label = f'PDCSAP phase-folded\nTLS Period = {period2:.4f} d\nSDE = {sde2:.2f}')
plaxes['E'].scatter(results2.model_folded_phase, results2.model_folded_model, s = 1, color = 'red', label = 'TLS MODEL for PDCSAP Flux')
plaxes['E'].set_title(f" TLS result algorithm on TIC {tic_id}")

# Predict transit times using TLS results
tls_period = results2.period
tls_t0 = results2.T0

# Compute all expected transit times within observed time span
epochs = np.arange(-1000, 1000)
transit_times = tls_t0 + (tls_period * epochs)
# Compute a y-position slightly below the light curve's minimum flux
y_marker = np.nanmin(flatten_lc2) - 0.005  # or adjust the offset

# Only keep transits that fall within your light curve time span
in_transit = (transit_times > pdc_time_binned.min()) & (transit_times < pdc_time_binned.max())
visible_transits = transit_times[in_transit]

# Plot blue triangles at each expected transit time
for t in visible_transits:
    plaxes['B'].scatter(t, y_marker, marker='^', color='blue', s=20, zorder=3, label='Transit time' if t==visible_transits[0] else "")
    plaxes['B'].scatter(t, y_marker, marker = '^', color = 'blue', s=20, zorder=3, label='Transit time' if t == visible_transits[0] else "")

#savpath2 = f"/home/gurmeher/gurmeher/detrending/TLS_TIC_{tic_id}.pdf"
'''for ax in axs2:
    ax.legend()'''

#figure2.savefig(savpath2, bbox_inches ='tight', format = 'pdf')
#plt.close(figure2)



'''with PdfPages(multipage_pdf_path) as pdf:

    # Save first figure as page 1
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
    # Saving figure as page 2
    pdf.savefig(fig_phase, bbox_inches = 'tight')
    plt.close(fig_phase)

    # Save figure as page 3
    pdf.savefig(fig3, bbox_inches = 'tight')
    plt.close(fig3)

    # Save figure as page 4
    pdf.savefig(figure2, bbox_inches='tight')
    plt.close(figure2)

    pdf.savefig(f, bbox_inches='tight')
    plt.close(f)'''

subplotter.savefig(pdf_path, bbox_inches='tight', format='pdf')
plt.close(subplotter)
        
'''model1 = transitleastsquares(sap_time_binned, flatten_lc1)
model2 = transitleastsquares(pdc_time_binned, flatten_lc2)

results1 = model1.power() # not inputting minimum and maximum period right now, can add if required
results2 = model2.power()

figure2, axs2 = plt.subplots(nrows = 2, figsize = (10,12))
plt.subplots_adjust(hspace=0.3)

# whats the TLS found orbiital period 
import IPython; IPython.embed()

axs2[0].scatter(results1.folded_phase, results1.folded_y, marker = 'o', s = 0.25, color = 'black', label = 'SAP binned and flattened data phase folded')
axs2[0].plot(results1.model_folded_phase, results1.model_folded_model, color = 'red', label = 'TLS MODEL for SAP Flux')

axs2[1].scatter(results2.folded_phase, results2.folded_y, marker = 'o', s = 0.25, color = 'black', label = 'PDCSAP binned and flattened data phase folded')
axs2[1].plot(results2.model_folded_phase, results2.model_folded_model, color = 'red', label = 'TLS MODEL for PDCSAP Flux')

savpath2 = f"/home/gurmeher/gurmeher/detrending/TLS_TIC_{tic_id}.pdf"
for ax in axs2:
    ax.legend()

figure2.savefig(savpath2, bbox_inches ='tight', format = 'pdf')
plt.close(figure2)'''

