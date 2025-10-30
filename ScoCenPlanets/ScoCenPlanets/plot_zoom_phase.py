import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("cached_models/TIC88297141_26_10_mcmc_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

def bin_lightcurve(time, flux, bin_minutes=30):
    '''BINNING TO SPECIFIED MINUTES'''
    bin_size = bin_minutes / (24 * 60)  # minutes to days
    bins = np.arange(np.nanmin(time), np.nanmax(time) + bin_size, bin_size)
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

def phase_bin(phase_hours, flux, bin_width_minutes=30):
    '''Bin data in phase space'''
    bin_width_hours = bin_width_minutes / 60.0
    bin_edges = np.arange(-5, 5 + bin_width_hours, bin_width_hours)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned_flux = []

    for i in range(len(bin_edges) - 1):
        in_bin = (phase_hours >= bin_edges[i]) & (phase_hours < bin_edges[i + 1])
        if np.any(in_bin):
            binned_flux.append(np.nanmean(flux[in_bin]))
        else:
            binned_flux.append(np.nan)

    return bin_centers, np.array(binned_flux)

time_clean = model_dict["time_clean"]
model_lc = model_dict["model_lc"]
flatten = model_dict["flatten_clean"]
t0_med = model_dict["t0_med"]
period_med = model_dict["period_med"]

# Bin the data to 10 minutes
flatten_binned_time, flatten_binned = bin_lightcurve(time_clean, flatten, bin_minutes=10)

phase_binned = ((flatten_binned_time - t0_med + 0.5 * period_med) % period_med) / period_med - 0.5
phase_binned_hours = phase_binned * period_med * 24 # Hour conversion


phase_model = ((time_clean - t0_med + 0.5 * period_med) % period_med) / period_med - 0.5
phase_model_hours = phase_model * period_med * 24  # Hour conversion


mask_data = np.abs(phase_binned_hours) <= 5
mask_model = np.abs(phase_model_hours) <= 5

# Phase bin the data
phase_bin_centers, phase_bin_flux = phase_bin(phase_binned_hours[mask_data], 
                                               flatten_binned[mask_data], 
                                               bin_width_minutes=15)

# Sort for plotting
sort_idx_model = np.argsort(phase_model_hours[mask_model])

# Plot
plt.figure(figsize=(14, 6))
plt.scatter(phase_binned_hours[mask_data], 
           flatten_binned[mask_data], 
           s=5, color="gray", alpha=0.3, label="Data")
plt.scatter(phase_bin_centers, phase_bin_flux, 
           s=40, color='dodgerblue', label='Binned Data', zorder=2)
plt.plot(phase_model_hours[mask_model][sort_idx_model], 
        model_lc[mask_model][sort_idx_model], 
        color="C1", label="Median Model", linewidth=2)

plt.xlabel("Time from Transit Center (hours)", fontsize=14)
plt.ylabel("Normalized Flux", fontsize=14)
plt.title("Phase-Folded Transit (±5 hours)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.xlim(-5, 5)
plt.tight_layout()
plt.savefig("mcmc_phase_folded_zoom_5hours.pdf", dpi=300, bbox_inches="tight")
plt.show()