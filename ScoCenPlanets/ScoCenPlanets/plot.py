import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("cached_models/TIC88297141_26_10_mcmc_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

def bin_lightcurve(time, flux, bin_minutes=30):
    '''BINNING TO 30 MINUTES'''
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


time_clean = model_dict["time_clean"]
model_lc = model_dict["model_lc"]
flatten = model_dict["flatten_clean"]

flatten_binned_time, flatten_binned = bin_lightcurve(time_clean, flatten, bin_minutes=100)

plt.figure(figsize=(19, 14))
plt.scatter(flatten_binned_time, flatten_binned, s=1, color="black", label="Cleaned Light Curve")
plt.plot(time_clean, model_lc, color="red", label="MCMC Model")
plt.xlabel("Time", fontsize=16)
plt.ylabel("Normalized Flux", fontsize=16)
plt.title("MCMC Model Fit to Cleaned Light Curve", fontsize=20)
plt.legend(fontsize=14)
plt.grid()
plt.show()

plt.savefig("mcmc_model_fit.pdf", dpi=300)
plt.close()


