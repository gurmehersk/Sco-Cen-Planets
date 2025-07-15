import numpy as np
import matplotlib.pyplot as plt

# Random seed for reproducibility
np.random.seed(42)

# Common parameters
num_points = 15
phi = np.linspace(0, 2 * np.pi, num_points)  # orbital phase
noise_std = 0.5  # km/s, observational noise

# === EB scenario ===
K_eb = 100  # km/s, semi-amplitude of binary RV
rv_eb = K_eb * np.sin(phi) + np.random.normal(0, noise_std, num_points)

# === Planet scenario ===
K_planet = 0.2  # km/s (200 m/s)
rv_planet = K_planet * np.sin(phi) + np.random.normal(0, noise_std, num_points)

# === Plot ===
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

axs[0].scatter(phi, rv_eb, color='darkred', label='EB data')
axs[0].plot(phi, K_eb * np.sin(phi), '--', color='black', label='Model')
axs[0].set_title('Eclipsing Binary')
axs[0].set_xlabel('Orbital Phase (rad)')
axs[0].set_ylabel('RV (km/s)')
axs[0].legend()

axs[1].scatter(phi, rv_planet, color='navy', label='Planet data')
axs[1].plot(phi, K_planet * np.sin(phi), '--', color='black', label='Model')
axs[1].set_title('Planetary Companion')
axs[1].set_xlabel('Orbital Phase (rad)')
axs[1].legend()

plt.tight_layout()
plt.show()
