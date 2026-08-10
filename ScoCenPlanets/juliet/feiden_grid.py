"""
feiden_grid.py -- parse and interpolate Feiden (2016) mass tracks.

Source: github.com/gfeiden/MagneticUpperSco (models/trk/std, models/trk/mag).
Solar metallicity only (GS98_p000) -- no [Fe/H] flexibility in this repo.
Repo README flags known issues in the public tracks below 10 Myr; email
gregory.feiden@ung.edu for the corrected grid if that matters for your age range.

.ntrk column format (whitespace-delimited, 1-indexed):
  1 Model#, 2 shells, 3 AGE [Gyr], 4 log(L/Lsun), 5 log(R/Rsun), 6 log(g),
  7 log(Teff), ... (rest not needed here; some files have corrupted values
  in later columns so we restrict to usecols=(0..6))
"""

import numpy as np
from pathlib import Path


def parse_ntrk(path):
    """Return age [Gyr], log10(R/Rsun), log10(Teff) arrays from a .ntrk file."""
    data = np.loadtxt(path, comments="#", usecols=(0, 1, 2, 3, 4, 5, 6))
    return data[:, 2], data[:, 4], data[:, 6]  # age, logR, logTeff


def mass_from_filename(path):
    # e.g. m0300_GS98_p000_p0_y28_mlt1.884.ntrk -> 0.300
    return int(path.name[1:5]) / 1000.0


class FeidenGrid:
    """
    Loads the tracks in [mass_lo, mass_hi] from a Feiden index file once,
    resamples each onto a common log-age grid, and interpolates in mass
    on demand via track_at_mass().
    """

    def __init__(self, models_dir, index_file, tracks_subdir, mass_lo, mass_hi,
                 age_grid_gyr=None):
        models_dir = Path(models_dir)
        with open(models_dir / index_file) as f:
            rel_paths = [line.strip() for line in f if line.strip()]
        all_masses = {mass_from_filename(models_dir / rp): (models_dir / rp)
                      for rp in rel_paths}

        self.masses = sorted(m for m in all_masses if mass_lo <= m <= mass_hi)
        if len(self.masses) < 2:
            raise ValueError(
                f"Need >=2 grid masses in [{mass_lo}, {mass_hi}] for {tracks_subdir}; "
                f"found {self.masses}. Widen the window."
            )

        if age_grid_gyr is None:
            age_grid_gyr = np.logspace(np.log10(0.0005), np.log10(0.5), 600)  # 0.5 Myr-500 Myr
        self.age_grid = age_grid_gyr

        self.logR_grid, self.logTeff_grid = {}, {}
        self.native_age_bounds = {}  # track each mass's own valid age range
        for m in self.masses:
            age, logR, logTeff = parse_ntrk(all_masses[m])
            self.native_age_bounds[m] = (age.min(), age.max())
            logR_i = np.interp(np.log10(self.age_grid), np.log10(age), logR)
            logTeff_i = np.interp(np.log10(self.age_grid), np.log10(age), logTeff)
            # np.interp silently clamps to the boundary value outside a track's
            # own tabulated range -- some Feiden magnetic tracks stop as early
            # as <5 Myr (evidently non-converged runs), so without this mask
            # any age beyond that gets a frozen, unphysical value with no
            # warning. Mask those out instead of trusting the clamp.
            out_of_range = (self.age_grid < age.min()) | (self.age_grid > age.max())
            logR_i[out_of_range] = np.nan
            logTeff_i[out_of_range] = np.nan
            self.logR_grid[m] = logR_i
            self.logTeff_grid[m] = logTeff_i

    def track_at_mass(self, m_target):
        """Return (age_grid [Gyr], R(age) [Rsun], Teff(age) [K]) at m_target,
        linearly interpolated between the two bracketing grid masses. Ages
        beyond either bracket's own native coverage come back as NaN rather
        than a silently clamped value -- callers should nan-aware argmin."""
        masses = np.array(self.masses)
        if m_target <= masses[0] or m_target >= masses[-1]:
            raise ValueError(f"M={m_target:.4f} Msun outside loaded grid "
                              f"[{masses[0]}, {masses[-1]}]; widen mass_lo/mass_hi.")
        i = np.searchsorted(masses, m_target) - 1
        m0, m1 = masses[i], masses[i + 1]
        w = (m_target - m0) / (m1 - m0)
        logR = (1 - w) * self.logR_grid[m0] + w * self.logR_grid[m1]
        logTeff = (1 - w) * self.logTeff_grid[m0] + w * self.logTeff_grid[m1]
        return self.age_grid, 10**logR, 10**logTeff

    def isochrone_at_age(self, target_age_gyr, n_fine=400):
        """Return (mass_fine [Msun], R(mass) [Rsun], Teff(mass) [K]) -- a
        cross-section across every loaded mass at the nearest tabulated age
        on the common age_grid, then refined onto a fine mass grid via 1-D
        interpolation. Masses whose own track doesn't natively reach
        target_age_gyr are dropped entirely rather than contributing a
        clamped/frozen value from their last real tabulated age."""
        if not (self.age_grid[0] <= target_age_gyr <= self.age_grid[-1]):
            raise ValueError(f"age={target_age_gyr:.5f} Gyr outside grid "
                              f"[{self.age_grid[0]:.5f}, {self.age_grid[-1]:.5f}]")
        idx = np.argmin(np.abs(self.age_grid - target_age_gyr))

        masses = np.array(self.masses)
        valid = np.array([lo <= target_age_gyr <= hi
                           for (lo, hi) in (self.native_age_bounds[m] for m in masses)])
        if valid.sum() < 2:
            raise ValueError(f"Fewer than 2 masses have native coverage at "
                              f"age={target_age_gyr*1000:.2f} Myr; dropped: "
                              f"{masses[~valid].tolist()}")
        masses = masses[valid]
        logR_at_age = np.array([self.logR_grid[m][idx] for m in masses])
        logTeff_at_age = np.array([self.logTeff_grid[m][idx] for m in masses])

        mass_fine = np.linspace(masses[0], masses[-1], n_fine)
        logR_fine = np.interp(mass_fine, masses, logR_at_age)
        logTeff_fine = np.interp(mass_fine, masses, logTeff_at_age)
        return mass_fine, 10**logR_fine, 10**logTeff_fine


def best_fit_ages(grid, mstar_samples, rstar_samples_rsun, teff_obs, sig_teff, sig_r):
    """
    For each (mass, radius) draw, interpolate the grid to that mass and find
    the age whose (Teff, R) jointly best match (teff_obs, that draw's R) --
    the 2-observable, 1-free-parameter age fit. Returns age in Myr, NaN for
    any draw whose mass falls outside the loaded grid window.
    """
    chi_array = []
    ages_myr = np.full(len(mstar_samples), np.nan)
    for i, (m, r) in enumerate(zip(mstar_samples, rstar_samples_rsun)):
        try:
            age_grid, R_model, Teff_model = grid.track_at_mass(m)
        except ValueError:
            continue
        chi2 = ((teff_obs - Teff_model) / sig_teff) ** 2 + ((r - R_model) / sig_r) ** 2
        chi_array.append(chi2[np.argmin(chi2)])
        ages_myr[i] = age_grid[np.argmin(chi2)] * 1000.0  # Gyr -> Myr
    return ages_myr, np.array(chi_array)


def best_fit_masses(grid, age_samples_gyr, rstar_samples_rsun, teff_obs, sig_teff, sig_r):
    """
    Mirror of best_fit_ages with mass and age swapped: for each (age, radius)
    draw, slice the grid to an isochrone at that age and find the mass whose
    (Teff, R) jointly best match (teff_obs, that draw's R). Returns mass in
    Msun, NaN for any draw whose age falls outside the loaded age_grid.
    """
    chi_array = []
    masses_out = np.full(len(age_samples_gyr), np.nan)
    for i, (age, r) in enumerate(zip(age_samples_gyr, rstar_samples_rsun)):
        try:
            mass_fine, R_model, Teff_model = grid.isochrone_at_age(age)
        except ValueError:
            continue
        chi2 = ((teff_obs - Teff_model) / sig_teff) ** 2 + ((r - R_model) / sig_r) ** 2
        chi_array.append(chi2[np.argmin(chi2)])
        masses_out[i] = mass_fine[np.argmin(chi2)]
    return masses_out, np.array(chi_array)