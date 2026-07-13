"""Standalone dilution-factor (mdilution) calculator.

Computes the transit-fitting dilution factor for a target star with
one resolved, non-varying neighbor blended into the same photometric
aperture -- the quantity juliet calls ``mdilution``.

HOW THE CALCULATION WORKS
--------------------------
Both stars are modeled as circular 2D Gaussian point-spread functions
(PSFs), each normalized so its integral over all space equals its
total flux (set from the target/neighbor magnitude difference). The
neighbor is placed at the given separation and position angle (0 deg
= North, increasing toward East; e.g., NE = 45 deg) relative to the
target.

The photometric aperture is a circle of the given radius, centered on
the target. Because a circular aperture centered on the TARGET does
not have a simple closed-form overlap with the NEIGHBOR's off-center
Gaussian, both PSFs are sampled on a fine pixel grid and numerically
summed over the aperture region to get how much of each star's true
flux actually lands inside it. The dilution factor is then:

    mdilution = F_target_in_aperture
                / (F_target_in_aperture + F_neighbor_in_aperture)

mdilution = 1.0 means no measurable dilution (the neighbor's flux
does not reach the aperture); smaller values mean a larger fraction
of the light inside the aperture is non-varying neighbor flux,
diluting the observed transit depth.

USAGE
-----
Edit the "USER INPUTS" section at the bottom of this file, then run:

    python compute_mdilution.py

Aperture radius and PSF width can each be entered in either pixels
or arcsec (set the matching *_UNIT variable); a plate scale
(arcsec/pixel) is required whenever anything is given in pixels.
"""

import logging
from dataclasses import dataclass

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def flux_ratio_from_delta_mag(delta_mag):
    """Convert a magnitude difference into a target/neighbor flux ratio.

    Parameters
    ----------
    delta_mag : float
        How many magnitudes fainter the neighbor is than the target
        (should be >= 0 if the neighbor really is the fainter one).

    Returns
    -------
    float
        target_flux / neighbor_flux, with neighbor_flux fixed at 1.0.
    """
    return 10.0 ** (delta_mag / 2.5)


def to_arcsec(value, unit, plate_scale_arcsec_per_pixel=None):
    """Convert a value given in 'arcsec' or 'pixels' into arcsec.

    Parameters
    ----------
    value : float
        The numeric value to convert.
    unit : str
        Either "arcsec" or "pixels" (case-insensitive).
    plate_scale_arcsec_per_pixel : float, optional
        Required only if ``unit`` is "pixels".

    Returns
    -------
    float
        The value expressed in arcsec.
    """
    unit_clean = unit.strip().lower()
    if unit_clean in ("arcsec", "arcseconds"):
        return value
    if unit_clean in ("pixel", "pixels", "px"):
        if plate_scale_arcsec_per_pixel is None:
            raise ValueError(
                "plate_scale_arcsec_per_pixel is required to convert "
                "a pixel value to arcsec."
            )
        return value * plate_scale_arcsec_per_pixel
    raise ValueError(
        f"Unrecognized unit '{unit}'; use 'arcsec' or 'pixels'."
    )


@dataclass
class DilutionInputs:
    """All inputs needed to compute a two-star dilution factor.

    Attributes
    ----------
    separation_arcsec : float
        On-sky separation between target and neighbor, in arcsec.
    position_angle_deg : float
        Position angle of the neighbor relative to the target, in
        degrees (0 = North, increasing toward East; e.g., NE = 45).
    delta_mag : float
        How many magnitudes fainter the neighbor is than the target.
    aperture_radius_arcsec : float
        Photometric aperture radius, in arcsec, centered on the
        target star.
    psf_fwhm_arcsec : float
        PSF full width at half maximum, in arcsec.
    pixel_scale_arcsec : float
        Simulation grid sampling (not the real instrument plate scale
        -- just how finely the Gaussians are numerically sampled).
    grid_half_width_arcsec : float or None
        Half-width of the simulated grid. If None, a value that
        comfortably covers the separation, aperture, and PSF wings is
        chosen automatically.
    """

    separation_arcsec: float
    position_angle_deg: float
    delta_mag: float
    aperture_radius_arcsec: float
    psf_fwhm_arcsec: float
    pixel_scale_arcsec: float = 0.02
    grid_half_width_arcsec: float = None

    def __post_init__(self):
        if self.grid_half_width_arcsec is None:
            self.grid_half_width_arcsec = (
                self.separation_arcsec
                + self.aperture_radius_arcsec
                + 5.0 * self.psf_fwhm_arcsec
            )


def compute_mdilution(inputs):
    """Compute the dilution factor for a target plus one resolved
    neighbor star.

    Parameters
    ----------
    inputs : DilutionInputs
        All geometric, photometric, and PSF parameters.

    Returns
    -------
    results : dict
        Keys: ``mdilution``, ``target_flux_in_aperture``,
        ``neighbor_flux_in_aperture``, ``neighbor_pct_of_aperture``,
        ``target_own_flux_recovered_pct``.
    """
    sigma = inputs.psf_fwhm_arcsec * FWHM_TO_SIGMA
    scale = inputs.pixel_scale_arcsec
    half_width = inputs.grid_half_width_arcsec

    n_pix = int(round(2.0 * half_width / scale))
    if n_pix % 2 == 0:
        n_pix += 1
    coords = (np.arange(n_pix) - n_pix // 2) * scale
    x_grid, y_grid = np.meshgrid(coords, coords)
    pixel_area = scale ** 2

    pa_rad = np.radians(inputs.position_angle_deg)
    neighbor_x0 = inputs.separation_arcsec * np.sin(pa_rad)
    neighbor_y0 = inputs.separation_arcsec * np.cos(pa_rad)

    target_flux = flux_ratio_from_delta_mag(inputs.delta_mag)
    neighbor_flux = 1.0

    def _gaussian_image(x0, y0, total_flux):
        amplitude = total_flux / (2.0 * np.pi * sigma ** 2)
        r_squared = (x_grid - x0) ** 2 + (y_grid - y0) ** 2
        density = amplitude * np.exp(-r_squared / (2.0 * sigma ** 2))
        return density * pixel_area

    target_image = _gaussian_image(0.0, 0.0, target_flux)
    neighbor_image = _gaussian_image(
        neighbor_x0, neighbor_y0, neighbor_flux
    )

    aperture_mask = (
        x_grid ** 2 + y_grid ** 2 <= inputs.aperture_radius_arcsec ** 2
    )

    f_target = target_image[aperture_mask].sum()
    f_neighbor = neighbor_image[aperture_mask].sum()
    total_in_aperture = f_target + f_neighbor

    mdilution = f_target / total_in_aperture
    neighbor_pct = 100.0 * f_neighbor / total_in_aperture
    target_recovered_pct = 100.0 * f_target / target_flux

    results = {
        "mdilution": mdilution,
        "target_flux_in_aperture": f_target,
        "neighbor_flux_in_aperture": f_neighbor,
        "neighbor_pct_of_aperture": neighbor_pct,
        "target_own_flux_recovered_pct": target_recovered_pct,
    }

    logger.info("=" * 60)
    logger.info("DILUTION FACTOR RESULT")
    logger.info("=" * 60)
    logger.info(
        "separation=%.3f\" @ PA=%.1f deg, delta_mag=%.2f",
        inputs.separation_arcsec,
        inputs.position_angle_deg,
        inputs.delta_mag,
    )
    logger.info(
        "aperture_radius=%.4f\", PSF_FWHM=%.4f\"",
        inputs.aperture_radius_arcsec,
        inputs.psf_fwhm_arcsec,
    )
    logger.info("target flux in aperture:   %.6f", f_target)
    logger.info(
        "neighbor flux in aperture: %.6f (%.4f%% of aperture flux)",
        f_neighbor, neighbor_pct,
    )
    logger.info("mdilution = %.6f", mdilution)
    logger.info(
        "target recovers %.2f%% of its own total flux at this "
        "aperture radius",
        target_recovered_pct,
    )
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    # ================================================================
    # USER INPUTS -- edit these values, then run: python
    # compute_mdilution.py
    # ================================================================
    SEPARATION_ARCSEC = 2.6
    POSITION_ANGLE_DEG = 45.0     # 0=North, 90=East, 45=NE, etc.
    DELTA_MAG = 2.5                # neighbor is this many mags fainter

    PLATE_SCALE_ARCSEC_PER_PIXEL = 0.389  # from the FITS header

    APERTURE_RADIUS_VALUE = 5.0
    APERTURE_RADIUS_UNIT = "pixels"        # "pixels" or "arcsec"

    PSF_FWHM_VALUE = 1.74
    PSF_FWHM_UNIT = "arcsec"               # "pixels" or "arcsec"
    # ================================================================

    aperture_radius_arcsec = to_arcsec(
        APERTURE_RADIUS_VALUE,
        APERTURE_RADIUS_UNIT,
        PLATE_SCALE_ARCSEC_PER_PIXEL,
    )
    psf_fwhm_arcsec = to_arcsec(
        PSF_FWHM_VALUE,
        PSF_FWHM_UNIT,
        PLATE_SCALE_ARCSEC_PER_PIXEL,
    )

    dilution_inputs = DilutionInputs(
        separation_arcsec=SEPARATION_ARCSEC,
        position_angle_deg=POSITION_ANGLE_DEG,
        delta_mag=DELTA_MAG,
        aperture_radius_arcsec=aperture_radius_arcsec,
        psf_fwhm_arcsec=psf_fwhm_arcsec,
    )
    compute_mdilution(dilution_inputs)
