"""Synthetic two-star scene generator and aperture-photometry calculator.

This module builds a 2D synthetic image containing two stars, each
modeled as a circular 2D Gaussian point-spread function (PSF), and then
measures what fraction of the total stellar flux falls within a fixed
circular photometric aperture centered on the brighter star.

Stated assumptions
-------------------
The problem statement leaves two conventions implicit, so they are
fixed explicitly here:

1. "PSF width" is taken to mean the Gaussian's full width at half
   maximum (FWHM), the standard way seeing/PSF widths are quoted in
   observational astronomy.
2. The "4 arcsecond aperture" is taken to be the aperture RADIUS
   (common convention in aperture photometry), not the diameter.

Both stars share the same PSF shape (sigma); only their total flux
differs, with the brighter star set to 10x the flux of the fainter
star. Because the 4" aperture radius is larger than the 2.2"
separation, the aperture centered on the bright star also picks up
essentially all of the fainter companion's flux -- this "companion
contamination" is reported explicitly alongside the total enclosed
fraction, since it is often the more scientifically relevant number
for blended/close-pair aperture photometry.

Modeling a real (e.g., AstroImageJ) aperture setup
---------------------------------------------------
Real observations are usually described by a position angle and
magnitude difference rather than a Cartesian offset and flux ratio,
and AstroImageJ (AIJ) ``.apertures`` files specify aperture and sky
annulus radii in PIXELS, not arcsec. Two helper functions bridge
these conventions:

* ``flux_ratio_from_delta_mag(delta_mag)`` converts a magnitude
  difference into the ``flux_ratio`` field above.
* ``arcsec_from_pixels(n_pixels, plate_scale_arcsec_per_pixel)``
  converts an AIJ pixel radius (e.g., ``.aperture.radius``,
  ``.aperture.rback1``, ``.aperture.rback2``) into arcsec using the
  instrument's plate scale, which is NOT stored in the ``.apertures``
  file itself and must be supplied separately.

``SceneConfig.position_angle_deg`` places the companion using the
standard astronomical convention: 0 deg = North, increasing toward
East (so "2.6 arcsec NE" is separation=2.6, position_angle_deg=45).
"""

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Conversion factor from FWHM to Gaussian sigma.
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass
class SceneConfig:
    """Configuration parameters for the synthetic two-star scene.

    Attributes
    ----------
    separation_arcsec : float
        On-sky separation between the two stars, in arcsec.
    position_angle_deg : float
        Position angle of the fainter star relative to the brighter
        star, in degrees, using the standard astronomical convention
        (0 = North, increasing toward East). For example, due East is
        90, and Northeast is 45.
    psf_fwhm_arcsec : float
        Gaussian PSF full width at half maximum, in arcsec.
    flux_ratio : float
        Ratio of the brighter star's total flux to the fainter star's
        total flux (bright / faint).
    aperture_radius_arcsec : float
        Radius of the circular photometric aperture, in arcsec.
    sky_annulus_arcsec : tuple of float or None
        Optional ``(r_in, r_out)`` sky-background annulus, in arcsec,
        centered on the brighter star (mirrors AIJ's ``rback1``/
        ``rback2``). If None, no background subtraction is performed.
    pixel_scale_arcsec : float
        Simulated pixel scale, in arcsec/pixel. This is the SIMULATION
        grid sampling and is independent of the real instrument's
        plate scale used to interpret an AIJ ``.apertures`` file.
    image_half_width_arcsec : float
        Half-width of the square simulated image, in arcsec.
    faint_star_flux : float
        Total flux of the fainter star, in arbitrary flux units.
    """

    separation_arcsec: float = 2.2
    position_angle_deg: float = 0.0
    psf_fwhm_arcsec: float = 1.0
    flux_ratio: float = 10.0
    aperture_radius_arcsec: float = 4.0
    sky_annulus_arcsec: tuple = None
    pixel_scale_arcsec: float = 0.05
    image_half_width_arcsec: float = 8.0
    faint_star_flux: float = 1.0


def flux_ratio_from_delta_mag(delta_mag):
    """Convert a magnitude difference into a (bright / faint) flux ratio.

    Parameters
    ----------
    delta_mag : float
        Magnitude difference, faint minus bright (i.e., how many
        magnitudes fainter the companion is). Must be non-negative for
        the companion to actually be the fainter star.

    Returns
    -------
    float
        The corresponding flux ratio, bright_flux / faint_flux.
    """
    return 10.0 ** (delta_mag / 2.5)


def arcsec_from_pixels(n_pixels, plate_scale_arcsec_per_pixel):
    """Convert a pixel-based radius (e.g., from an AIJ .apertures file)
    into arcsec, given the instrument's plate scale.

    Parameters
    ----------
    n_pixels : float
        Radius in pixels (e.g., ``.aperture.radius``, ``.rback1``).
    plate_scale_arcsec_per_pixel : float
        Instrument plate scale, in arcsec/pixel. This value is NOT
        stored in an AIJ ``.apertures`` file and must be looked up for
        the specific camera/binning mode used.

    Returns
    -------
    float
        Radius in arcsec.
    """
    return n_pixels * plate_scale_arcsec_per_pixel


def _make_pixel_grid(config):
    """Build a 1D pixel-center coordinate array and a 2D coordinate grid.

    Returns an odd-length coordinate array so that a well-defined
    central pixel exists at arcsec offset (0, 0).
    """
    n_pix = int(round(
        2 * config.image_half_width_arcsec / config.pixel_scale_arcsec
    ))
    if n_pix % 2 == 0:
        n_pix += 1
    coords_1d = (np.arange(n_pix) - n_pix // 2) * config.pixel_scale_arcsec
    x_grid, y_grid = np.meshgrid(coords_1d, coords_1d)
    return coords_1d, x_grid, y_grid


def _gaussian_flux_density(x_grid, y_grid, x0, y0, sigma, total_flux):
    """Evaluate a 2D circular Gaussian flux-density field.

    The Gaussian is normalized so that its integral over an infinite
    plane equals ``total_flux``. Multiplying the returned array by the
    pixel area and summing over a region approximates the flux
    integral over that region.
    """
    amplitude = total_flux / (2.0 * np.pi * sigma ** 2)
    r_squared = (x_grid - x0) ** 2 + (y_grid - y0) ** 2
    return amplitude * np.exp(-r_squared / (2.0 * sigma ** 2))


def generate_two_star_scene(config=None):
    """Generate a synthetic 2D image containing two Gaussian-PSF stars.

    Parameters
    ----------
    config : SceneConfig, optional
        Scene configuration. Defaults to ``SceneConfig()``, which
        matches the nominal parameters (2.2" separation, 1" FWHM,
        10:1 flux ratio, 4" aperture radius).

    Returns
    -------
    image : numpy.ndarray
        2D array of flux-per-pixel for the combined (bright + faint)
        scene.
    coords_1d : numpy.ndarray
        1D array of pixel-center coordinates in arcsec, shared by both
        image axes (x and y).
    star_positions : dict
        Per-star metadata (position, intrinsic flux, and individual
        flux-per-pixel image) keyed by ``"bright"`` and ``"faint"``.
    """
    if config is None:
        config = SceneConfig()

    logger.info(
        "Building scene: separation=%.2f\" @ PA=%.1f deg, "
        "PSF FWHM=%.2f\", flux_ratio=%.2f:1, aperture_radius=%.2f\"",
        config.separation_arcsec,
        config.position_angle_deg,
        config.psf_fwhm_arcsec,
        config.flux_ratio,
        config.aperture_radius_arcsec,
    )

    sigma = config.psf_fwhm_arcsec * FWHM_TO_SIGMA
    coords_1d, x_grid, y_grid = _make_pixel_grid(config)
    pixel_area = config.pixel_scale_arcsec ** 2

    bright_flux = config.faint_star_flux * config.flux_ratio
    faint_flux = config.faint_star_flux

    # The bright star anchors the coordinate system at (0, 0). The
    # faint companion is placed using the standard astronomical
    # position-angle convention: 0 deg = North (+y), increasing toward
    # East (+x), so "NE" corresponds to PA = 45 deg.
    pa_rad = np.radians(config.position_angle_deg)
    bright_x0, bright_y0 = 0.0, 0.0
    faint_x0 = config.separation_arcsec * np.sin(pa_rad)
    faint_y0 = config.separation_arcsec * np.cos(pa_rad)

    bright_density = _gaussian_flux_density(
        x_grid, y_grid, bright_x0, bright_y0, sigma, bright_flux
    )
    faint_density = _gaussian_flux_density(
        x_grid, y_grid, faint_x0, faint_y0, sigma, faint_flux
    )

    bright_image = bright_density * pixel_area
    faint_image = faint_density * pixel_area
    image = bright_image + faint_image

    logger.info(
        "Simulated grid: %d x %d pixels, pixel_scale=%.3f\"/pix, "
        "sigma=%.3f\"",
        coords_1d.size, coords_1d.size,
        config.pixel_scale_arcsec, sigma,
    )
    logger.info(
        "Flux injection check -- bright: %.6f (target %.6f); "
        "faint: %.6f (target %.6f)",
        bright_image.sum(), bright_flux,
        faint_image.sum(), faint_flux,
    )

    star_positions = {
        "bright": {
            "x0": bright_x0,
            "y0": bright_y0,
            "flux": bright_flux,
            "image": bright_image,
        },
        "faint": {
            "x0": faint_x0,
            "y0": faint_y0,
            "flux": faint_flux,
            "image": faint_image,
        },
    }
    return image, coords_1d, star_positions


def compute_enclosed_flux_fraction(
    image, coords_1d, star_positions, config=None
):
    """Perform aperture photometry centered on the brighter star.

    Parameters
    ----------
    image : numpy.ndarray
        Combined two-star image, as returned by
        ``generate_two_star_scene``.
    coords_1d : numpy.ndarray
        Shared 1D pixel-center coordinate array, in arcsec.
    star_positions : dict
        Per-star metadata, as returned by ``generate_two_star_scene``.
    config : SceneConfig, optional
        Supplies the aperture radius; defaults to ``SceneConfig()``.

    Returns
    -------
    results : dict
        Dictionary of derived photometric quantities, including the
        enclosed flux fraction and the companion-contamination
        fraction.
    """
    if config is None:
        config = SceneConfig()

    x_grid, y_grid = np.meshgrid(coords_1d, coords_1d)
    x0 = star_positions["bright"]["x0"]
    y0 = star_positions["bright"]["y0"]
    radius = config.aperture_radius_arcsec

    aperture_mask = (x_grid - x0) ** 2 + (y_grid - y0) ** 2 <= radius ** 2

    sky_level_per_pixel = 0.0
    if config.sky_annulus_arcsec is not None:
        r_in, r_out = config.sky_annulus_arcsec
        r_from_bright = np.sqrt((x_grid - x0) ** 2 + (y_grid - y0) ** 2)
        annulus_mask = (r_from_bright >= r_in) & (r_from_bright <= r_out)
        sky_level_per_pixel = float(np.median(image[annulus_mask]))
        n_aperture_pixels = int(aperture_mask.sum())
        logger.info(
            "Sky annulus %.2f\"-%.2f\": median=%.3e flux/pixel "
            "(subtracting %.3e over %d aperture pixels)",
            r_in, r_out, sky_level_per_pixel,
            sky_level_per_pixel, n_aperture_pixels,
        )

    aperture_flux_total = (
        image[aperture_mask].sum()
        - sky_level_per_pixel * aperture_mask.sum()
    )
    aperture_flux_bright = (
        star_positions["bright"]["image"][aperture_mask].sum()
    )
    aperture_flux_faint = (
        star_positions["faint"]["image"][aperture_mask].sum()
    )

    bright_intrinsic_flux = star_positions["bright"]["flux"]
    faint_intrinsic_flux = star_positions["faint"]["flux"]
    total_injected_flux = bright_intrinsic_flux + faint_intrinsic_flux

    fraction_of_total = aperture_flux_total / total_injected_flux
    fraction_of_bright_own_flux = (
        aperture_flux_bright / bright_intrinsic_flux
    )
    fraction_of_faint_own_flux = (
        aperture_flux_faint / faint_intrinsic_flux
    )
    contamination_fraction = (
        aperture_flux_faint / aperture_flux_total
        if aperture_flux_total > 0 else 0.0
    )

    results = {
        "aperture_radius_arcsec": radius,
        "aperture_flux_total": aperture_flux_total,
        "aperture_flux_from_bright": aperture_flux_bright,
        "aperture_flux_from_faint": aperture_flux_faint,
        "total_injected_flux": total_injected_flux,
        "fraction_of_total_flux_enclosed": fraction_of_total,
        "fraction_of_bright_own_flux_enclosed": fraction_of_bright_own_flux,
        "fraction_of_faint_own_flux_enclosed": fraction_of_faint_own_flux,
        "contamination_fraction_from_companion": contamination_fraction,
        "sky_annulus_arcsec": config.sky_annulus_arcsec,
        "sky_level_per_pixel": sky_level_per_pixel,
        "aperture_mask": aperture_mask,
    }

    logger.info("=" * 62)
    logger.info("APERTURE PHOTOMETRY RESULTS")
    logger.info("=" * 62)
    logger.info(
        "Aperture: radius=%.2f\", centered on the brighter star",
        radius,
    )
    logger.info(
        "Total flux enclosed (bright + faint): %.6f flux units",
        aperture_flux_total,
    )
    logger.info(
        "  bright-star contribution: %.6f (%.4f%% of its own flux)",
        aperture_flux_bright, 100 * fraction_of_bright_own_flux,
    )
    logger.info(
        "  faint-star contribution:  %.6f (%.4f%% of its own flux)",
        aperture_flux_faint, 100 * fraction_of_faint_own_flux,
    )
    logger.info(
        "Fraction of TOTAL stellar flux (both stars) enclosed: "
        "%.6f (%.4f%%)",
        fraction_of_total, 100 * fraction_of_total,
    )
    logger.info(
        "Companion contamination: %.4f%% of the flux inside the "
        "aperture actually comes from the fainter neighbor",
        100 * contamination_fraction,
    )
    logger.info("=" * 62)

    return results


def plot_results(image, coords_1d, star_positions, results, output_path):
    """Create and save a two-panel summary figure.

    Left panel: the synthetic scene with the photometric aperture
    overlaid. Right panel: the curve of growth (enclosed flux fraction
    versus aperture radius), with the requested aperture marked.
    """
    fig, (ax_image, ax_curve) = plt.subplots(1, 2, figsize=(12, 5.5))

    extent = [coords_1d[0], coords_1d[-1], coords_1d[0], coords_1d[-1]]
    # Show 10 decades of dynamic range below the peak pixel; this keeps
    # the display legible and avoids float64 underflow far from either
    # star (log10 of an exact 0.0 would otherwise emit a warning).
    peak_value = image.max()
    log_floor = peak_value * 1e-10
    log_image = np.log10(np.maximum(image, log_floor))
    image_display = ax_image.imshow(
        log_image,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=np.log10(log_floor),
        vmax=np.log10(peak_value),
    )
    fig.colorbar(
        image_display, ax=ax_image, label=r"$\log_{10}$(flux per pixel)"
    )

    bright_pos = (
        star_positions["bright"]["x0"], star_positions["bright"]["y0"]
    )
    aperture_circle = plt.Circle(
        bright_pos,
        results["aperture_radius_arcsec"],
        edgecolor="cyan",
        facecolor="none",
        linewidth=1.8,
        label="Photometric aperture",
    )
    ax_image.add_patch(aperture_circle)

    if results.get("sky_annulus_arcsec") is not None:
        r_in, r_out = results["sky_annulus_arcsec"]
        for r_annulus in (r_in, r_out):
            ax_image.add_patch(plt.Circle(
                bright_pos, r_annulus, edgecolor="yellow",
                facecolor="none", linewidth=1.2, linestyle="--",
            ))

    ax_image.scatter(
        *bright_pos, marker="*", s=180, color="white",
        edgecolor="black", label="Bright star", zorder=5,
    )
    ax_image.scatter(
        star_positions["faint"]["x0"], star_positions["faint"]["y0"],
        marker="*", s=90, color="lightgray",
        edgecolor="black", label="Faint star", zorder=5,
    )

    # Standard sky orientation: North up, East to the left.
    ax_image.set_xlim(coords_1d[-1], coords_1d[0])
    ax_image.set_aspect("equal")
    ax_image.set_xlabel("East offset (arcsec)")
    ax_image.set_ylabel("North offset (arcsec)")
    ax_image.set_title("Synthetic two-star scene (N up, E left)")
    ax_image.legend(loc="upper left", fontsize=8, framealpha=0.85)

    x_grid, y_grid = np.meshgrid(coords_1d, coords_1d)
    r_from_bright = np.sqrt(
        (x_grid - bright_pos[0]) ** 2 + (y_grid - bright_pos[1]) ** 2
    )
    radii = np.linspace(0.05, coords_1d[-1] * 0.95, 150)
    total_flux = results["total_injected_flux"]
    enclosed_fractions = [
        image[r_from_bright <= r].sum() / total_flux for r in radii
    ]

    ax_curve.plot(radii, enclosed_fractions, color="darkorange", lw=2)
    ax_curve.axvline(
        results["aperture_radius_arcsec"],
        color="cyan",
        linestyle="--",
        label=f"{results['aperture_radius_arcsec']:.1f}\" aperture",
    )
    ax_curve.axhline(
        results["fraction_of_total_flux_enclosed"],
        color="gray",
        linestyle=":",
    )
    ax_curve.set_xlabel("Aperture radius (arcsec)")
    ax_curve.set_ylabel("Fraction of total stellar flux enclosed")
    ax_curve.set_title("Curve of growth (centered on bright star)")
    ax_curve.set_ylim(0, 1.05)
    ax_curve.legend(loc="lower right", fontsize=9)
    ax_curve.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Figure saved to %s", output_path)
    plt.close(fig)


def run_scenario(config, output_path, scenario_name):
    """Run the generate -> photometer -> plot pipeline for one config."""
    logger.info("--- Scenario: %s ---", scenario_name)
    image, coords_1d, star_positions = generate_two_star_scene(config)
    results = compute_enclosed_flux_fraction(
        image, coords_1d, star_positions, config
    )
    plot_results(image, coords_1d, star_positions, results, output_path)
    return results


def main():
    """Run two scenarios: the illustrative demo, and a real AIJ setup.

    NOTE: the plate scale below (0.4352 arcsec/pixel) is the actual
    value read from the Swope 1m (Las Campanas) FITS header for this
    dataset, used to convert the AIJ pixel-based aperture/annulus radii
    into arcsec. The PSF FWHM (1.0") is still a placeholder for the
    real seeing on that night -- swap in a measured value if known.
    """
    demo_config = SceneConfig()
    run_scenario(
        demo_config, "two_star_demo.png", "illustrative 2.2\"/10:1 demo",
    )

    plate_scale = 0.4352  # arcsec/pixel -- Swope 1m, from FITS header
    real_config = SceneConfig(
        separation_arcsec=2.6,
        position_angle_deg=45.0,  # NE
        flux_ratio=flux_ratio_from_delta_mag(2.5),
        aperture_radius_arcsec=arcsec_from_pixels(2.0, plate_scale),
        sky_annulus_arcsec=(
            arcsec_from_pixels(12.0, plate_scale),
            arcsec_from_pixels(20.0, plate_scale),
        ),
        psf_fwhm_arcsec=1.0,  # placeholder -- use real seeing if known
        image_half_width_arcsec=20.0,
    )
    real_results = run_scenario(
        real_config, "two_star_real_system.png", "real AIJ-derived system",
    )
    return real_results


if __name__ == "__main__":
    main()