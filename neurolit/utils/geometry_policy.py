"""Shared geometry policy helpers for conforming and VINN scaling.
"""

from collections.abc import Sequence

DEFAULT_CONFORM_VOX_SIZE: str = "min"
DEFAULT_CONFORM_IMG_SIZE: str = "auto"
DEFAULT_CONFORM_ORIENTATION: str = "lia"
DEFAULT_CONFORM_RESCALE: int = 255
VINN_REFERENCE_FOV_MM: float = 256.0


def default_conform_kwargs() -> dict[str, str | int | None]:
    """Return default conform parameters used by the inpainting workflow."""
    return {
        "vox_size": DEFAULT_CONFORM_VOX_SIZE,
        "img_size": DEFAULT_CONFORM_IMG_SIZE,
        "orientation": DEFAULT_CONFORM_ORIENTATION,
        "rescale": DEFAULT_CONFORM_RESCALE,
        "dtype": None,
    }


def vinn_internal_resolution_mm(
    internal_size: Sequence[int],
    *,
    reference_fov_mm: float = VINN_REFERENCE_FOV_MM,
) -> float:
    """Compute VINN internal resolution in mm from reference FoV and size."""
    if not internal_size:
        raise ValueError("internal_size must not be empty")
    side = int(internal_size[0])
    if side <= 0:
        raise ValueError("internal_size values must be > 0")
    if reference_fov_mm <= 0:
        raise ValueError("reference_fov_mm must be > 0")
    return float(reference_fov_mm) / float(side)


def vinn_scale_factor_from_zooms(
    internal_size: Sequence[int],
    zooms: Sequence[float],
    *,
    axis: int = 0,
    reference_fov_mm: float = VINN_REFERENCE_FOV_MM,
) -> float:
    """Compute scalar VINN scale factor used in inference.

    The current inference code path consumes a scalar and therefore uses one
    representative zoom axis (default: x-axis).
    """
    if not zooms:
        raise ValueError("zooms must not be empty")
    if axis < 0 or axis >= len(zooms):
        raise ValueError("axis out of range for zooms")
    zoom = float(zooms[axis])
    if zoom <= 0:
        raise ValueError("zooms values must be > 0")
    return vinn_internal_resolution_mm(internal_size, reference_fov_mm=reference_fov_mm) / zoom