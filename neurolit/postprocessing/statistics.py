"""Statistics backend selection, independent of postprocessing execution order."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

StatsBackend = Literal["fastsurfer", "freesurfer"]
StepStatus = Literal["succeeded", "skipped", "failed"]


@dataclass(frozen=True)
class StepResult:
    """Record a step's outcome and the inputs and outputs used in this run."""

    name: str
    status: StepStatus
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    backend: str | None = None
    reason: str = ""

    def __bool__(self) -> bool:
        """Retain boolean success checks for callers of execution helpers."""
        return self.status == "succeeded"


def select_segstats_backend(config: dict[str, Any], default_backend: StatsBackend) -> StatsBackend:
    """Select a tool without deciding when the call should run.

    Parameters
    ----------
    config : dict
        Segstats call configuration. Annotation and intensity-input calls need
        FreeSurfer, as does an explicit ``use_mri_segstats`` override.
    default_backend : StatsBackend
        Tool chosen by installation discovery.

    Returns
    -------
    StatsBackend
        Backend required by this call, otherwise the discovered default.
    """
    if "annot" in config or "in_file" in config or config.get("use_mri_segstats", False):
        return "freesurfer"
    return default_backend


def segstats_requires_surfaces(config: dict[str, Any]) -> bool:
    """Determine surface scheduling from inputs, not from the chosen backend.

    Parameters
    ----------
    config : dict
        Segstats call configuration.

    Returns
    -------
    bool
        Whether the call uses annotations or a file in the subject's surf directory.
    """
    return "annot" in config or ("in_file" in config and Path(config["in_file"]).parent.name == "surf")
