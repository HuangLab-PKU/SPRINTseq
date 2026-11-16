"""Locate run-level acquisition and decode check records."""
from pathlib import Path


def checks_directory(run):
    """Use logs for new runs and preserve existing legacy receipt chains."""
    run = Path(run)
    primary = run / "logs" / "observation_checks"
    legacy = run / "observation_checks"
    if primary.exists() and legacy.exists():
        raise ValueError("Two observation-check directories exist for " + str(run))
    return legacy if legacy.is_dir() else primary
