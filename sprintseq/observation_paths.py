"""Stable published-result pointers outside disposable observation work directories."""
from pathlib import Path



def repair_directory(run_dir):
    run = Path(run_dir)
    current = run / "archive" / "observation_repair"
    legacy = run / "observation_repair"
    if current.exists() and legacy.exists():
        raise ValueError("Ambiguous observation repair workspace")
    return legacy if legacy.is_dir() else current


def _marker(run_dir, root_name, legacy_name, *, for_write=False):
    run = Path(run_dir)
    primary = run / root_name
    if primary.is_symlink():
        raise ValueError("Published-result marker must be a regular file: " + str(primary))
    if for_write or primary.exists():
        return primary
    legacy = repair_directory(run) / legacy_name
    return legacy if legacy.exists() else primary


def activation_path(run_dir, *, for_write=False):
    return _marker(run_dir, "observation_default_activation.json", "default_activation.json", for_write=for_write)


def current_path(run_dir, *, for_write=False):
    return _marker(run_dir, "observation_current.json", "current.json", for_write=for_write)


def pointer_history_path(run_dir):
    return Path(run_dir) / "archive" / "result_history" / "observation_pointer_history"
