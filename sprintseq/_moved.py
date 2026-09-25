"""Forwarding for the downstream that moved to the spatial-cells package (2026-09-25).

Segmentation, the cell x gene matrix, density maps, density stacks, cell maps and spot
maps are platform-agnostic, so they left this repository for `spatial-cells`
(github.com/HuangLab-PKU/spatial-cells); sprintseq now stops at decoded spots. The last
version that still carries them is tag `v0.1.0-pre-split`.

sprintseq does NOT depend on spatial-cells. The old entry points keep working through
this module only when spatial-cells is installed next to it (as in the lab's `spatial`
env), and say where the code went; otherwise they fail with the install line. They are a
transition aid for per-run scripts and drivers written before the split -- new code
imports `spatial_cells` directly.
"""

import importlib
import warnings

__all__ = ["INSTALL_HINT", "MOVED_COMMANDS", "forward_attr", "import_moved", "run_moved_command"]

INSTALL_HINT = (
    "It moved to the spatial-cells package: "
    "pip install -e <path to spatial-cells> --no-deps --no-build-isolation. "
    "Tag v0.1.0-pre-split of sprintseq still carries the old code."
)

#: sprintseq subcommands that now live in `spatial-cells`, with identical flags.
MOVED_COMMANDS = ("density", "density-stack", "segment", "cell-map", "spot-map")


def import_moved(new_module, old_name):
    """Import ``new_module`` from spatial-cells for the old ``old_name``, or explain."""
    try:
        return importlib.import_module(new_module)
    except ModuleNotFoundError as e:
        if e.name is None or not e.name.startswith("spatial_cells"):
            raise
        raise ModuleNotFoundError(f"{old_name} is no longer part of sprintseq. {INSTALL_HINT}",
                                  name=e.name) from e


def forward_attr(old_module, new_modules, name):
    """PEP 562 ``__getattr__`` body: resolve ``name`` from the first new module that has it."""
    if name.startswith("__"):
        raise AttributeError(name)
    for new in new_modules:
        mod = import_moved(new, old_module)
        if hasattr(mod, name):
            obj = getattr(mod, name)
            home = getattr(obj, "__module__", None) or new      # where it is defined, not re-exported
            warnings.warn(f"{old_module}.{name} moved to {home}.{name} (spatial-cells); import it "
                          "from there.", FutureWarning, stacklevel=3)
            return obj
    raise AttributeError(f"module {old_module!r} has no attribute {name!r} "
                         f"(its code moved to {', '.join(new_modules)})")


def run_moved_command(command, argv):
    """Run ``sprintseq <command> ...`` as ``spatial-cells <command> ...``."""
    cli = import_moved("spatial_cells.cli.main", f"`sprintseq {command}`")
    warnings.warn(f"`sprintseq {command}` moved to `spatial-cells {command}` (same flags); "
                  "call that directly.", FutureWarning, stacklevel=2)
    return cli.main([command, *argv])
