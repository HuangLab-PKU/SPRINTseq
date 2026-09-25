"""The downstream moved to spatial-cells; what sprintseq still owes the scripts written before.

Three things have to hold. sprintseq must import and run its own stages with spatial-cells
absent -- it is not a dependency. With spatial-cells installed, the old entry points
(`sprintseq cell-map ...`, `from sprintseq.segment.utils import ...`) must reach the same
objects and say where they went. Without it, they must fail with the install line rather
than a bare ModuleNotFoundError.
"""
import subprocess
import sys
import textwrap

import pytest

from sprintseq import _moved


def _block_spatial_cells(monkeypatch):
    for name in [m for m in sys.modules if m == "spatial_cells" or m.startswith("spatial_cells.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "spatial_cells", None)


def test_sprintseq_imports_without_spatial_cells():
    code = textwrap.dedent("""
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "spatial_cells" or name.startswith("spatial_cells."):
                    raise ModuleNotFoundError(f"blocked {name}", name=name)
        sys.meta_path.insert(0, Block())
        import sprintseq.cli.main, sprintseq.readout, sprintseq.gene_calling, sprintseq.qc
        assert not any(m.startswith("spatial_cells") for m in sys.modules)
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_moved_subcommand_is_forwarded_with_its_flags(monkeypatch):
    cli = pytest.importorskip("spatial_cells.cli.main")
    from sprintseq.cli import main as main_mod

    seen = {}
    monkeypatch.setattr(cli, "main", lambda argv: seen.setdefault("argv", argv))
    with pytest.warns(FutureWarning, match="spatial-cells cell-map"):
        main_mod.main(["cell-map", "--run-id", "R1", "--all", "-Q", "30"])
    assert seen["argv"] == ["cell-map", "--run-id", "R1", "--all", "-Q", "30"]


@pytest.mark.parametrize("old_module, name, new_module", [
    ("sprintseq.cli.density", "generate_density_maps", "spatial_cells.maps.density"),
    ("sprintseq.cli.density", "parse_gene_name", "spatial_cells.genes"),
    ("sprintseq.cli.density_stack", "build_density_stack", "spatial_cells.maps.density_stack"),
    ("sprintseq.cli.density_stack", "thermal_colormap", "spatial_cells.maps.density_stack"),
    ("sprintseq.segment.utils", "assign_spots_to_cells", "spatial_cells.segment.utils"),
    ("sprintseq.segment", "auto_detect_dapi", "spatial_cells.segment"),
    ("sprintseq.segment.cellmap", "build_cell_maps", "spatial_cells.maps.cellmap"),
    ("sprintseq.readout.spot_geojson", "write_multipoint_geojson", "spatial_cells.maps.spot_geojson"),
    ("sprintseq.qc", "generate_density_qc", "spatial_cells.qc"),
])
def test_old_import_reaches_the_moved_object(old_module, name, new_module):
    import importlib
    pytest.importorskip("spatial_cells")
    old = importlib.import_module(old_module)
    with pytest.warns(FutureWarning, match=new_module.replace(".", r"\.")):
        obj = getattr(old, name)
    assert obj is getattr(importlib.import_module(new_module), name)


def test_unknown_name_is_still_an_attribute_error():
    pytest.importorskip("spatial_cells")
    import sprintseq.cli.density as old
    with pytest.raises(AttributeError, match="moved to"):
        old.no_such_function


def test_without_spatial_cells_the_error_says_how_to_install(monkeypatch):
    _block_spatial_cells(monkeypatch)
    import sprintseq.segment.utils as old
    with pytest.raises(ModuleNotFoundError, match="spatial-cells package"):
        old.assign_spots_to_cells
    from sprintseq.cli import main as main_mod
    with pytest.raises(ModuleNotFoundError, match="spatial-cells package"):
        main_mod.main(["density", "--run-id", "R1"])


def test_moved_commands_are_exactly_the_spatial_cells_ones():
    cli = pytest.importorskip("spatial_cells.cli.main")
    assert set(_moved.MOVED_COMMANDS) == set(cli.SUBCOMMANDS)
