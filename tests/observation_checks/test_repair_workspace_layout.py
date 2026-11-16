import pytest

from sprintseq.observation_paths import repair_directory as sprintseq_repair
from spatial_cells.observation_paths import repair_directory as cells_repair


@pytest.mark.parametrize("resolve", [sprintseq_repair, cells_repair])
def test_repair_workspace_layout(tmp_path, resolve):
    archived = tmp_path / "archive" / "observation_repair"
    legacy = tmp_path / "observation_repair"
    assert resolve(tmp_path) == archived
    assert not legacy.exists()
    archived.mkdir(parents=True)
    assert resolve(tmp_path) == archived
    archived.rmdir()
    legacy.mkdir()
    assert resolve(tmp_path) == legacy
    archived.mkdir()
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve(tmp_path)
