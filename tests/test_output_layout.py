from pathlib import Path
import pytest
from sprintseq.observation_paths import repair_directory, pointer_history_path


def test_fresh_run_keeps_intermediates_out_of_root(tmp_path):
    assert repair_directory(tmp_path) == tmp_path / 'archive/observation_repair'
    assert pointer_history_path(tmp_path) == tmp_path / 'archive/result_history/observation_pointer_history'


def test_legacy_repair_and_ambiguous_layout(tmp_path):
    legacy = tmp_path / 'observation_repair'
    legacy.mkdir()
    assert repair_directory(tmp_path) == legacy
    (tmp_path / 'archive/observation_repair').mkdir(parents=True)
    with pytest.raises(ValueError, match='Ambiguous'):
        repair_directory(tmp_path)
