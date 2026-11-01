"""Public command dispatch and argument contract, without executing a real run."""
from pathlib import Path
import pytest
from sprintseq.cli.main import main


def test_repair_command_routes_all_options(monkeypatch, tmp_path):
    import sprintseq.observation_repair.pipeline as pipeline
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    main(["observation-repair", "--run-dir", str(tmp_path), "--output-root", str(tmp_path / "v1"),
          "--model-dir", str(tmp_path / "model"), "--qc-dir", str(tmp_path / "qc"),
          "--fit-device", "auto", "--publish"])
    args, kwargs = calls.pop()
    assert args == (tmp_path, tmp_path / "v1", tmp_path / "model")
    assert kwargs == dict(publish=True, fit_device="auto", qc_dir=tmp_path / "qc", qualify_first_pause=False, localize_active=False)


def test_repair_command_requires_explicit_destination():
    with pytest.raises(SystemExit) as error:
        main(["observation-repair", "--run-dir", "example"])
    assert error.value.code == 2
