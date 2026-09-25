import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sprintseq.qc.report import (
    generate_gene_calling_qc,
    generate_readout_qc,
)


class TestGenerateReadoutQC:
    def test_creates_files(self, synthetic_intensity_df, synthetic_position_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_readout_qc(
                intensity_df=synthetic_intensity_df,
                position_df=synthetic_position_df,
                output_dir=tmpdir,
                run_id="test_run",
                channels=["cy3", "cy5"],
                seq_cycle=4,
                detection_stats={"per_channel": {"cy3": 300, "cy5": 250}, "total_raw": 550, "total_after_exact_dedup": 500},
                filter_stats={"n_before": 500, "n_after": 480, "threshold": 50},
                dedup_stats={"n_before": 480, "n_after": 450},
            )
            assert (Path(tmpdir) / "readout_qc.json").exists()
            assert (Path(tmpdir) / "readout_qc.png").exists()

    def test_json_schema(self, synthetic_intensity_df, synthetic_position_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_readout_qc(
                intensity_df=synthetic_intensity_df,
                position_df=synthetic_position_df,
                output_dir=tmpdir,
                run_id="test_run",
                channels=["cy3", "cy5"],
                seq_cycle=4,
                detection_stats={"per_channel": {"cy3": 300, "cy5": 250}, "total_raw": 550, "total_after_exact_dedup": 500},
                filter_stats={"n_before": 500, "n_after": 480, "threshold": 50},
                dedup_stats={"n_before": 480, "n_after": 450},
            )
            with open(Path(tmpdir) / "readout_qc.json") as f:
                data = json.load(f)
            for key in ("run_id", "timestamp", "detection", "filtering", "deduplication", "intensity_summary", "final_spot_count"):
                assert key in data, f"Missing key: {key}"


class TestGenerateGeneCallingQC:
    def test_creates_files(self, synthetic_mapping_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_gene_calling_qc(
                result_df=synthetic_mapping_result,
                output_dir=tmpdir,
                run_id="test_run",
            )
            assert (Path(tmpdir) / "gene_calling_qc.json").exists()
            assert (Path(tmpdir) / "gene_calling_qc.png").exists()

    def test_json_has_qscores(self, synthetic_mapping_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_gene_calling_qc(
                result_df=synthetic_mapping_result,
                output_dir=tmpdir,
                run_id="test_run",
            )
            with open(Path(tmpdir) / "gene_calling_qc.json") as f:
                data = json.load(f)
            assert "q_score" in data
            assert "fraction_q20" in data["q_score"]
            assert "fraction_q30" in data["q_score"]

    def test_json_has_alerts(self, synthetic_mapping_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_gene_calling_qc(
                result_df=synthetic_mapping_result,
                output_dir=tmpdir,
                run_id="test_run",
            )
            with open(Path(tmpdir) / "gene_calling_qc.json") as f:
                data = json.load(f)
            assert "alert_flags" in data

    def test_json_has_convergence(self, synthetic_mapping_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_gene_calling_qc(
                result_df=synthetic_mapping_result,
                output_dir=tmpdir,
                run_id="test_run",
                diagnostics={"losses": [100.0, 80.0, 60.0, 50.0, 45.0]},
            )
            with open(Path(tmpdir) / "gene_calling_qc.json") as f:
                data = json.load(f)
            assert "convergence" in data


class TestQCHandlesEmpty:
    def test_empty_mapping_df(self):
        df = pd.DataFrame({
            "index": pd.Series(dtype=int),
            "Sequence": pd.Series(dtype=str),
            "Gene": pd.Series(dtype=str),
            "Probability": pd.Series(dtype=float),
            "Probability_2": pd.Series(dtype=float),
            "Entropy": pd.Series(dtype=float),
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_gene_calling_qc(result_df=df, output_dir=tmpdir, run_id="empty")
            with open(Path(tmpdir) / "gene_calling_qc.json") as f:
                data = json.load(f)
            assert data["total_spots"] == 0
