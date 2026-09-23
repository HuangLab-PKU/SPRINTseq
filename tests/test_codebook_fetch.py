"""sprintseq codebook: fetch a run's codebook from probe-bank and snapshot it.

A throwaway HTTP server stands in for probe-bank; every barcode is synthetic
(this repository is public-facing and must never hold real probe sequences).
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sprintseq.cli import codebook as cb

GOOD = {
    "run_id": "R1", "decodable": True, "pools": ["mk = M@v1:padlock"], "decoys": [],
    "n_codewords": 3, "layers": {"marker": 2, "TCR": 1}, "min_hamming": 3,
    "errors": [], "warnings": [],
    "codewords": [
        {"no": "1", "gene": "EPCAM", "barcode": "GGGCCC", "layer": "marker"},
        {"no": "2", "gene": "CD3E", "barcode": "CCCGGG", "layer": "marker"},
        {"no": "7", "gene": "S1_clonotype4_2", "barcode": "GGTTTC", "layer": "TCR"},
    ],
}
BAD = {**GOOD, "run_id": "R2", "decodable": False, "errors": ["No. 2 is used twice"]}


@pytest.fixture
def bank():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            run = self.path.split("/runs/")[1].split("/")[0]
            body = {"R1": GOOD, "R2": BAD}.get(run)
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body or {"detail": "unknown run"}).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()


def test_snapshot_is_the_decoder_table_plus_the_bank_answer(bank, tmp_path):
    url, seen = bank
    path = cb.run("R1", bank=url, out_dir=tmp_path, min_hamming=3)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "No.,Gene,Barcode", "1,EPCAM,GGGCCC", "2,CD3E,CCCGGG", "7,S1_clonotype4_2,GGTTTC"]
    saved = json.loads((tmp_path / "codebook.json").read_text(encoding="utf-8"))
    assert saved["pools"] == ["mk = M@v1:padlock"]
    assert saved["_fetched"]["url"].endswith("/api/bank/model/runs/R1/codebook?min_hamming=3")
    assert "min_hamming=3" in seen[0]


def test_an_undecodable_codebook_is_refused_and_nothing_is_written(bank, tmp_path):
    with pytest.raises(cb.CodebookError, match="used twice"):
        cb.run("R2", bank=bank[0], out_dir=tmp_path)
    assert not (tmp_path / "codebook.csv").exists()


def test_an_unrecorded_run_says_what_to_do(bank, tmp_path):
    with pytest.raises(cb.CodebookError, match="Record the run"):
        cb.run("nope", bank=bank[0], out_dir=tmp_path)


def test_a_past_snapshot_is_not_replaced_silently(bank, tmp_path):
    (tmp_path / "codebook.csv").write_text("No.,Gene,Barcode\n1,OLD,GGGCCC\n", encoding="utf-8")
    with pytest.raises(cb.CodebookError, match="record of a past decode"):
        cb.run("R1", bank=bank[0], out_dir=tmp_path)
    assert "OLD" in (tmp_path / "codebook.csv").read_text(encoding="utf-8")
    cb.run("R1", bank=bank[0], out_dir=tmp_path, force=True)
    assert "OLD" not in (tmp_path / "codebook.csv").read_text(encoding="utf-8")
    cb.run("R1", bank=bank[0], out_dir=tmp_path)          # identical: fine without --force


def test_bank_url_precedence(monkeypatch):
    monkeypatch.delenv("SPRINTSEQ_PROBE_BANK", raising=False)
    assert cb.bank_url() == cb.DEFAULT_BANK
    monkeypatch.setenv("SPRINTSEQ_PROBE_BANK", "http://x:1/")
    assert cb.bank_url() == "http://x:1"
    assert cb.bank_url("http://y:2") == "http://y:2"


def test_dark_prefix_report():
    assert cb.dark_prefix_report(["GGGCCC", "GCCCCC", "CCCCCC"], max_cycles=3) == {1: 2, 2: 1, 3: 1}
