"""Fetch a run's decoding codebook from probe-bank and snapshot it into the run.

The codebook is not written by hand per run any more: probe-bank derives it from
the pools the run records (marker tube + the sample's TCR / allele tubes, with
their design versions), labels each codeword the way the analysis names things,
and refuses to hand out an ambiguous table. This command asks for it and keeps a
copy beside the data -- the copy is the debug record of what this run was
decoded with, even after the bank moves on.

    sprintseq codebook --run-id <RUN_ID>
        -> <RUN_ID>_processed/codebook/codebook.csv    (No., Gene, Barcode)
           <RUN_ID>_processed/codebook/codebook.json   (everything the bank said)
    sprintseq gene-calling --run-id <RUN_ID> --ref-file <...>/codebook/codebook.csv

The bank URL comes from ``--bank``, else ``$SPRINTSEQ_PROBE_BANK``, else the lab
server. An existing snapshot that differs is never overwritten without
``--force``: a run decoded with one table must not silently change under it.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DEST_DIRECTORY = r'\\10.10.10.1\NAS Processed Images'
DEFAULT_BANK = "http://10.10.10.1:8001"
DARK_BASE = "G"      # SP369 on cy3/cy5: G is dark in both channels


class CodebookError(RuntimeError):
    """The bank would not give a decodable codebook for this run."""


def bank_url(explicit: Optional[str] = None) -> str:
    return (explicit or os.environ.get("SPRINTSEQ_PROBE_BANK") or DEFAULT_BANK).rstrip("/")


def _get(url: str, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        if exc.code == 404:
            raise CodebookError(f"probe-bank has no run record for this RUN_ID ({url}). "
                                "Record the run and its pools in the ledger first.") from None
        raise CodebookError(f"probe-bank answered {exc.code} for {url}: {body}") from None


def fetch_codebook(run_id: str, *, bank: Optional[str] = None, min_hamming: int = 3,
                   timeout: float = 60.0) -> dict:
    """The bank's JSON answer for ``run_id``; raises CodebookError unless decodable."""
    base = f"{bank_url(bank)}/api/bank/model/runs/{urllib.parse.quote(run_id)}/codebook"
    query = urllib.parse.urlencode({"min_hamming": min_hamming})
    data = json.loads(_get(f"{base}?{query}", timeout))
    data["_fetched"] = {"url": f"{base}?{query}",
                        "at": dt.datetime.now().isoformat(timespec="seconds")}
    if not data.get("decodable"):
        raise CodebookError("probe-bank says this run's codebook is not decodable:\n  "
                            + "\n  ".join(data.get("errors") or ["(no reason given)"]))
    return data


def codebook_csv(data: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["No.", "Gene", "Barcode"])
    for c in data["codewords"]:
        w.writerow([c["no"], c["gene"], c["barcode"]])
    return buf.getvalue()


def dark_prefix_report(barcodes, max_cycles: int = 5) -> dict:
    """Codewords dark in every one of the first k cycles, for k = 1..max_cycles.

    A spot is only found in a detection cycle where it lights up, so a codeword
    whose first k bases are all dark is invisible to detection on cycles 1..k.
    """
    return {k: sum(all(b == DARK_BASE for b in bc[:k]) for bc in barcodes)
            for k in range(1, max_cycles + 1)}


def write_snapshot(data: dict, out_dir: Path, *, force: bool = False) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / "codebook.csv", out_dir / "codebook.json"
    text = codebook_csv(data)
    if csv_path.exists() and csv_path.read_text(encoding="utf-8") != text and not force:
        raise CodebookError(f"{csv_path} exists and differs from what probe-bank returns now; "
                            "it is the record of a past decode. Pass --force to replace it, "
                            "or --out-dir to write elsewhere.")
    csv_path.write_text(text, encoding="utf-8", newline="")
    json_path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_path


def run(run_id: str, *, bank: Optional[str] = None, out_dir: Optional[str] = None,
        min_hamming: int = 3, force: bool = False) -> Path:
    data = fetch_codebook(run_id, bank=bank, min_hamming=min_hamming)
    dest = Path(out_dir) if out_dir else Path(BASE_DEST_DIRECTORY) / f"{run_id}_processed" / "codebook"
    logger.info("ledger %s", data.get("ledger_rev") or "(commit unknown)")
    for p in data.get("pools", []):
        logger.info("pool   %s", p)
    for d in data.get("decoys", []):
        logger.info("decoys unmixed members of %s", d)
    logger.info("codewords %d %s, min Hamming %s", data["n_codewords"], data["layers"],
                data["min_hamming"])
    for w in data.get("warnings", []):
        logger.warning("%s", w)
    dark = dark_prefix_report([c["barcode"] for c in data["codewords"]])
    logger.info("all-dark codewords if detection uses cycles 1..k: %s",
                ", ".join(f"1-{k}: {n}" if k > 1 else f"1: {n}" for k, n in dark.items()))
    path = write_snapshot(data, dest, force=force)
    logger.info("wrote %s (+ codebook.json)", path)
    return path
