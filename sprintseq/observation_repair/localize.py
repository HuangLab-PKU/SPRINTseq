"""Emit only qualified pause updates when extending an already activated revision.

Numerically equivalent recomputation outside that policy is checked, then omitted
so the current rows remain bit-for-bit unchanged. Material differences fail closed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .__main__ import digest


def localize_calls(frame, calls, genes, allowed_ids):
    from spatial_cells.observation_revision import apply_repair
    revised = apply_repair(frame, calls, genes)
    before = frame.set_index("transcript_id"); after = revised.set_index("transcript_id").loc[before.index]
    outside = ~before.index.isin(allowed_ids)
    for col in before.columns:
        a, b = before[col], after[col]
        if pd.api.types.is_numeric_dtype(a.dtype):
            equal = np.isclose(a.to_numpy(dtype=float, na_value=np.nan), b.to_numpy(dtype=float, na_value=np.nan), atol=1e-10, rtol=0, equal_nan=True)
        else:
            av, bv = a.astype(object).to_numpy(), b.astype(object).to_numpy()
            equal = (av == bv) | (pd.isna(av) & pd.isna(bv))
        if not np.all(equal[outside]):
            raise ValueError("Material change outside qualified pause policy: " + col)
    keep = np.isin(calls["transcript_ids"], allowed_ids)
    return {key: np.asarray(calls[key])[keep] for key in calls.files}


def run(run_dir, repair_dir, output):
    run, rd, out = map(Path, [run_dir, repair_dir, output])
    if out.exists():raise ValueError("Immutable localized repair exists")
    receipt = json.loads((rd / "receipt.json").read_text()); source = run / "readout/transcripts.parquet"
    if digest(source) != receipt["source_transcripts_sha256"]:raise ValueError("Active source changed")
    marker = json.loads((run / "observation_repair/default_activation.json").read_text())
    if digest(marker["receipt"]) != marker["receipt_sha256"]:raise ValueError("Activation changed")
    activation = json.loads(Path(marker["receipt"]).read_text()); prior_revision = json.loads((Path(activation["revision"]) / "receipt.json").read_text())
    if prior_revision["transcripts_sha256"] != digest(source):raise ValueError("Default differs from activated revision")
    prior_repair = json.loads(Path(prior_revision["source_repair_receipt"]).read_text())
    previous_qc = json.loads(Path(prior_repair["qc_receipt"]).read_text()); current_qc = json.loads(Path(receipt["qc_receipt"]).read_text())
    if previous_qc["point_validity_sha256"] != current_qc["point_validity_sha256"] or prior_repair["model_receipt_sha256"] != receipt["model_receipt_sha256"]:
        raise ValueError("Cannot localize across changed image validity or fitted model")
    policy_receipt = Path(receipt["pause_policy_receipt"])
    if digest(policy_receipt) != receipt["pause_policy_receipt_sha256"]:raise ValueError("Pause policy receipt changed")
    pr = json.loads(policy_receipt.read_text()); pp = policy_receipt.parent / "pause_policy.npz"
    if digest(pp) != pr["policy_sha256"]:raise ValueError("Pause policy changed")
    policy = np.load(pp); allowed = policy["transcript_ids"][policy["first_pause"]]
    calls = np.load(rd / "repair_calls.npz")
    if digest(rd / "repair_calls.npz") != receipt["repair_calls_sha256"]:raise ValueError("Repair calls changed")
    for f,h in receipt["source_code"].items():
        if digest(f)!=h:raise ValueError("Repair source code changed")
    genes = pd.read_csv(run / "codebook/codebook.csv").Gene.astype(str).tolist()
    scoped = localize_calls(pq.read_table(source).to_pandas(), calls, genes, allowed)
    if digest(source)!=receipt["source_transcripts_sha256"]:raise ValueError("Source changed during localization")
    out.mkdir(parents=True);np.savez_compressed(out / "repair_calls.npz", **scoped)
    receipt.update(affected_rows=len(scoped["transcript_ids"]), unknown_rows=int((scoped["candidate_index"]<0).sum()),
                   target_Q20_rows=int(((scoped["candidate_index"]>=0)&(scoped["candidate_index"]<len(genes))&(scoped["probability"]>.99)).sum()),
                   repair_calls_sha256=digest(out / "repair_calls.npz"), parent_repair_receipt=str(rd / "receipt.json"),
                   parent_repair_receipt_sha256=digest(rd / "receipt.json"), outside_policy_preserved_from_active_baseline=True,
                   localization_semantics="Only qualified first-pause rows emitted. Outside-policy equivalence checked at absolute tolerance 1e-10 before preserving exact baseline rows.")
    receipt["source_code"][str(Path(__file__))]=digest(__file__)
    (out / "receipt.json").write_text(json.dumps(receipt,indent=2));print(json.dumps({k:receipt[k] for k in ["affected_rows","unknown_rows","target_Q20_rows"]}),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["run-dir","repair-dir","output"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.run_dir,a.repair_dir,a.output)

if __name__=="__main__":main()
