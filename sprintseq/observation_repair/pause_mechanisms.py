"""Qualify a first-acquisition pause from complementary held-out observations.

This deliberately narrow extension does not select by yield or density. Opposite
ID halves, normal controls and local blocks must independently support the pause.
Historical v3 inference remains unchanged outside the qualified policy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import ndimage as ndi
from .__main__ import digest
from .model import HistoryModel, grouped_posterior, log_likelihood


def summarize(delta, keep, ids):
    def one(mask):
        return {"n": int(mask.sum()), "pause_fraction": float((delta[mask] > 0).mean()) if mask.any() else 0.0}
    result = one(keep)
    result["halves"] = [one(keep & (ids % 2 == k)) for k in (0, 1)]
    return result


def accepts_pause(affected, normal, blocks):
    """Both complementary folds and local split-ID evidence are mandatory."""
    if len(affected) != 2 or len(normal) != 2:
        return []
    for a, n in zip(affected, normal):
        if a["n"] < 100 or a["pause_fraction"] < .85 or n["n"] < 100 or n["pause_fraction"] > .15:
            return []
        if any(h["n"] < 30 or h["pause_fraction"] < .80 for h in a["halves"]):
            return []
        if any(h["n"] < 30 or h["pause_fraction"] > .20 for h in n["halves"]):
            return []
    qualified = [b["key"] for b in blocks if b["n"] >= 30 and b["pause_fraction"] >= .85
                 and all(h["n"] >= 10 and h["pause_fraction"] >= .80 for h in b["halves"])]
    return qualified if len(qualified) >= 3 else []


def heldout_scores(hm, values, train, observed):
    """Conditional predictive log densities: joint evidence minus train evidence."""
    evidence, anchors = [], []
    for history in [(), (0,)]:
        means, priors, groups = hm.hypothesis(history)
        scores, eligible = [], []
        collisions = hm.ambiguous_targets(train, [history])
        for start in range(0, len(values), 256):
            x = values[start:start + 256]
            _, joint = grouped_posterior(log_likelihood(x, means, hm.m["covariance"], observed) + np.log(priors), groups)
            posterior, training = grouped_posterior(log_likelihood(x, means, hm.m["covariance"], train) + np.log(priors), groups)
            best = posterior.argmax(axis=1)
            eligible.extend((posterior.max(axis=1) > .99) & (best < hm.K) & ~np.isin(best, list(collisions)))
            scores.extend(joint - training)
        evidence.append(np.asarray(scores)); anchors.append(np.asarray(eligible))
    return evidence[1] - evidence[0], anchors[0] | anchors[1]


def infer_qualified_pause(hm, values, observed):
    """Condition on the qualified region history, retaining per-spot contradiction veto.

Within-history code collisions remain unknown. Strong optical evidence (BF>100)
returns the original mixture instead of forcing a regional assumption on that spot.
"""
    observed = np.asarray(observed, bool)
    if len(observed) != 20 or observed[:2].any() or not observed[2:].all():
        raise ValueError("Qualified pause only supports a missing first cycle and nine observed cycles")
    means, priors, groups = hm.hypothesis((0,))
    post, pause_ev = grouped_posterior(log_likelihood(values, means, hm.m["covariance"], observed) + np.log(priors), groups)
    mu, pr, gr = hm.hypothesis(())
    _, optical_ev = grouped_posterior(log_likelihood(values, mu, hm.m["covariance"], observed) + np.log(pr), gr)
    best = post.argmax(axis=1); ordered = np.sort(post, axis=1)
    ambiguous = np.isin(best, list(hm.ambiguous_targets(observed, [(0,)])))
    result = dict(index=best.copy(), probability=ordered[:, -1].copy(), probability_2=ordered[:, -2].copy(),
                  entropy=-(post * np.log(np.maximum(post, 1e-300))).sum(axis=1),
                  history_index=np.ones(len(values), int), history_probability=np.ones(len(values)),
                  ambiguous=ambiguous, histories=[(), (0,)], status="REGION_QUALIFIED_FIRST_PAUSE")
    for key in ["probability", "probability_2", "entropy"]:
        result[key][ambiguous] = 0
    result["index"][ambiguous] = -1
    veto = optical_ev - pause_ev > np.log(100.)
    if veto.any():
        original = hm.infer(values[veto], observed, [0])
        for key in ["index", "probability", "probability_2", "entropy", "history_index", "history_probability", "ambiguous"]:
            result[key][veto] = original[key]
    result["veto"] = veto
    # History certainty is conditional on an independently qualified region, not a measured base.
    return result


def run(run_dir, qc_dir, model_dir, output):
    p, qc, md, out = map(Path, [run_dir, qc_dir, model_dir, output])
    if out.exists():
        raise ValueError("Existing immutable pause-policy generation")
    files = [Path(__file__), Path(__file__).with_name("model.py")]
    frozen = {str(f): digest(f) for f in files}
    qr = json.loads((qc / "receipt.json").read_text()); mr = json.loads((md / "receipt.json").read_text())
    source = p / "readout/transcripts.parquet"; intensity = p / "readout/transcript_intensities.parquet"
    if qr["cycles"] != 10 or qr["channels"] != ["cy3", "cy5"] or qr["run_id"] != mr["run_id"]:
        raise ValueError("Unsupported pause-qualification scope")
    if digest(source) != qr["source_transcripts_sha256"] or digest(intensity) != mr["input_parquet_sha256"]:
        raise ValueError("Changed observation sources")
    if digest(qc / "point_validity.npz") != qr["point_validity_sha256"] or digest(md / "model.npz") != mr["model_sha256"]:
        raise ValueError("Changed validity/model")
    v = np.load(qc / "point_validity.npz"); ids = v["transcript_ids"]; obs = v["observed"]; reasons = v["reason"]
    masks = np.load(qc / "image_validity.npz"); image = masks["reason"]; H, W = masks["full_shape"]
    if digest(qc / "image_validity.npz") != qr["image_validity_sha256"]:
        raise ValueError("Changed image masks")
    py = np.minimum((v["y_px"] * image.shape[-2] / H).astype(int), image.shape[-2]-1)
    px = np.minimum((v["x_px"] * image.shape[-1] / W).astype(int), image.shape[-1]-1)
    labels, _ = ndi.label((image[0] > 0).all(axis=0)); component = labels[py, px]
    eligible = (~obs[:, 0]).all(axis=1) & obs[:, 1:].all(axis=(1, 2)) & ((reasons[:, 0] & 3) > 0).all(axis=1) & ~((reasons[:, 0] & 4) > 0).any(axis=1)
    normals = np.flatnonzero(obs.all(axis=(1, 2)))
    normals = normals[np.linspace(0, len(normals)-1, min(32768, len(normals)), dtype=int)]
    raw = pq.read_table(intensity).to_pandas().set_index("transcript_id").loc[ids, mr["intensity_order"]].to_numpy(float)
    model = np.load(md / "model.npz"); hm = HistoryModel(model, mr["gene_classes"])
    observed = np.ones(20, bool); observed[:2] = False
    blocks = (v["y_px"] // 1024).astype(int) * 100000 + (v["x_px"] // 1024).astype(int)
    policy = np.zeros(len(ids), bool); records = []
    for comp in np.unique(component[eligible]):
        region = np.flatnonzero(eligible & (component == comp))
        if comp == 0 or len(region) < 1000:
            continue
        joined = np.r_[region, normals]; z = (np.log10(raw[joined] + model["log_add"]) - model["data_log_mean"]) / model["data_log_std"]
        affected, controls, local = [], [], []
        for fold, cycles in enumerate([range(1, 5), range(5, 10)]):
            train = np.zeros(20, bool)
            for t in cycles: train[2*t:2*t+2] = True
            delta, anchor = heldout_scores(hm, z, train, observed)
            ar = np.arange(len(joined)) < len(region)
            affected.append(summarize(delta, anchor & ar, ids[joined]))
            controls.append(summarize(delta, anchor & ~ar, ids[joined]))
            if fold == 1:
                for key in np.unique(blocks[region]):
                    item = summarize(delta, anchor & ar & (blocks[joined] == key), ids[joined]); item["key"] = int(key); local.append(item)
        qualified = accepts_pause(affected, controls, local)
        chosen = region[np.isin(blocks[region], qualified)]; policy[chosen] = True
        record = dict(component=int(comp), cycle=1, region_points=len(region), affected_folds=affected, normal_folds=controls,
                      spatial_blocks=local, qualified_blocks=qualified, applied_points=len(chosen))
        records.append(record); print(json.dumps(record), flush=True)
    if frozen != {f: digest(f) for f in frozen} or digest(source) != qr["source_transcripts_sha256"]:
        raise ValueError("Sources changed during qualification")
    out.mkdir(parents=True); np.savez_compressed(out / "pause_policy.npz", transcript_ids=ids, first_pause=policy)
    report = dict(status="FIRST_PAUSE_REGION_POLICY_QUALIFIED", run_id=qr["run_id"], first_pause_points=int(policy.sum()), regions=records,
                  policy_sha256=digest(out / "pause_policy.npz"), qc_receipt_sha256=digest(qc / "receipt.json"),
                  model_receipt_sha256=digest(md / "receipt.json"), source_code=frozen,
                  criteria="Two complementary heldout folds: >=100 unique Q20 anchors each in affected and normal sets; affected >=85% pause preference, normal <=15%; disjoint ID halves >=30, affected >=80%, normal <=20%. At least 3 local 1024px blocks: >=30 anchors, >=85% pause preference, each half >=10 and >=80%. Individual optical BF>100 veto. First-cycle-only, all later observations valid.",
                  scientific_accuracy_accepted=False, limitations=["Conditional on region history; not universal biological FDR.", "First-acquisition single-pause qualification only; other patterns retain original inference."])
    (out / "receipt.json").write_text(json.dumps(report, indent=2)); return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["run-dir", "qc-dir", "model-dir", "output"]: p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args(); run(a.run_dir, a.qc_dir, a.model_dir, a.output)

if __name__ == "__main__": main()
