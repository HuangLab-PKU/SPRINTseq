import json, pprint

base = r"\\10.10.10.1\NAS Processed Images\20260430_ZCH_BZ23_mut_1_with_marker_processed\readout"

print("=== GENE CALLING QC ===")
with open(f"{base}\\gene_calling_qc.json") as f:
    d = json.load(f)
d.pop("per_gene_stats", None)
d.get("convergence", {}).pop("losses", None)
d.pop("w_star", None)
pprint.pprint(d, width=100)

print("\n=== READOUT QC ===")
with open(f"{base}\\readout_qc.json") as f:
    d2 = json.load(f)
d2.pop("intensity_summary", None)
pprint.pprint(d2, width=100)

print("\n=== DENSITY QC ===")
with open(f"{base}\\density_0.95\\density_qc.json") as f:
    d3 = json.load(f)
d3["per_gene_counts"] = d3["per_gene_counts"][:10]  # top 10 only
pprint.pprint(d3, width=100)
