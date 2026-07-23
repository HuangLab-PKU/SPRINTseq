import numpy as np
import pandas as pd
from sprintseq.qc.metrics import compute_phred_qscore, MAX_QSCORE

print(f"MAX_QSCORE = {MAX_QSCORE}")

# Check a few known values
test_p = np.array([0.9, 0.99, 0.999, 0.9999, 0.99999, 1.0])
test_q = compute_phred_qscore(test_p)
for p, q in zip(test_p, test_q):
    print(f"  P={p:.6f} -> Q={q:.2f}")

# Check real data distribution
path = r"\\10.10.10.1\NAS Processed Images\20260430_ZCH_BZ23_mut_1_with_marker_processed\readout\mapping_postcode.csv"
df = pd.read_csv(path)
prob = np.nan_to_num(df["Probability"].to_numpy(dtype=np.float64), nan=0.0)
q = compute_phred_qscore(prob)

print(f"\nReal data Q-score distribution:")
print(f"  min={q.min():.1f}, max={q.max():.1f}")
print(f"  Q>=40: {(q >= 40).sum():,} ({(q >= 40).mean()*100:.1f}%)")
print(f"  Q>=45: {(q >= 45).sum():,} ({(q >= 45).mean()*100:.1f}%)")
print(f"  Q==50: {(q == 50).sum():,} ({(q == 50).mean()*100:.1f}%)")
print(f"  Q in [39,41]: {((q >= 39) & (q <= 41)).sum():,}")
print(f"  Q in [49,50]: {((q >= 49) & (q <= 50)).sum():,}")

# Check P distribution near 1.0
print(f"\nProbability near 1.0:")
print(f"  P==1.0: {(prob == 1.0).sum():,}")
print(f"  P>=0.9999: {(prob >= 0.9999).sum():,}")
print(f"  P>=0.99999: {(prob >= 0.99999).sum():,}")

# Histogram of Q values at the high end
bins = np.arange(35, 52)
counts, _ = np.histogram(q, bins=bins)
for b, c in zip(bins[:-1], counts):
    if c > 0:
        print(f"  Q=[{b},{b+1}): {c:,}")
