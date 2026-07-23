"""Regenerate QC for a single run to verify plot improvements."""
import sys
sys.path.insert(0, ".")
from scripts.batch_qc_existing import run_qc_for_run
run_qc_for_run("20260430_ZCH_BZ23_mut_1_with_marker")
