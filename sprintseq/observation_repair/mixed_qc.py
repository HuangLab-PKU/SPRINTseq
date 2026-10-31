"""Readout QC for explicitly mixed observed dimensions, without zero imputation."""
import json
from pathlib import Path
import numpy as np

def observed_summary(frame, columns):
    result={}
    for column in columns:
        values=frame[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError('Infinite intensity is not declared missingness')
        measured=values[np.isfinite(values)]
        stats={key:None for key in ['mean','std','p5','p25','p50','p75','p95']}
        if len(measured):
            stats.update(mean=float(measured.mean()),std=float(measured.std()))
            stats.update(zip(['p5','p25','p50','p75','p95'],map(float,np.percentile(measured,[5,25,50,75,95]))))
        stats.update(measured_rows=len(measured),missing_rows=len(values)-len(measured))
        result[column]=stats
    return result

def write_mixed_qc(readout,run_id,shape,pixel_size,prepare_receipt):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sprintseq.qc.metrics import compute_spatial_density_grid,compute_pipeline_funnel
    from sprintseq.qc.plots import plot_readout_qc
    out=Path(readout);raw=pd.read_csv(out/'intensity.csv');pos=pd.read_csv(out/'position.csv')
    if not raw['index'].equals(pos['index']):raise ValueError('Readout QC ID mismatch')
    columns=[f'cyc_{t}_{c}' for t in range(1,11) for c in ['cy3','cy5']]
    summary=observed_summary(raw,columns);n=len(pos);funnel=compute_pipeline_funnel(n,n,n)
    report={'run_id':run_id,'final_spot_count':n,'mode':'MIXED_OBSERVED_READOUT_ASSEMBLY','source_prepare_receipt':str(prepare_receipt),'intensity_summary':summary,'funnel':funnel,'funnel_semantics':'Assembly rows only; not a claim about original raw detection or filtering yield.','missing_values_imputed':False}
    (out/'readout_qc.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    grid=compute_spatial_density_grid(pos.Y.to_numpy(),pos.X.to_numpy(),200,shape)
    fig=plot_readout_qc(summary,funnel,grid,['cy3','cy5'],10,pixel_size=pixel_size,bin_size=200)
    fig.suptitle('Mixed-observation readout: observed-only intensity statistics; assembly counts')
    fig.savefig(out/'readout_qc.png',dpi=200,bbox_inches='tight');plt.close(fig)
    return report
