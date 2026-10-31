"""Technical-reference erasure and injected-history qualification.

Controls are measured unaffected points, not biological truth. Synthetic pause
uses the actual fitted mean map plus each control's empirical residual.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from .model import acquisition_codes
from .history_observability import ObservableHistoryModel as HistoryModel
from .__main__ import digest

def run(run_dir,qc_dir,model_dir,output):
    p=Path(run_dir);qc=Path(qc_dir);md=Path(model_dir);out=Path(output)
    if out.exists():raise ValueError('Existing qualification')
    qr=json.loads((qc/'receipt.json').read_text());mr=json.loads((md/'receipt.json').read_text());v=np.load(qc/'point_validity.npz',allow_pickle=False);model=np.load(md/'model.npz',allow_pickle=False);hm=HistoryModel(model,mr['gene_classes']);K=mr['gene_classes']
    assert digest(md/'model.npz')==mr['model_sha256'] and digest(qc/'point_validity.npz')==qr['point_validity_sha256']
    tr=pq.read_table(p/'readout/transcripts.parquet',columns=['transcript_id','feature_name','probability','is_gene']).to_pandas().set_index('transcript_id');names=pd.read_csv(p/'codebook/codebook.csv').Gene.astype(str).tolist();good=v['observed'].all(axis=(1,2));ids=v['transcript_ids'][good];sub=tr.loc[ids];sub=sub[sub.is_gene&(sub.probability>.99)];sub=sub.iloc[np.linspace(0,len(sub)-1,min(8192,len(sub)),dtype=int)];assert len(sub)>=100;ids=sub.index.to_numpy();truth=sub.feature_name.astype(object).map({g:i for i,g in enumerate(names)}).to_numpy(int)
    raw=pq.read_table(p/'readout/transcript_intensities.parquet',filters=[('transcript_id','in',ids.tolist())]).to_pandas().set_index('transcript_id').loc[ids,mr['intensity_order']].to_numpy(np.float32);z=(np.log10(raw+model['log_add'])-model['data_log_mean'])/model['data_log_std']
    conditions=[('terminal_optical_loss',list(range(14,20)),[7,8,9],[]),('terminal_source_loss',list(range(14,20)),[],[]),('injected_first_cycle_pause',[0,1],[0],[0]),('source_missing_single_channel',[7],[],[]),('optical_missing_both',[14,15],[],[]),('image_failure_unknown_history',[14,15],[7],[]),('injected_single_pause',[10,11],[5],[5]),('injected_double_pause',[10,11,16,17],[5,8],[5,8])];reports=[]
    for name,missing,eligible,injected in conditions:
        mask=np.ones(z.shape[1],bool);mask[missing]=False;data=z.copy()
        if injected:
            c=acquisition_codes(model['all_codes'],injected,[(0,0)]*len(injected));means=np.column_stack([c,np.ones(len(c))])@hm.coef;data=data-model['means'][truth]+means[truth]
        data[:,~mask]=np.nan;idx=[];pr=[]
        for start in range(0,len(data),64):
            result=hm.infer(data[start:start+64],mask,eligible);idx.extend(result['index']);pr.extend(result['probability'])
        idx=np.asarray(idx);pr=np.asarray(pr);keep=(idx>=0)&(idx<K)&(pr>.99);errors=int((keep&(idx!=truth)).sum());rate=errors/int(keep.sum()) if keep.any() else None
        reports.append(dict(condition=name,reference_rows=len(truth),retained_Q20=int(keep.sum()),retention=float(keep.mean()),retained_disagreements=errors,disagreement_fraction=rate,pass_reference_gate=bool(keep.sum()>=50 and rate<=.01)));print(json.dumps(reports[-1]),flush=True)
    result=dict(status='CONTROL_QUALIFICATION_PASS' if all(r['pass_reference_gate'] for r in reports) else 'CONTROL_QUALIFICATION_HOLD',run_id=qr['run_id'],conditions=reports,criteria='At least50 retained reference calls and <=1% disagreements in each condition; technical-reference qualification only.',limitations=['Q20 full-observation controls are technical labels, not biological truth.','Injected history assumes exported model response/covariance; this tests implementation, not chemical mechanism.','False-negative rate on all-slide image anomalies is not estimated by these controls.'],sources={str(f):digest(f) for f in [qc/'receipt.json',md/'receipt.json',Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('history_observability.py')]})
    out.write_text(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--qc-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.run_dir,a.qc_dir,a.model_dir,a.output)
    if r['status']!='CONTROL_QUALIFICATION_PASS':raise SystemExit(2)
if __name__=='__main__':main()
