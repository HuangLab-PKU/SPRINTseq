"""Pipeline stage: immutable source-bound molecular repair calls."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from .history_observability import ObservableHistoryModel as HistoryModel
from .pause_mechanisms import infer_qualified_pause

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def run(run_dir,qc_dir,model_dir,output,*,max_events=2,chunk_size=128,mechanism_dir=None,pause_mechanism_dir=None):
    p=Path(run_dir);qc=Path(qc_dir);md=Path(model_dir);out=Path(output)
    if out.exists():raise ValueError('Existing immutable repair generation')
    frozen={str(f):digest(f) for f in [Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('pause_mechanisms.py'),Path(__file__).with_name('history_observability.py')]}
    qr=json.loads((qc/'receipt.json').read_text());mr=json.loads((md/'receipt.json').read_text());rid=p.name.removesuffix('_processed');assert qr['run_id']==mr['run_id']==rid
    source=p/'readout/transcripts.parquet';intensity=p/'readout/transcript_intensities.parquet';panel=p/'codebook/codebook.csv';image=p/'stitched/mosaic.ome.tif'
    if digest(source)!=qr['source_transcripts_sha256'] or [image.stat().st_size,image.stat().st_mtime_ns]!=qr['source_tiff_identity']:raise ValueError('QC source changed')
    if digest(intensity)!=mr['input_parquet_sha256'] or digest(panel)!=mr['panel_sha256'] or digest(md/'model.npz')!=mr['model_sha256']:raise ValueError('Model source changed')
    if digest(qc/'point_validity.npz')!=qr['point_validity_sha256']:raise ValueError('Validity changed')
    v=np.load(qc/'point_validity.npz',allow_pickle=False);observed=v['observed'].reshape(len(v['transcript_ids']),-1);reason=v['reason'];ids=v['transcript_ids'];affected=(~observed).any(axis=1);selected=np.flatnonzero(affected);control=np.flatnonzero(~affected);control=control[np.linspace(0,len(control)-1,min(4096,len(control)),dtype=int)]
    if len(control)<100:raise ValueError('Insufficient unaffected controls')
    table=pq.read_table(intensity).to_pandas().set_index('transcript_id');cols=mr['intensity_order'];assert cols==[f'r{t:02d}_{c}' for t in range(1,qr['cycles']+1) for c in qr['channels']];assert table.index.is_unique and set(ids)==set(table.index)
    raw=table.loc[ids[selected],cols].to_numpy(np.float32);controlraw=table.loc[ids[control],cols].to_numpy(np.float32);del table
    model=np.load(md/'model.npz',allow_pickle=False);hm=HistoryModel(model,mr['gene_classes']);K=mr['gene_classes'];genes=pd.read_csv(panel).Gene.astype(str).tolist();enc={'A':[1,1],'T':[1,0],'C':[0,1],'G':[0,0]};codes=np.array([[b for ch in str(c) for b in enc[ch]] for c in pd.read_csv(panel).Barcode]);assert np.array_equal(codes,model['all_codes'][:K])
    def normalize(x,mask):
        safe=np.where(mask,x+model['log_add'],1.)
        if np.any(mask&((safe<=0)|~np.isfinite(safe))):raise ValueError('Unmarked nonfinite intensity')
        y=(np.log10(safe)-model['data_log_mean'])/model['data_log_std'];y[~mask]=np.nan;return y
    full=np.ones_like(controlraw,bool);cx=normalize(controlraw,full);controlidx=[];controlprob=[]
    for i in range(0,len(cx),chunk_size):
        r=hm.infer(cx[i:i+chunk_size],full[0],[]);controlidx.extend(r['index']);controlprob.extend(r['probability'])
    cb=pq.read_table(source,columns=['transcript_id','feature_name','probability','is_gene'],filters=[('transcript_id','in',ids[control].tolist())]).to_pandas().set_index('transcript_id').loc[ids[control]];original=cb.feature_name.astype(object).map({g:i for i,g in enumerate(genes)}).fillna(-1).to_numpy();trusted=cb.is_gene.to_numpy(bool)&(cb.probability.to_numpy()>.99);agreement=float((np.asarray(controlidx)[trusted]==original[trusted]).mean())
    if trusted.sum()<100 or agreement<.99:raise ValueError('Full-observation control replay fails qualification')
    out.mkdir(parents=True);state={'status':'RUNNING','run_id':rid,'affected_rows':len(selected),'completed_rows':0,'full_control_agreement':agreement};(out/'state.json').write_text(json.dumps(state,indent=2))
    obs=observed[selected];rr=reason[selected];eligible=((~obs.reshape(len(obs),observed.shape[1]//2,2)).all(axis=2)&((rr&3)>0).all(axis=2)&~((rr&4)>0).any(axis=2))
    mechanism_record=None
    if mechanism_dir is not None:
        mechanism_dir=Path(mechanism_dir);mechanism_record=json.loads((mechanism_dir/'receipt.json').read_text())
        if mechanism_record['run_id']!=rid or mechanism_record['status']!='REGION_MECHANISM_POLICY_QUALIFIED' or mechanism_record['qc_receipt_sha256']!=digest(qc/'receipt.json') or mechanism_record['model_receipt_sha256']!=digest(md/'receipt.json'):raise ValueError('Mechanism source mismatch')
        if digest(mechanism_dir/'mechanism_policy.npz')!=mechanism_record['policy_sha256']:raise ValueError('Mechanism policy changed')
        policy=np.load(mechanism_dir/'mechanism_policy.npz',allow_pickle=False)
        if not np.array_equal(policy['transcript_ids'],ids) or policy['no_pause'].dtype!=bool or policy['no_pause'].shape!=observed.reshape(len(ids),-1,2).shape[:2]:raise ValueError('Mechanism IDs or dimensions differ')
        no_pause=policy['no_pause'][selected]
        if np.any(no_pause&~eligible):raise ValueError('Mechanism policy exceeds eligible image-failure observations')
        eligible &= ~no_pause
        for f,h in mechanism_record['source_code'].items():
            if digest(f)!=h:raise ValueError('Mechanism code changed')
    if pause_mechanism_dir is None:raise ValueError('Qualified pause policy required')
    pdirectory=Path(pause_mechanism_dir);pr=json.loads((pdirectory/'receipt.json').read_text());pp=np.load(pdirectory/'pause_policy.npz')
    if pr['status']!='FIRST_PAUSE_REGION_POLICY_QUALIFIED' or pr['run_id']!=rid or pr['qc_receipt_sha256']!=digest(qc/'receipt.json') or pr['model_receipt_sha256']!=digest(md/'receipt.json') or pr['policy_sha256']!=digest(pdirectory/'pause_policy.npz'):raise ValueError('Pause policy source mismatch')
    if not np.array_equal(pp['transcript_ids'],ids) or pp['first_pause'].dtype!=bool or pp['first_pause'].shape!=(len(ids),):raise ValueError('Pause policy ID mismatch')
    for f,h in pr['source_code'].items():
        if digest(f)!=h:raise ValueError('Pause-policy source code changed')
    forced=pp['first_pause'][selected]
    if np.any(forced&(~eligible[:,0]|~obs[:,2:].all(axis=1))) or np.any(pp['first_pause'][~affected]):raise ValueError('Pause policy exceeds eligible scope')
    veto_count=0
    signatures=np.concatenate([obs,eligible,forced[:,None]],axis=1);patterns,inverse=np.unique(signatures,axis=0,return_inverse=True);N=len(selected);indices=np.full(N,-1);prob=np.zeros(N);prob2=np.zeros(N);entropy=np.zeros(N);hprob=np.zeros(N);hbits=np.zeros(N,np.uint16);ambiguous=np.ones(N,bool);summaries=[];done=0;values=normalize(raw,obs)
    for number,pattern in enumerate(patterns):
        rows=np.flatnonzero(inverse==number);mask=pattern[:obs.shape[1]];cycles=np.flatnonzero(pattern[obs.shape[1]:-1]).tolist();histories=[]
        for start in range(0,len(rows),chunk_size):
            pick=rows[start:start+chunk_size];result=infer_qualified_pause(hm,values[pick],mask) if pattern[-1] else hm.infer(values[pick],mask,cycles,max_events=max_events);veto_count+=int(result.get('veto',np.zeros(len(pick),bool)).sum());indices[pick]=result['index'];prob[pick]=result['probability'];prob2[pick]=result['probability_2'];entropy[pick]=result['entropy'];hprob[pick]=result['history_probability'];ambiguous[pick]=result['ambiguous'];histories=result['histories']
            if histories:
                bits=np.array([sum(1<<c for c in h) for h in histories],np.uint16);hbits[pick]=bits[result['history_index']]
            done+=len(pick)
        summaries.append({'pattern':number,'rows':len(rows),'observed_dimensions':int(mask.sum()),'eligible_pause_cycles':[c+1 for c in cycles],'histories':[[c+1 for c in h] for h in histories],'unknown_rows':int((indices[rows]<0).sum())})
        state.update(completed_rows=done,patterns_completed=number+1);(out/'state.json').write_text(json.dumps(state,indent=2));print(f'REPAIRED {done}/{N} PATTERN {number+1}/{len(patterns)}',flush=True)
    if np.any(prob<0)|np.any(prob>1+1e-10)|np.any(prob2>prob+1e-10):raise ValueError('Invalid output confidence ordering')
    if frozen!={f:digest(f) for f in frozen} or digest(source)!=qr['source_transcripts_sha256'] or digest(intensity)!=mr['input_parquet_sha256']:raise ValueError('Repair sources changed')
    np.savez_compressed(out/'repair_calls.npz',transcript_ids=ids[selected],candidate_index=indices,probability=np.clip(prob,0,1),probability_2=np.clip(prob2,0,1),entropy=entropy,history_probability=hprob,history_cycle_bits=hbits,ambiguous=ambiguous,observed=obs,reason=rr)
    report=dict(schema_version='1.0.0',status='REPAIR_CALLS_READY_FOR_DOWNSTREAM_VALIDATION',run_id=rid,affected_rows=N,unknown_rows=int((indices<0).sum()),target_Q20_rows=int(((indices>=0)&(indices<K)&(prob>.99)).sum()),max_pause_events=max_events,history_prior_sensitivity=[.1,.5,.9],probability_semantics='Same-label across all three history priors; probability/second/entropy from the least-confident prior. Exact cross-history projected-code collisions are UNKNOWN.',full_observation_control={'rows':len(control),'trusted_reference_rows':int(trusted.sum()),'agreement':agreement},patterns=summaries,qc_receipt=str(qc/'receipt.json'),qc_receipt_sha256=digest(qc/'receipt.json'),model_receipt=str(md/'receipt.json'),model_receipt_sha256=digest(md/'receipt.json'),repair_calls_sha256=digest(out/'repair_calls.npz'),source_transcripts_sha256=qr['source_transcripts_sha256'],source_intensities_sha256=mr['input_parquet_sha256'],source_code=frozen,scientific_accuracy_accepted=False,limitations=['No biological FDR calibration.','More than max_pause_events image-failed cycles remain UNKNOWN; source-missing cycles never imply chemical pause.','Unobserved final bases remain unobserved.','Image masks have finite resolution and incomplete source coverage on legacy runs.','No new spots invented when detection itself missed every available cycle.'],production_changes=False)
    report['probability_semantics']='Outside qualified pause policy: original prior-sensitive history mixture. Inside: conditional first-pause posterior, within-history collisions unknown, optical BF>100 veto; no universal biological FDR.'
    report.update(pause_policy_receipt=str(pdirectory/'receipt.json'),pause_policy_receipt_sha256=digest(pdirectory/'receipt.json'),qualified_pause_points=int(forced.sum()),individual_optical_veto_points=veto_count)
    if mechanism_record is not None:
        report.update(mechanism_receipt=str(mechanism_dir/'receipt.json'),mechanism_receipt_sha256=digest(mechanism_dir/'receipt.json'),no_pause_supported_points=mechanism_record['no_pause_points'])
    (out/'receipt.json').write_text(json.dumps(report,indent=2));state.update(status=report['status']);(out/'state.json').write_text(json.dumps(state,indent=2));print(json.dumps({k:report[k] for k in ['status','run_id','affected_rows','unknown_rows','target_Q20_rows']}),flush=True);return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--qc-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mechanism-dir',type=Path);p.add_argument('--pause-mechanism-dir',type=Path,required=True);p.add_argument('--max-events',type=int,default=2);p.add_argument('--chunk-size',type=int,default=128);a=p.parse_args();run(a.run_dir,a.qc_dir,a.model_dir,a.output,max_events=a.max_events,chunk_size=a.chunk_size,mechanism_dir=a.mechanism_dir,pause_mechanism_dir=a.pause_mechanism_dir)
if __name__=='__main__':main()
