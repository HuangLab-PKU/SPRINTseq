"""Export an unmodified installed PoSTcode fit before its full E-step.

A process-local profiler captures the actual first E-step call and normalization
locals, verifies full-observation parity, then stops this new candidate fit.
It does not rerun readout or modify the installed mapping function.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from .model import log_likelihood
from scipy.special import logsumexp

def marginal_gaussian_posterior(values, observed, means, covariance, priors):
    if not np.asarray(observed).all():
        raise ValueError('Export parity requires complete observations')
    joint=log_likelihood(values,means,covariance,np.ones(values.shape[1],bool))+np.log(priors)[None]
    return {'posterior':np.exp(joint-logsumexp(joint,axis=1)[:,None])}

class ExportComplete(Exception):pass

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--qc-dir',type=Path);ap.add_argument('--cycles',type=int,default=10);ap.add_argument('--iterations',type=int,default=60);a=ap.parse_args()
 if a.output.exists():raise ValueError('Candidate model output exists')
 import sprintseq.gene_calling.gene_mapping as gm
 import postcode.decoding_functions as backend
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 sources={str(Path(m.__file__).resolve()):sha(m.__file__) for m in [gm,backend]};datafile=a.run_dir/'readout/transcript_intensities.parquet';panel=a.run_dir/'codebook/codebook.csv';identity=[datafile.stat().st_size,datafile.stat().st_mtime_ns];datahash=sha(datafile);panelhash=sha(panel)
 df=pd.read_parquet(datafile);cols=[f'r{c:02d}_{ch}' for c in range(1,a.cycles+1) for ch in ['cy3','cy5']];expected=['transcript_id',*cols]
 if set(df.columns)!=set(expected) or not df.transcript_id.is_unique:raise ValueError('Intensity schema/identities differ')
 training_coverage=None
 if a.qc_dir is not None:
  qreceipt=a.qc_dir/'receipt.json';qr=json.loads(qreceipt.read_text());validity=a.qc_dir/'point_validity.npz'
  if qr['source_transcripts_sha256']!=sha(a.run_dir/'readout/transcripts.parquet') or sha(validity)!=qr['point_validity_sha256']:raise ValueError('Training validity source changed')
  valid=np.load(validity);good=np.asarray(valid['observed']).all(axis=(1,2));selected=valid['transcript_ids'][good]
  if len(selected)<100:raise ValueError('Insufficient completely observed training points')
  df=df[df.transcript_id.isin(selected)].copy()
  if len(df)!=len(selected) or not np.isfinite(df[cols].to_numpy()).all():raise ValueError('Training IDs/nonfinite observed values differ')
  training_coverage={'qc_receipt':str(qreceipt),'qc_receipt_sha256':sha(qreceipt),'validity_sha256':sha(validity),'fully_observed_training_candidates':len(df),'training_ids_sha256':hashlib.sha256(np.sort(df.transcript_id.to_numpy()).tobytes()).hexdigest(),'invalid_dimensions_used_for_training':False}
 df=df.rename(columns={'transcript_id':'index',**{f'r{c:02d}_{ch}':f'cyc_{c}_{ch}' for c in range(1,a.cycles+1) for ch in ['cy3','cy5']}})
 captured={}
 def callback(frame,event,arg):
  if event!='call' or frame.f_code is not gm._postcode_e_step_stable.__code__:return
  sys.setprofile(None);local=frame.f_locals;parent=frame.f_back.f_locals
  def cpu(x):return x.detach().cpu().numpy()
  model={'means':cpu(local['theta']),'covariance':cpu(local['sigma']),'priors':cpu(local['w']),'log_add':cpu(parent['log_add']),'data_log_mean':cpu(parent['data_log_mean']),'data_log_std':cpu(parent['data_log_std']),'all_codes':cpu(parent['all_codes']),'codes_transform':cpu(parent['codes_tr_v_star']),'codes_constants':cpu(parent['codes_tr_consts_v_star'])}
  count=min(64,len(local['data']));normalized=cpu(local['data'][:count]);torch_p=cpu(gm._postcode_e_step_stable(local['data'][:count],local['w'],local['theta'],local['sigma'],count,local['n_classes']))
  raw_sample=cpu(parent['data'][:count]);roundtrip=(np.log10(raw_sample+model['log_add'])-model['data_log_mean'])/model['data_log_std'];normalization_error=float(np.max(np.abs(roundtrip-normalized)))
  if normalization_error>5e-5:raise ValueError('Raw normalization roundtrip failed '+str(normalization_error))
  result=marginal_gaussian_posterior(roundtrip,np.ones_like(normalized,dtype=bool),model['means'],model['covariance'],model['priors']);difference=float(np.max(np.abs(result['posterior']-torch_p)))
  if difference>5e-5:raise ValueError('Full-observation model parity failed '+str(difference))
  K=int(parent['K']);groups=np.full(len(model['means']),K+1,dtype=np.int64);groups[:K]=np.arange(K);groups[np.atleast_1d(parent['bkg_ind']).astype(int)]=K;model['class_groups']=groups
  captured.update(model=model,parity={'rows':count,'maximum_absolute_probability_difference':difference,'tolerance':5e-5,'raw_normalization_maximum_error':normalization_error,'raw_normalization_tolerance':5e-5},fit_rows=len(df),training_rows=len(parent['ind_keep']),gene_classes=K,inference_classes=len(groups))
  raise ExportComplete()
 if sys.getprofile() is not None:raise ValueError('Existing process profiler would be replaced')
 try:
  sys.setprofile(callback)
  gm.postcode_mapping(df,str(panel),cyc_num=a.cycles,channels=['cy3','cy5'],num_iter=a.iterations,verbose=True)
 except ExportComplete:pass
 finally:sys.setprofile(None)
 if not captured:raise RuntimeError('Installed model never reached export boundary')
 if sources!={str(Path(m.__file__).resolve()):sha(m.__file__) for m in [gm,backend]} or panelhash!=sha(panel) or identity!=[datafile.stat().st_size,datafile.stat().st_mtime_ns] or datahash!=sha(datafile):raise ValueError('Source changed while fitting')
 a.output.mkdir(parents=True);np.savez_compressed(a.output/'model.npz',**captured.pop('model'));receipt={'status':'NEW_FIT_EXPORTED_FULL_OBSERVATION_PARITY_PASS','run_id':a.run_dir.name.removesuffix('_processed'),'input_parquet_sha256':datahash,'panel_sha256':panelhash,'source_modules':sources,'iterations':a.iterations,'seed':1,'intensity_order':cols,'normalization':'(log10(raw+log_add)-data_log_mean)/data_log_std','normalization_id':'sha256:'+hashlib.sha256(b''.join(np.load(a.output/'model.npz')[key].tobytes() for key in ['log_add','data_log_mean','data_log_std'])).hexdigest(),'model_sha256':sha(a.output/'model.npz'),'model_generation':'New deterministic-recipe refit; original production model was not saved. Baseline label agreement and held-out missingness qualification remain required.','production_changes':False,'training_coverage':training_coverage,**captured};(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:receipt[k] for k in ['status','fit_rows','training_rows','inference_classes','parity']}))
if __name__=='__main__':main()
