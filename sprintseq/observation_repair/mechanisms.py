"""Region-level no-history-shift qualification independent of Q20 gains.

Image components propose regions. Identities use prefix observations only;
normal controls calibrate later-cycle templates. Ambiguous regions retain the
original history mixture. No component is forced to a pause history.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from scipy import ndimage as ndi
from .model import HistoryModel
from .__main__ import digest

def decide_no_shift(affected,controls,blocks):
    """Fixed technical gates; positive result permits optical-only inference."""
    if affected['n']<100 or controls['n']<30 or controls['reference_n']<30 or controls['prefix_reference_agreement']<.99:
        return []
    if controls['pause_fraction']>.15 or affected['pause_fraction']>.15:
        return []
    if any(s['n']<30 or s['pause_fraction']>.15 for s in affected['halves']):return []
    qualified=[b['key'] for b in blocks if b['n']>=20 and b['pause_fraction']<=.25 and all(s['n']>=5 and s['pause_fraction']<=.30 for s in b['halves'])]
    return qualified if len(qualified)>=3 else []

def run(run_dir,qc_dir,model_dir,output):
    p=Path(run_dir);qc=Path(qc_dir);md=Path(model_dir);out=Path(output)
    if out.exists():raise ValueError('Immutable mechanism generation exists')
    qr=json.loads((qc/'receipt.json').read_text());mr=json.loads((md/'receipt.json').read_text());rid=p.name.removesuffix('_processed');assert qr['run_id']==mr['run_id']==rid
    source=p/'readout/transcripts.parquet';intensity=p/'readout/transcript_intensities.parquet'
    assert digest(source)==qr['source_transcripts_sha256'] and digest(intensity)==mr['input_parquet_sha256'];assert digest(md/'model.npz')==mr['model_sha256'];assert digest(qc/'point_validity.npz')==qr['point_validity_sha256'];assert digest(qc/'image_validity.npz')==qr['image_validity_sha256']
    frozen={str(f):digest(f) for f in [Path(__file__),Path(__file__).with_name('model.py')]}
    v=np.load(qc/'point_validity.npz',allow_pickle=False);im=np.load(qc/'image_validity.npz',allow_pickle=False);ids=v['transcript_ids'];obs=v['observed'];reasons=v['reason'];N=len(ids);mask=im['reason'];H,W=im['full_shape'];py=np.minimum((v['y_px']*mask.shape[-2]/H).astype(int),mask.shape[-2]-1);px=np.minimum((v['x_px']*mask.shape[-1]/W).astype(int),mask.shape[-1]-1)
    raw=pq.read_table(intensity).to_pandas().set_index('transcript_id').loc[ids,mr['intensity_order']].to_numpy(np.float32);base=pq.read_table(source,columns=['transcript_id','feature_name','is_gene','probability']).to_pandas().set_index('transcript_id').loc[ids];genes=pd.read_csv(p/'codebook/codebook.csv').Gene.astype(str).tolist();labels=base.feature_name.astype(object).map({g:i for i,g in enumerate(genes)}).fillna(-1).to_numpy(int);K=len(genes);model=np.load(md/'model.npz',allow_pickle=False);hm=HistoryModel(model,K);codes=model['all_codes'][:K].reshape(K,10,2)
    normals=np.flatnonzero(obs.all(axis=(1,2))&base.is_gene.to_numpy(bool)&(base.probability.to_numpy()>.99)&(labels>=0));normals=normals[np.linspace(0,len(normals)-1,min(16384,len(normals)),dtype=int)];train=normals[ids[normals]%2==0];control=normals[ids[normals]%2==1];assert len(train)>=100 and len(control)>=100
    no_pause=np.zeros((N,10),bool);records=[];blockkeys=(v['y_px']//2048).astype(int)*100000+(v['x_px']//2048).astype(int)
    for cycle in range(10):
        imagebad=(mask[cycle]>0).all(axis=0);components,n=ndi.label(imagebad);pointcomp=components[py,px];eligible=(~obs[:,cycle]).all(axis=1)&((reasons[:,cycle]&3)>0).all(axis=1)&~((reasons[:,cycle]&4)>0).any(axis=1)
        for component in np.unique(pointcomp[eligible]):
            if component==0:continue
            region=np.flatnonzero(eligible&(pointcomp==component));record={'cycle':cycle+1,'component':int(component),'region_points':len(region),'status':'HISTORY_UNRESOLVED','qualified_blocks':[]}
            if cycle<2 or cycle>7 or len(region)<200:
                record['reason']='Insufficient independent prefix/later cycles or region points';records.append(record);continue
            selected=region[obs[region,:cycle].all(axis=(1,2))]
            if len(selected)<100:record['reason']='Earlier observations already unreliable';records.append(record);continue
            joined=np.concatenate([selected,control]);prefixmask=np.zeros(20,bool);prefixmask[:2*cycle]=True;z=(np.log10(raw[joined]+model['log_add'])-model['data_log_mean'])/model['data_log_std'];best=[];confidence=[]
            for start in range(0,len(joined),256):
                r=hm.infer(z[start:start+256],prefixmask,[]);best.extend(r['index']);confidence.extend(r['probability'])
            best=np.asarray(best,int);confidence=np.asarray(confidence);valid=(best>=0)&(best<K)&(confidence>.99);safe=np.clip(best,0,K-1)
            values=np.log1p(raw[joined].reshape(-1,10,2));amp=np.maximum(np.quantile(values[:,:cycle].reshape(len(joined),-1),.9,axis=1),1);values=values/amp[:,None,None];tv=np.log1p(raw[train].reshape(-1,10,2));ta=np.maximum(np.quantile(tv[:,:cycle].reshape(len(train),-1),.9,axis=1),1);tv/=ta[:,None,None];off=np.zeros((10,2));on=np.zeros((10,2));calibrated=True
            for t in range(10):
                for c in range(2):
                    bit=codes[labels[train],t,c];vv=tv[:,t,c]
                    if min((bit==0).sum(),(bit==1).sum())<15:calibrated=False;continue
                    off[t,c]=np.median(vv[bit==0]);on[t,c]=np.median(vv[bit==1])
            if not calibrated or np.any(on<=off):record['reason']='Normal control templates unidentifiable';records.append(record);continue
            later=np.arange(cycle+1,10);use=obs[joined][:,later];normalbits=codes[safe][:,later];pausebits=codes[safe][:,later-1];informative=((normalbits!=pausebits)&use).any(axis=(1,2));valid &= informative&(use.any(axis=2).sum(axis=1)>=2)
            pred0=off[later][None]+normalbits*(on-off)[later][None];pred1=off[later][None]+pausebits*(on-off)[later][None];d=(((values[:,later]-pred0)**2-(values[:,later]-pred1)**2)*use).sum(axis=(1,2));preference=d>0;joinedids=ids[joined]
            def summary(selection):
                s=selection&valid;return {'n':int(s.sum()),'pause_fraction':float(preference[s].mean()) if s.any() else 1.}
            a=np.arange(len(joined))<len(selected);b=~a;ar=summary(a);ar['halves']=[summary(a&(joinedids%2==k)) for k in [0,1]];cr=summary(b);cr['reference_n']=cr['n'];cr['prefix_reference_agreement']=float((best[b&valid]==labels[joined[b&valid]]).mean()) if cr['n'] else 0.
            blocks=[]
            for key in np.unique(blockkeys[selected]):
                s=a&(blockkeys[joined]==key);recordblock=summary(s);recordblock.update(key=int(key),halves=[summary(s&(joinedids%2==k)) for k in [0,1]]);blocks.append(recordblock)
            qualified=decide_no_shift(ar,cr,blocks);record.update(affected_prefix_holdout=ar,normal_prefix_holdout=cr,spatial_blocks=blocks,qualified_blocks=qualified)
            if qualified:
                chosen=region[np.isin(blockkeys[region],qualified)];no_pause[chosen,cycle]=True;record.update(status='NO_HISTORY_SHIFT_SUPPORTED',applied_points=len(chosen))
            else:record['reason']='Preserve mixture: no-shift evidence did not pass all independent gates'
            records.append(record);print(json.dumps({k:record[k] for k in ['cycle','component','status','region_points']}),flush=True)
    assert not (no_pause&((reasons&4)>0).any(axis=2)).any()
    assert frozen=={f:digest(f) for f in frozen};assert digest(source)==qr['source_transcripts_sha256'] and digest(intensity)==mr['input_parquet_sha256']
    out.mkdir(parents=True);np.savez_compressed(out/'mechanism_policy.npz',transcript_ids=ids,no_pause=no_pause)
    receipt={'status':'REGION_MECHANISM_POLICY_QUALIFIED','run_id':rid,'no_pause_points':int(no_pause.any(axis=1).sum()),'per_cycle_no_pause':no_pause.sum(axis=0).tolist(),'regions':records,'qc_receipt':str(qc/'receipt.json'),'qc_receipt_sha256':digest(qc/'receipt.json'),'model_receipt_sha256':digest(md/'receipt.json'),'policy_sha256':digest(out/'mechanism_policy.npz'),'source_code':frozen,'source_transcripts_sha256':qr['source_transcripts_sha256'],'source_intensities_sha256':mr['input_parquet_sha256'],'criteria':'Prefix P>0.99 with projected uniqueness; >=100 informative affected and >=30 controls; control prefix/reference agreement>=0.99; pause preference<=0.15 in affected/control and each disjoint affected ID half. At least3 spatial blocks pass n>=20, preference<=0.25 and each ID half n>=5/preference<=0.30. No Q20-gain or density-smoothness criterion.','limitations':['Technical reference labels, not independent biological truth.','First-cycle, insufficient-prefix and mixed-history regions retain the original mixture.','Qualified no-shift inference applies only to locally supported blocks within the image component.'],'production_changes':False}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'run_id':rid,'no_pause_points':receipt['no_pause_points'],'per_cycle':receipt['per_cycle_no_pause']}),flush=True);return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--qc-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run_dir,a.qc_dir,a.model_dir,a.output)
if __name__=='__main__':main()
