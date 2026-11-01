"""Automatic all-spot discovery with source-qualified later-round fallback.

Image components propose the scope, never gene names. All candidates remain
measured coordinates; invalid dimensions and unresolved histories stay unknown.
"""
import argparse,json,os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .__main__ import digest
from .discovery_policy import DiscoveryPolicy,early_failure_regions,alignment_gate,fallback_required,normalize_observed,candidate_gate
from .history_observability import ObservableHistoryModel
from .model import acquisition_codes
from .recover_detection import novel_candidates


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def source_manifest(run,repair_root,model_dir):
    from . import model,history_observability,discovery_policy
    from sprintseq.readout import spot_detection,intensity_readout,deduplicate,mosaic
    paths=[run/'readout/position.csv',run/'readout/intensity.csv',run/'readout/transcripts.parquet',run/'readout/transcript_intensities.parquet',run/'codebook/codebook.csv',repair_root/'repair/receipt.json',repair_root/'repair/repair_calls.npz',repair_root/'image_qc/receipt.json',repair_root/'image_qc/image_validity.npz',model_dir/'receipt.json',model_dir/'model.npz',Path(__file__)]
    paths += [Path(module.__file__) for module in [model,history_observability,discovery_policy,spot_detection,intensity_readout,deduplicate,mosaic]]
    for name in ['mechanisms','pause_mechanisms']:
        p=repair_root/name/'receipt.json'
        if p.exists():paths.append(p)
    result={str(p):digest(p) for p in paths}
    qr=json.loads((repair_root/'image_qc/receipt.json').read_text())
    for p,h in qr.get('coverage_sources',{}).items():
        if digest(p)!=h:raise ValueError('Renderer coverage changed')
        result[p]=h
    return result


def blocks(points,roi,shape,size=1024):
    y0,x0,y1,x1=roi
    for sy in range(y0,y1,size):
        for sx in range(x0,x1,size):
            ey=min(sy+size,y1);ex=min(sx+size,x1)
            ix=np.flatnonzero((points[:,0]>=sy)&(points[:,0]<ey)&(points[:,1]>=sx)&(points[:,1]<ex))
            if len(ix):yield ix,max(0,sy-32),max(0,sx-32),min(shape[0],ey+32),min(shape[1],ex+32)


def sample(region,points,shape):
    from spatial_img_core.observation_qc import sample_mask
    return sample_mask(region[None,None],points[:,1],points[:,0],shape).reshape(-1)


def detect(run,cycles,region,roi,shape,normalization,policy,device):
    from sprintseq.readout.mosaic import open_mosaic
    from sprintseq.readout.spot_detection import get_spot_coordinates,mosaic_percentiles
    from sprintseq.readout.intensity_readout import read_intensity_tophat
    coords=[];scores=[];channels=[];y0,x0,y1,x1=roi
    for cycle in cycles:
        for c,ch in enumerate(['cy3','cy5']):
            im=open_mosaic(run/'stitched',cycle,ch);key=f'{cycle}:{ch}'
            if key not in normalization:normalization[key]=list(map(float,mosaic_percentiles(im)))
            for sy in range(y0,y1,policy.block_size):
                for sx in range(x0,x1,policy.block_size):
                    ey=min(sy+policy.block_size,y1);ex=min(sx+policy.block_size,x1)
                    ay=max(0,sy-64);ax=max(0,sx-64);by=min(shape[0],ey+64);bx=min(shape[1],ex+64)
                    tile=np.asarray(im[ay:by,ax:bx]);found=get_spot_coordinates(tile,method='spotiflow',pretrained_name='hybiss',prob_thresh=.01,norm_range=normalization[key],device=device)
                    strength=read_intensity_tophat(tile,found);points=np.round(found)+[ay,ax]
                    keep=(points[:,0]>=sy)&(points[:,0]<ey)&(points[:,1]>=sx)&(points[:,1]<ex)&sample(region,points,shape).astype(bool)
                    coords.extend(points[keep]);scores.extend(strength[keep]);channels.extend([c]*int(keep.sum()))
            print('DISCOVERY_DETECTED',cycle,ch,len(coords),flush=True)
    return novel_candidates(np.asarray(coords).reshape(-1,2),np.asarray(channels,int),np.asarray(scores),np.empty((0,2)))


def measurements(run,points,roi,shape,cycles,policy):
    import cv2
    from sprintseq.readout.mosaic import open_mosaic
    from sprintseq.readout.intensity_readout import read_intensity_tophat
    raw=np.full((len(points),20),np.nan,np.float32);peaks={}
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    for t in sorted(set(range(1,11))|set(cycles)):
        for c,ch in enumerate(['cy3','cy5']):
            im=open_mosaic(run/'stitched',t,ch);peak=np.zeros(len(points),bool)
            for ix,ay,ax,by,bx in blocks(points,roi,shape,policy.block_size):
                tile=np.asarray(im[ay:by,ax:bx],dtype=np.float32);local=points[ix]-[ay,ax]
                if t<=10:raw[ix,2*(t-1)+c]=read_intensity_tophat(tile,local)
                top=cv2.morphologyEx(tile,cv2.MORPH_TOPHAT,kernel);inner=ndi.maximum_filter(top,3);outer=ndi.maximum_filter(top,7);yy,xx=np.round(local).astype(int).T
                peak[ix]=(inner[yy,xx]>=policy.peak_minimum)&(inner[yy,xx]>=outer[yy,xx])
            peaks[(t,c)]=peak
        print('DISCOVERY_MEASURED',t,len(points),flush=True)
    return raw,peaks


def spatial_history_policies(root,reason,points,shape,eligible):
    """Project already qualified region/block policies; do not infer new chemistry."""
    from .pause_mechanisms import infer_qualified_pause
    eligible=eligible.copy();forced=np.zeros(len(points),bool)
    for name,size in [('mechanisms',2048),('pause_mechanisms',1024)]:
        path=root/name/'receipt.json'
        if not path.exists():continue
        record=json.loads(path.read_text());keys=(points[:,0]//size).astype(int)*100000+(points[:,1]//size).astype(int)
        for row in record['regions']:
            qualified=row.get('qualified_blocks',[])
            if not qualified:continue
            t=int(row.get('cycle',1))-1;labels,_=ndi.label((reason[t]>0).all(axis=0));inside=sample(labels,points,shape)==row['component'];choose=inside&np.isin(keys,qualified)&eligible[:,t]
            if name=='mechanisms':eligible[choose,t]=False
            else:forced[choose]=True
    return eligible,forced


def exact_point_reason(run,reason,points,shape):
    """Use pixel-level rendered support for every newly measured coordinate."""
    import zlib
    from spatial_img_core.observation_qc import sample_mask
    from spatial_img_core.observation_qc.segment import sample_packed_support
    result=sample_mask(reason,points[:,1],points[:,0],shape)
    path=Path(run)/'stitched/finite_stitch_report.json'
    if not path.exists():return result
    planes={(r['cycle'],r['channel']):r for r in json.loads(path.read_text())['planes']}
    result &= np.uint8(255-4)
    for t in range(1,11):
        for c,ch in enumerate(['cy3','cy5']):
            record=planes.get((t,ch))
            if record is None or record['support_encoding']!='zlib, row-major np.packbits(axis=1, bitorder=little)':raise ValueError('Incomplete exact renderer support')
            packed=zlib.decompress(Path(record['support_path']).read_bytes());valid=sample_packed_support(packed,shape,points[:,1],points[:,0]);result[~valid,t-1,c]|=4
    return result


def decode_candidates(root,reason,points,shape,raw,peaks,discovery_cycles,aligned,model,hm,policy,point_reason=None):
    from spatial_img_core.observation_qc import sample_mask
    from .pause_mechanisms import infer_qualified_pause
    rr=sample_mask(reason,points[:,1],points[:,0],shape) if point_reason is None else np.asarray(point_reason);observed=(rr==0).reshape(-1,20)
    eligible=(rr>0).all(axis=2)&((rr&3)>0).all(axis=2)&~((rr&4)>0).any(axis=2)
    eligible,forced=spatial_history_policies(root,reason,points,shape,eligible)
    forced &= ~observed[:,:2].any(axis=1)&observed[:,2:].all(axis=1)
    values=normalize_observed(raw,observed,model);signature=np.c_[observed,eligible,forced];patterns,inverse=np.unique(signature,axis=0,return_inverse=True)
    n=len(points);index=np.full(n,-1);prob=np.zeros(n);p2=prob.copy();entropy=prob.copy();hp=prob.copy();hbits=np.zeros(n,np.uint16);sequence=np.zeros(n,bool);risk=np.zeros(n,bool)
    physical=np.stack([peaks[(t,c)] for t in range(1,11) for c in range(2)],axis=1)
    fallback=policy.allspot_cycle not in discovery_cycles
    for k,pattern in enumerate(patterns):
        rows=np.flatnonzero(inverse==k);mask=pattern[:20];events=np.flatnonzero(pattern[20:30]).tolist()
        for start in range(0,len(rows),32):
            ix=rows[start:start+32];r=infer_qualified_pause(hm,values[ix],mask) if pattern[-1] else hm.infer(values[ix],mask,events)
            index[ix]=r['index'];prob[ix]=r['probability'];p2[ix]=r['probability_2'];entropy[ix]=r['entropy'];hp[ix]=r['history_probability']
            for h,history in enumerate(r['histories']):
                take=ix[(r['history_index']==h)&(r['index']>=0)&(r['index']<hm.K)]
                if not len(take):continue
                codes=model['all_codes'][index[take]];acquired=acquisition_codes(codes,history,[(0,0)]*len(history)).astype(bool);hbits[take]=sum(1<<c for c in history)
                expected=physical[take]&observed[take]&acquired
                sequence[take]=expected.reshape(len(take),10,2).any(axis=2).sum(axis=1)>=(2 if fallback else 1)
                window=2*policy.discovery_window
                risk[take]=(codes[:,:window].astype(bool)&~observed[take,:window]).any(axis=1)|(codes[:,:window].astype(bool)&~acquired[:,:window]).any(axis=1)
    discovery=np.any(np.stack([peaks[(t,c)] for t in discovery_cycles for c in range(2)]),axis=0)
    adopt=candidate_gate(index,prob,hp,hm.K,aligned,discovery,sequence,risk,policy)
    raw=raw.copy();raw[(rr.reshape(n,20)&4)>0]=np.nan
    return dict(y_px=points[:,0],x_px=points[:,1],intensities=raw,observed=observed,reason=rr,candidate_index=index,probability=np.clip(prob,0,1),probability_2=np.clip(p2,0,1),entropy=entropy,history_probability=hp,history_cycle_bits=hbits,alignment_qualified=np.asarray(aligned,bool),physical_discovery=discovery,physical_sequence=sequence,lost_detection_support=risk,adoptable=adopt)


def _run(run_dir,repair_root,output,*,policy=DiscoveryPolicy(),device='cpu'):
    from sprintseq.readout.mosaic import list_mosaics
    run_dir=Path(run_dir).resolve();root=Path(repair_root).resolve();out=Path(output).resolve()
    repair=json.loads((root/'repair/receipt.json').read_text());md=Path(repair['model_receipt']).parent;mr=json.loads((md/'receipt.json').read_text());qr=json.loads((root/'image_qc/receipt.json').read_text());image=run_dir/'stitched/mosaic.ome.tif';identity=[image.stat().st_size,image.stat().st_mtime_ns]
    if identity!=qr['source_tiff_identity'] or digest(run_dir/'readout/transcripts.parquet')!=repair['source_transcripts_sha256'] or digest(md/'model.npz')!=mr['model_sha256']:
        raise ValueError('Changed molecular/source binding before discovery')
    if digest(root/'image_qc/image_validity.npz')!=qr['image_validity_sha256'] or digest(run_dir/'readout/transcript_intensities.parquet')!=mr['input_parquet_sha256']:raise ValueError('Changed image validity/model input')
    bound=source_manifest(run_dir,root,md);contract={'source_hashes':bound,'image_identity':identity,'policy':policy.as_dict()}
    out.mkdir(parents=True,exist_ok=True);binding=out/'binding.json'
    if binding.exists() and json.loads(binding.read_text())!=contract:raise ValueError('Discovery checkpoint inputs changed')
    save(binding,contract)
    if (out/'receipt.json').exists():
        result=json.loads((out/'receipt.json').read_text())
        for path,h in result['products'].items():
            if digest(path)!=h:raise ValueError('Discovery checkpoint product changed')
        return result
    v=np.load(root/'image_qc/image_validity.npz');reason=v['reason'];shape=tuple(v['full_shape']);labels=early_failure_regions(reason,policy);boxes=ndi.find_objects(labels)
    base=pq.read_table(run_dir/'readout/transcripts.parquet').to_pandas();old=base[['y_px','x_px']].to_numpy();oldtree=cKDTree(old)
    rawtable=pq.read_table(run_dir/'readout/transcript_intensities.parquet').to_pandas().set_index('transcript_id');model=np.load(md/'model.npz');hm=ObservableHistoryModel(model,mr['gene_classes']);available=set(list_mosaics(run_dir/'stitched'));allspot=all((policy.allspot_cycle,c) in available for c in ['cy3','cy5']);late=[t for t in range(policy.discovery_window+1,11) if all((t,c) in available for c in ['cy3','cy5'])];normalization={};records=[];accepted=[]
    for component,box in enumerate(boxes,1):
        if box is None:continue
        region=labels==component;scale=np.asarray(shape)/labels.shape;y0,x0=np.maximum(0,np.floor([box[0].start*scale[0],box[1].start*scale[1]])-32).astype(int);y1,x1=np.minimum(shape,np.ceil([box[0].stop*scale[0],box[1].stop*scale[1]])+32).astype(int);roi=list(map(int,[y0,x0,y1,x1]));inside=sample(region,old,shape).astype(bool);anchors=old[inside&base.is_gene.to_numpy(bool)&(base.probability.to_numpy()>.99)]
        if len(anchors)<policy.minimum_anchors:
            records.append({'component':component,'status':'INSUFFICIENT_ALIGNMENT_ANCHORS','roi':roi,'anchors':len(anchors),'adoptable':0,'reason':'Neither all-spot nor later-round alignment can meet the fixed minimum support; no density filling.'});continue
        attempts=[('allspot',[policy.allspot_cycle])] if allspot else [('late',late)]
        for kind,cycles in attempts:
            folder=out/f'component_{component}_{kind}';folder.mkdir(exist_ok=True)
            if (folder/'receipt.json').exists():
                record=json.loads((folder/'receipt.json').read_text());data=np.load(folder/'candidates.npz');payload={k:data[k] for k in data.files}
                if digest(folder/'candidates.npz')!=record['candidates_sha256']:raise ValueError('Changed discovery component')
            else:
                if not cycles:
                    records.append({'component':component,'kind':kind,'status':'NO_AVAILABLE_MEASURED_DISCOVERY_CYCLE','roi':roi});continue
                save(out/'state.json',{'status':'DETECTING','component':component,'kind':kind,'cycles':cycles})
                detected,bits=detect(run_dir,cycles,region,roi,shape,normalization,policy,device);novel=oldtree.query(detected,p=np.inf)[0]>policy.exclusion_radius;points=detected[novel]
                if accepted:points=points[cKDTree(np.concatenate(accepted)).query(points,p=np.inf)[0]>policy.exclusion_radius]
                aligned,alignment=alignment_gate(anchors,detected,points,policy)
                checks=np.flatnonzero(inside);checks=checks[np.linspace(0,len(checks)-1,min(128,len(checks)),dtype=int)]
                raw,peaks=measurements(run_dir,np.r_[points,old[checks]],roi,shape,cycles,policy)
                reference=rawtable.loc[base.transcript_id.to_numpy()[checks],mr['intensity_order']].to_numpy();measured=np.isfinite(reference)
                if not np.array_equal(raw[len(points):][measured],reference[measured]):raise ValueError('Old-spot intensity replay differs')
                payload=decode_candidates(root,reason,points,shape,raw[:len(points)],{k:v[:len(points)] for k,v in peaks.items()},cycles,aligned,model,hm,policy,point_reason=exact_point_reason(run_dir,reason,points,shape))
                np.savez_compressed(folder/'candidates.npz',**payload)
                record={'component':component,'kind':kind,'cycles':cycles,'roi':roi,'new_candidates':len(points),'adoptable':int(payload['adoptable'].sum()),'alignment':alignment,'intensity_replay_rows':len(checks),'intensity_replay_exact':True,'normalization':{key:value for key,value in normalization.items() if int(key.split(':')[0]) in cycles},'candidates':str(folder/'candidates.npz'),'candidates_sha256':digest(folder/'candidates.npz')};save(folder/'receipt.json',record)
            records.append(record);take=payload['adoptable'];accepted.append(np.c_[payload['y_px'][take],payload['x_px'][take]])
            if kind=='allspot' and fallback_required(True,record['alignment']):attempts.append(('late',late))
            print('DISCOVERY_COMPONENT_COMPLETE',component,kind,int(take.sum()),flush=True)
    if bound!=source_manifest(run_dir,root,md) or identity!=[image.stat().st_size,image.stat().st_mtime_ns]:raise ValueError('Source changed during discovery')
    result={'status':'SUPPLEMENTAL_DISCOVERY_TECHNICALLY_VERIFIED_WITH_LIMITS','run_id':repair['run_id'],'components':records,'adoptable':sum(r.get('adoptable',0) for r in records),'source_hashes':bound,'policy':policy.as_dict(),'detector_recipe':{'method':'spotiflow','pretrained_name':'hybiss','probability_threshold':.01,'normalization':'global_per_mosaic'},'products':{r['candidates']:r['candidates_sha256'] for r in records if 'candidates' in r},'limitations':['All-spot is coordinate discovery only, never a barcode base.','Poor local alignment retains unknown coverage and triggers later measured rounds.','Later-round two-observed-cycle peak requirement may reduce sparse-codeword recall.','No image or expression interpolation; conditional confidence is not biological FDR.']};save(out/'receipt.json',result);save(out/'state.json',{'status':'COMPLETE','receipt':str(out/'receipt.json')});return result


def run(run_dir,repair_root,output,*,policy=DiscoveryPolicy(),device='cpu'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);lock=out/'discovery.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'run_dir':str(run_dir)},f)
    try:return _run(run_dir,repair_root,out,policy=policy,device=device)
    finally:lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True,type=Path);p.add_argument('--repair-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--allspot-cycle',type=int,default=11);p.add_argument('--device',default='cpu');a=p.parse_args();print(json.dumps(run(a.run_dir,a.repair_root,a.output,policy=DiscoveryPolicy(allspot_cycle=a.allspot_cycle),device=a.device)),flush=True)

if __name__=='__main__':main()
