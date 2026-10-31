"""Recover measured late-cycle spots inside a qualified first-pause image region.

Discovery is label independent, uses the baseline Spotiflow recipe, preserves old
coordinates, and requires independent late-cycle detections. No density filling.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from sprintseq.readout.mosaic import open_mosaic
from sprintseq.readout.spot_detection import get_spot_coordinates,mosaic_percentiles
from sprintseq.readout.intensity_readout import read_intensity_tophat
from sprintseq.readout.deduplicate import deduplicate_coordinates
from spatial_img_core.observation_qc import sample_mask
from .__main__ import digest
from .model import HistoryModel
from .pause_mechanisms import infer_qualified_pause


def novel_multicycle_candidates(coords, cycles, scores, existing, exclusion_radius=3.):
    """Native 2px dedup, >=2 acquisition cycles, and >3px from any existing spot."""
    if not len(coords):return np.empty((0,2)),np.empty(0,np.uint16)
    keep,unique=deduplicate_coordinates(coords,threshold=2,scores=scores)
    distance,nearest=cKDTree(unique).query(coords,p=np.inf)
    bits=np.zeros(len(unique),np.uint16);valid=distance<=2
    np.bitwise_or.at(bits,nearest[valid],(1<<np.asarray(cycles)[valid]).astype(np.uint16))
    support=np.array([int(b).bit_count() for b in bits])>=2
    if len(existing):support &= cKDTree(existing).query(unique,p=np.inf)[0]>exclusion_radius
    return unique[support],bits[support]


def run(run_dir,policy_dir,model_dir,output,*,control_offset_x=0,device='cuda'):
    run,pol,md,out=map(Path,[run_dir,policy_dir,model_dir,output])
    if out.exists():raise ValueError('Existing immutable redetection output')
    out.mkdir(parents=True)
    source=run/'readout/transcripts.parquet';intensity=run/'readout/transcript_intensities.parquet';image=run/'stitched/mosaic.ome.tif'
    bound={str(f):digest(f) for f in [source,intensity,pol/'receipt.json',pol/'pause_policy.npz',md/'receipt.json',md/'model.npz',Path(__file__)]}
    imageid=[image.stat().st_size,image.stat().st_mtime_ns]
    pr=json.loads((pol/'receipt.json').read_text());assert pr['status']=='FIRST_PAUSE_REGION_POLICY_QUALIFIED';assert digest(pol/'pause_policy.npz')==pr['policy_sha256']
    # The pause policy's sibling image QC is the source of the physical component.
    qc=pol.parent/'image_qc';qr=json.loads((qc/'receipt.json').read_text());assert digest(qc/'receipt.json')==pr['qc_receipt_sha256']
    masks=np.load(qc/'image_validity.npz');reason=masks['reason'];full=tuple(masks['full_shape']);components,_=ndi.label((reason[0]>0).all(axis=0))
    allowed=np.zeros(components.shape,bool);keys=[]
    for region in pr['regions']:
        if region['qualified_blocks']:allowed |= components==region['component'];keys.extend(region['qualified_blocks'])
    yy,xx=np.where(allowed)
    if not len(yy):raise ValueError('No qualified first-pause region')
    scale=np.array(full)/np.array(allowed.shape);y0=max(0,int(yy.min()*scale[0])-32);y1=min(full[0],int((yy.max()+1)*scale[0])+32);x0=max(0,int(xx.min()*scale[1])+control_offset_x-32);x1=min(full[1],int((xx.max()+1)*scale[1])+control_offset_x+32)
    base=pq.read_table(source,columns=['transcript_id','y_px','x_px']).to_pandas();oldcoords=base[['y_px','x_px']].to_numpy();roiold=(oldcoords[:,0]>=y0)&(oldcoords[:,0]<y1)&(oldcoords[:,1]>=x0)&(oldcoords[:,1]<x1)
    coords=[];cycles=[];scores=[];norms={};steps=[]
    for t in range(4,10):
        for c,ch in enumerate(['cy3','cy5']):
            mosaic=open_mosaic(run/'stitched',t+1,ch);norm=mosaic_percentiles(mosaic);norms[f'cyc_{t+1}_{ch}']=list(norm)
            count=0
            for sy in range(y0,y1,1024):
                for sx in range(x0,x1,1024):
                    ey=min(sy+1024,y1);ex=min(sx+1024,x1);ay=max(0,sy-64);ax=max(0,sx-64);by=min(full[0],ey+64);bx=min(full[1],ex+64)
                    tile=np.asarray(mosaic[ay:by,ax:bx]);found=get_spot_coordinates(tile,method='spotiflow',pretrained_name='hybiss',prob_thresh=.01,norm_range=norm,device=device)
                    if not len(found):continue
                    score=read_intensity_tophat(tile,found,tophat_radius=3,search_radius=1);found=np.round(found)+[ay,ax];keep=(found[:,0]>=sy)&(found[:,0]<ey)&(found[:,1]>=sx)&(found[:,1]<ex)
                    found=found[keep];score=score[keep]
                    if not len(found):continue
                    rr=sample_mask(reason,found[:,1],found[:,0],full)
                    if control_offset_x:
                        valid=(rr==0).all(axis=(1,2))
                    else:
                        block=(found[:,0]//1024).astype(int)*100000+(found[:,1]//1024).astype(int)
                        valid=(rr[:,0]>0).all(axis=1)&(rr[:,1:]==0).all(axis=(1,2))&np.isin(block,keys)
                    coords.extend(found[valid]);scores.extend(score[valid]);cycles.extend([t]*int(valid.sum()));count+=int(valid.sum())
            steps.append(dict(cycle=t+1,channel=ch,candidates=count));(out/'state.json').write_text(json.dumps(dict(status='DETECTING',steps=steps),indent=2));print('DETECTED',t+1,ch,count,flush=True)
    coords=np.asarray(coords).reshape(-1,2);unique,bits=novel_multicycle_candidates(coords,np.asarray(cycles),np.asarray(scores),oldcoords)
    print('NOVEL_MULTICYCLE',len(unique),flush=True)
    # Reread old points through the same local interface to verify coordinate/intensity parity.
    ix=np.flatnonzero(roiold);ix=ix[np.linspace(0,len(ix)-1,min(256,len(ix)),dtype=int)];checkcoords=oldcoords[ix];allcoords=np.r_[unique,checkcoords];raw=np.zeros((len(allcoords),20),np.float32)
    for t in range(10):
        for c,ch in enumerate(['cy3','cy5']):
            mosaic=open_mosaic(run/'stitched',t+1,ch)
            for sy in range(y0,y1,1024):
                for sx in range(x0,x1,1024):
                    ey=min(sy+1024,y1);ex=min(sx+1024,x1);pick=(allcoords[:,0]>=sy)&(allcoords[:,0]<ey)&(allcoords[:,1]>=sx)&(allcoords[:,1]<ex)
                    if not pick.any():continue
                    ay=max(0,sy-64);ax=max(0,sx-64);by=min(full[0],ey+64);bx=min(full[1],ex+64);tile=np.asarray(mosaic[ay:by,ax:bx]);raw[pick,2*t+c]=read_intensity_tophat(tile,allcoords[pick]-[ay,ax],tophat_radius=3,search_radius=1)
        print('INTENSITY_READ',t+1,flush=True)
    mr=json.loads((md/'receipt.json').read_text());model=np.load(md/'model.npz');hm=HistoryModel(model,mr['gene_classes']);cols=mr['intensity_order'];oldraw=pq.read_table(intensity,filters=[('transcript_id','in',base.transcript_id.to_numpy()[ix].tolist())]).to_pandas().set_index('transcript_id').loc[base.transcript_id.to_numpy()[ix],cols].to_numpy()
    error=float(np.max(np.abs(raw[len(unique):]-oldraw)))
    if error!=0:raise ValueError('Reread disagrees with baseline intensities: '+str(error))
    raw=raw[:len(unique)];z=(np.log10(raw+model['log_add'])-model['data_log_mean'])/model['data_log_std'];mask=np.ones(20,bool)
    if not control_offset_x:mask[:2]=False
    index=[];prob=[];second=[];entropy=[]
    for start in range(0,len(z),128):
        result=hm.infer(z[start:start+128],mask,[]) if control_offset_x else infer_qualified_pause(hm,z[start:start+128],mask)
        index.extend(result['index']);prob.extend(result['probability']);second.extend(result['probability_2']);entropy.extend(result['entropy'])
    index=np.asarray(index);prob=np.asarray(prob);genes=pd.read_csv(run/'codebook/codebook.csv').Gene.astype(str).tolist();q=(index>=0)&(index<len(genes))&(prob>.99)
    np.savez_compressed(out/'new_spots.npz',y_px=unique[:,0],x_px=unique[:,1],intensities=raw,detection_cycle_bits=bits,candidate_index=index,probability=prob,probability_2=np.asarray(second),entropy=np.asarray(entropy))
    if bound!={f:digest(f) for f in bound} or imageid!=[image.stat().st_size,image.stat().st_mtime_ns]:raise ValueError('Redetection sources changed')
    report=dict(status='MEASURED_LATE_CYCLE_CANDIDATES_READY_FOR_REVIEW',run_id=pr['run_id'],control_offset_x=control_offset_x,roi_yxyx=[y0,x0,y1,x1],existing_roi_spots=int(roiold.sum()),new_multicycle_candidates=len(unique),new_Q20=int(q.sum()),per_gene_Q20={g:int((q&(index==i)).sum()) for i,g in enumerate(genes)},intensity_reread_controls=len(ix),intensity_reread_max_difference=error,normalization=norms,detection=steps,source_hashes=bound,source_tiff_identity=imageid,new_spots_sha256=digest(out/'new_spots.npz'),criteria='Baseline hybiss p=.01 and global normalization. At least two distinct late detection cycles; native 2px dedup; >3px Chebyshev distance from all old spots; exact old-spot intensity replay. Bubble additions restricted to qualified image component and spatial blocks. No original rows changed.',scientific_accuracy_accepted=False)
    (out/'receipt.json').write_text(json.dumps(report,indent=2));(out/'state.json').write_text(json.dumps({'status':'COMPLETE','new_Q20':int(q.sum())}));print(json.dumps({k:report[k] for k in ['status','new_multicycle_candidates','new_Q20','intensity_reread_max_difference']}),flush=True);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir','policy-dir','model-dir','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--control-offset-x',type=int,default=0);p.add_argument('--device',default='cuda');a=p.parse_args();run(a.run_dir,a.policy_dir,a.model_dir,a.output,control_offset_x=a.control_offset_x,device=a.device)

if __name__=='__main__':main()
