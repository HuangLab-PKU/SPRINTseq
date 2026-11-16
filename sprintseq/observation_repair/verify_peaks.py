"""Reject rescued detector candidates located on shoulders of brighter nearby peaks."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from sprintseq.readout.mosaic import open_mosaic
from .__main__ import digest


def peak_supported(tophat,coords,search_radius=1,comparison_radius=3):
    """Require a true peak within native readout tolerance, not a nearby peak tail."""
    coords=np.round(coords).astype(int)
    inner=maximum_filter(tophat,size=2*search_radius+1,mode='constant')
    outer=maximum_filter(tophat,size=2*comparison_radius+1,mode='constant')
    v=inner[coords[:,0],coords[:,1]]
    return (v>=50)&(v>=outer[coords[:,0],coords[:,1]])


def run(run_dir,detection_dir,output):
    run,dd,out=map(Path,[run_dir,detection_dir,output])
    if out.exists():raise ValueError('Immutable peak validation exists')
    report=json.loads((dd/'receipt.json').read_text());data=np.load(dd/'new_spots.npz');assert digest(dd/'new_spots.npz')==report['new_spots_sha256']
    for f,h in report['source_hashes'].items():
        if digest(f)!=h:raise ValueError('Changed detection source')
    coords=np.column_stack([data['y_px'],data['x_px']]);support=np.zeros(len(coords),np.uint16);y0,x0,y1,x1=report['roi_yxyx'];kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    for t in range(4,10):
        discovered=(data['detection_cycle_bits']&(1<<t))>0
        for ch in ['cy3','cy5']:
            mosaic=open_mosaic(run/'stitched',t+1,ch)
            for sy in range(y0,y1,1024):
                for sx in range(x0,x1,1024):
                    ey=min(sy+1024,y1);ex=min(sx+1024,x1);pick=discovered&(coords[:,0]>=sy)&(coords[:,0]<ey)&(coords[:,1]>=sx)&(coords[:,1]<ex)
                    if not pick.any():continue
                    ay=max(0,sy-64);ax=max(0,sx-64);tile=np.asarray(mosaic[ay:ey+64,ax:ex+64],dtype=np.float32);top=cv2.morphologyEx(tile,cv2.MORPH_TOPHAT,kernel);valid=peak_supported(top,coords[pick]-[ay,ax]);rows=np.flatnonzero(pick)[valid];support[rows]|=np.uint16(1<<t)
    keep=support>0;out.mkdir(parents=True);payload={k:data[k][keep] for k in data.files};payload['verified_peak_cycle_bits']=support[keep];np.savez_compressed(out/'new_spots.npz',**payload)
    import pandas as pd
    genes=pd.read_csv(run/'codebook/codebook.csv').Gene.astype(str).tolist();idx=payload['candidate_index'];q=(idx>=0)&(idx<len(genes))&(payload['probability']>.99);report.update(new_candidates=int(keep.sum()),new_Q20=int(q.sum()),per_gene_Q20={g:int((q&(idx==i)).sum()) for i,g in enumerate(genes)},parent_detection_receipt=str(dd/'receipt.json'),parent_detection_receipt_sha256=digest(dd/'receipt.json'),new_spots_sha256=digest(out/'new_spots.npz'),peak_localization_rejected=int((~keep).sum()),peak_localization_Q20_rejected=int((((data['candidate_index']>=0)&(data['candidate_index']<len(genes))&(data['probability']>.99))&~keep).sum()))
    report['source_hashes'][str(Path(__file__))]=digest(__file__);report['source_hashes'][str(dd/'receipt.json')]=digest(dd/'receipt.json');report['criteria']+=' Added source-pixel check: at least one discovery-plane top-hat peak within native radius 1 equals radius-3 maximum; peak >=50. Reject shoulders of brighter peaks; no coordinate shifting.'
    (out/'receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['new_candidates','new_Q20','peak_localization_rejected','peak_localization_Q20_rejected']}),flush=True);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir','detection-dir','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.run_dir,a.detection_dir,a.output)

if __name__=='__main__':main()
