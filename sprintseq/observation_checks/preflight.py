"""Pre-readout source integrity and conservative codebook coverage planning."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from sprintseq.readout.mosaic import list_mosaics,open_mosaic
from sprintseq.observation_repair.model import acquisition_codes
from sprintseq.observation_repair.history_observability import relevant_pauses
from sprintseq.observation_repair.__main__ import digest


class ObservationHold(ValueError):
    """Processing cannot proceed safely with the recorded observations."""


def require_finite(frame,columns):
    values=frame[list(columns)].to_numpy(dtype=float);bad=~np.isfinite(values)
    if bad.any():
        counts={c:int(bad[:,i].sum()) for i,c in enumerate(columns) if bad[:,i].any()}
        raise ObservationHold('HOLD_MISSING_NOT_MEASURED: failed intensity reads must not become zero: '+json.dumps(counts))


def coverage_plan(codes,requested,reason,max_events=2,allow_local_nodata=False):
    """Cover each feature under observed erasures and all eligible pause histories.

Cycles are zero-based here. Image-only hypotheses are conservative acquisition
plans, never chemical diagnoses. Out-of-sequencing detection rounds stay selected
but cannot establish codebook coverage without a separate protocol contract.
"""
    codes=np.asarray(codes);T,C=reason.shape[:2]
    if codes.ndim!=2 or codes.shape[1]!=2*T or C!=2:raise ObservationHold('HOLD_CODEBOOK_LAYOUT')
    failure=(reason>0).reshape(2*T,-1);keys=np.sum(failure.astype(np.uint64)*(np.uint64(1)<<np.arange(2*T,dtype=np.uint64))[:,None],axis=0);footprint=~((reason&4)>0).all(axis=(0,1))
    if not footprint.any():raise ObservationHold('HOLD_NO_ACQUIRED_SEQUENCING_SUPPORT')
    patterns,counts=np.unique(keys[footprint.ravel()],return_counts=True)
    requirements=[];scenarios=[];repair_scope_exceeded=[]
    # Normal support is always checked, including when no image defect is found.
    patterns=np.r_[np.uint64(0),patterns[counts>=64]];patterns=np.unique(patterns)
    for key in patterns:
        missing=((np.uint64(key)>>np.arange(2*T,dtype=np.uint64))&1).astype(bool).reshape(T,2)
        # SOURCE_MISSING only is never eligible for a chemistry pause.
        pixels=keys==key;chemical=np.any((reason.reshape(2*T,-1)[:,pixels]&3)>0,axis=1).reshape(T,2) if pixels.any() else np.zeros((T,2),bool)
        source_missing=np.any((reason.reshape(2*T,-1)[:,pixels]&4)>0,axis=1).reshape(T,2) if pixels.any() else np.zeros((T,2),bool)
        eligible=np.flatnonzero(missing.all(axis=1)&chemical.all(axis=1)&~source_missing.any(axis=1)).tolist()
        eligible=list(relevant_pauses(~missing.ravel(),eligible))
        if len(eligible)>max_events:repair_scope_exceeded.append({'pattern':int(key),'eligible_pause_cycles':[t+1 for t in eligible]})
        for k in range(len(eligible)+1):
            for pauses in itertools.combinations(eligible,k):
                if len(scenarios)>=4096:raise ObservationHold('HOLD_DETECTION_PLANNING_BUDGET: too many acquisition scenarios')
                acquired=acquisition_codes(codes,pauses,[(0,0)]*len(pauses)).reshape(len(codes),T,2)
                support=(acquired.astype(bool)&~missing[None]).any(axis=2);requirements.append(support);scenarios.append({'pattern':int(key),'missing_dimensions':np.flatnonzero(missing.ravel()).tolist(),'candidate_pause_cycles':[t+1 for t in pauses]})
    support=np.concatenate(requirements);recoverable=support.any(axis=1);unobservable=[{'scenario':int(i//len(codes)),'feature_index':int(i%len(codes))} for i in np.flatnonzero(~recoverable)]
    if unobservable and not allow_local_nodata:raise ObservationHold('HOLD_UNOBSERVABLE_CODEWORDS')
    selected=sorted(set(requested));extra=[];supplement_regions={}
    before=np.flatnonzero(~support[:,[t for t in selected if 0<=t<T]].any(axis=1))
    while True:
        uncovered=(~support[:,[t for t in selected if 0<=t<T]].any(axis=1))&recoverable
        if not uncovered.any():break
        available=[t for t in range(T) if t not in selected];gain={t:int(support[uncovered,t].sum()) for t in available}
        if not gain or max(gain.values())==0:raise ObservationHold('HOLD_UNOBSERVABLE_CODEWORDS: no measured acquisition can detect all required codewords')
        choice=max(available,key=lambda t:(gain[t],-t));needed=np.flatnonzero(uncovered&support[:,choice]);supplement_regions[str(choice+1)]=sorted({scenarios[int(i//len(codes))]['pattern'] for i in needed});selected.append(choice);extra.append(choice)
    return {'supplement_pattern_keys':supplement_regions,'unobservable_feature_patterns':unobservable,'repair_scope_exceeded':repair_scope_exceeded,'requested_cycles':[t+1 for t in requested],'effective_cycles':[t+1 for t in sorted(selected)],'added_cycles':[t+1 for t in extra],'features_at_risk':sorted(set((before%len(codes)).tolist())),'scenarios':scenarios,'outside_acquisition_canvas_pixels':int((~footprint).sum()),'scope':'Global minimum greedy cycle union across image-derived erasure/history scenarios; no expression-count optimization.'}


def registration_evidence(run):
    run=Path(run);report=run/'registration_report.json';unresolved=run/'registration_unresolved.json'
    if not report.exists():
        if unresolved.exists():raise ObservationHold('HOLD_UNRESOLVED_REGISTRATION: no source-bound solved report')
        return {'status':'UNAVAILABLE_LEGACY_NOT_ASSUMED_ACCEPTED','sources':{}}
    data=json.loads(report.read_text());shift=run/'shift_table.csv'
    if not shift.exists() or digest(shift)!=data.get('shift_table_sha256'):raise ObservationHold('HOLD_REGISTRATION_SOURCE_MISMATCH')
    measurement=data.get('measurement',{})
    if 'unresolved' not in measurement or measurement['unresolved']:raise ObservationHold('HOLD_UNRESOLVED_REGISTRATION')
    return {'status':'NUMERIC_REGISTRATION_COMPLETE_NOT_SCIENTIFIC_ACCEPTANCE','quality_exceptions':len(measurement.get('quality_exceptions',[])),'sources':{str(report):digest(report),str(shift):digest(shift)}}


def run(run_dir,ref_file=None,detection_cycles=(1,2,3,4),seq_cycles=10,channels=('cy3','cy5')):
    run=Path(run_dir);ref=Path(ref_file) if ref_file else run/'codebook/codebook.csv';dest=run/'observation_checks';dest.mkdir(exist_ok=True)
    def hold(message):
        (dest/'preflight_hold.json').write_text(json.dumps({'status':'HOLD','reason':message,'run_id':run.name,'recovery':'Restore/qualify sources or supply the exact codebook, then rerun preflight; do not fill missing intensities.'},indent=2));raise ObservationHold(message)
    from .acquisition import acquisition_check
    try:acquisition=acquisition_check(run)
    except ObservationHold as error:hold(str(error))
    try:geometry=registration_evidence(run)
    except ObservationHold as error:hold(str(error))
    if list(channels)!=['cy3','cy5'] or seq_cycles!=10:hold('HOLD_UNQUALIFIED_PROTOCOL: integrated artifact policy currently qualified for ten two-colour sequencing cycles')
    if not ref.is_file():hold('HOLD_CODEBOOK_REQUIRED_BEFORE_READOUT: '+str(ref))
    book=pd.read_csv(ref)
    if not {'Gene','Barcode'}<=set(book.columns) or not len(book) or book.Gene.duplicated().any():hold('HOLD_INVALID_CODEBOOK')
    enc={'A':[1,1],'T':[1,0],'C':[0,1],'G':[0,0]}
    if any(len(str(b))!=seq_cycles or set(str(b))-set(enc) for b in book.Barcode):hold('HOLD_CODEBOOK_CYCLE_MISMATCH')
    codes=np.array([[v for base in str(b) for v in enc[base]] for b in book.Barcode]);available=set(list_mosaics(run/'stitched'));required={(t,ch) for t in set(range(1,seq_cycles+1))|set(detection_cycles) for ch in channels};missing=required-available
    if missing:hold('HOLD_MISSING_MOSAIC_PLANES: '+repr(sorted(missing)))
    shapes={tuple(open_mosaic(run/'stitched',t,ch).shape) for t,ch in required}
    if len(shapes)!=1:hold('HOLD_MOSAIC_FRAME_MISMATCH')
    image=run/'stitched/mosaic.ome.tif'
    if not image.is_file():hold('HOLD_FINALIZED_IMAGE_REQUIRED')
    from spatial_img_core.observation_qc import prescan,segment
    sources={str(Path(__file__)):digest(__file__),str(Path(prescan.__file__)):digest(prescan.__file__),str(Path(segment.__file__)):digest(segment.__file__)}
    from sprintseq.observation_repair import model as history_model,history_observability
    from . import acquisition
    sources.update({str(Path(module.__file__)):digest(module.__file__) for module in [history_model,history_observability,acquisition]})
    identity=[image.stat().st_size,image.stat().st_mtime_ns];contract=dict(acquisition=acquisition,registration=geometry,run_id=run.name.removesuffix('_processed'),image_identity=identity,codebook_sha256=digest(ref),requested_cycles=list(detection_cycles),seq_cycles=seq_cycles,channels=list(channels),source_code=sources)
    finite=run/'stitched/finite_stitch_report.json';contract['finite_report_sha256']=digest(finite) if finite.exists() else None
    key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()[:20];out=dest/'preflight'/key;receipt=out/'receipt.json'
    if receipt.exists():
        result=json.loads(receipt.read_text())
        for path,h in result['contract']['registration']['sources'].items():
            if digest(path)!=h:hold('HOLD_CHANGED_REGISTRATION')
        if digest(out/'detection_rois.npz')!=result['roi_sha256']:hold('HOLD_CHANGED_DETECTION_ROIS')
        if digest(out/'image_validity.npz')!=result['validity_sha256']:hold('HOLD_CHANGED_VALIDITY_CACHE')
        for path,h in result['scan']['coverage_sources'].items():
            if digest(path)!=h:hold('HOLD_CHANGED_RENDER_SUPPORT')
    else:
        try:reason,scan=prescan.scan(run,seq_cycles);plan=coverage_plan(codes,[t-1 for t in detection_cycles],reason,allow_local_nodata=True)
        except ValueError as error:hold(str(error))
        out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'image_validity.npz',reason=reason,full_shape=scan['full_shape'])
        from scipy.ndimage import binary_dilation
        bits=np.sum((reason>0).reshape(2*seq_cycles,-1).astype(np.uint64)*(np.uint64(1)<<np.arange(2*seq_cycles,dtype=np.uint64))[:,None],axis=0).reshape(reason.shape[-2:])
        rois={}
        for cycle,pattern_keys in plan['supplement_pattern_keys'].items():
            region=np.ones(bits.shape,bool) if 0 in pattern_keys else binary_dilation(np.isin(bits,pattern_keys),iterations=2)
            rois['cycle_'+cycle]=region
        np.savez_compressed(out/'detection_rois.npz',**rois)
        result=dict(roi_path=str(out/'detection_rois.npz'),roi_sha256=digest(out/'detection_rois.npz'),supplemental_area_fraction={k:float(v.mean()) for k,v in rois.items()},status='READY_READOUT_POSTDECODE_AUDIT_REQUIRED',contract=contract,scan=scan,plan=plan,codebook=str(ref),n_features=len(book),feature_names=book.Gene.astype(str).tolist(),validity_sha256=digest(out/'image_validity.npz'),image_failure_pixels=int((reason>0).any(axis=(0,1)).sum()),acquired_image_failure_pixels=int(((reason>0).any(axis=(0,1))&~((reason&4)>0).all(axis=(0,1))).sum()),receipt=str(receipt))
        receipt.write_text(json.dumps(result,indent=2))
    (dest/'current_preflight.json').write_text(json.dumps({'receipt':str(receipt),'sha256':digest(receipt)},indent=2));return result


def bind_readout(run,preflight):
    run=Path(run);paths=[run/'readout/position.csv',run/'readout/intensity.csv']
    binding={'status':'READOUT_COMPLETE_AUDIT_PENDING','preflight_receipt':preflight['receipt'],'preflight_sha256':digest(preflight['receipt']),'products':{str(p):digest(p) for p in paths}}
    (run/'observation_checks/readout_binding.json').write_text(json.dumps(binding,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--ref-file',type=Path);p.add_argument('--detection-cycles',default='1,2,3,4');a=p.parse_args();print(json.dumps(run(a.run_dir,a.ref_file,[int(t) for t in a.detection_cycles.split(',')]),indent=2))

if __name__=='__main__':main()


def validate_readout_binding(run,ref_file):
    run=Path(run);path=run/'observation_checks/readout_binding.json'
    if not path.exists():raise ObservationHold('HOLD_UNBOUND_READOUT: run the pre-readout workflow or explicitly audit historical data')
    binding=json.loads(path.read_text());receipt=Path(binding['preflight_receipt'])
    if digest(receipt)!=binding['preflight_sha256']:raise ObservationHold('HOLD_CHANGED_PREFLIGHT')
    pre=json.loads(receipt.read_text());image=Path(pre['scan']['source_image'])
    if digest(ref_file)!=pre['contract']['codebook_sha256'] or [image.stat().st_size,image.stat().st_mtime_ns]!=pre['contract']['image_identity']:raise ObservationHold('HOLD_CHANGED_CODEBOOK_OR_IMAGE')
    for f,h in binding['products'].items():
        if digest(f)!=h:raise ObservationHold('HOLD_CHANGED_READOUT')
    frame=pd.read_csv(run/'readout/intensity.csv');require_finite(frame,[c for c in frame if c.startswith('cyc_')])
    return pre
