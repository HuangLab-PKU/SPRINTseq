"""Audit all panel genes against image-defined failure regions after decoding.

Local biological heterogeneity can trigger this screen. It proposes review and
recovery; it never converts a density difference into a corrected molecular call.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import fisher_exact
from .preflight import ObservationHold
from sprintseq.observation_repair.__main__ import digest


def bh_adjust(values):
    p=np.asarray(values,float);order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];out=np.empty_like(q);out[order]=np.minimum(q,1);return out


def screen_region(genes,inside_names,ring_names,n_inside,n_ring):
    rows=[]
    for gene in genes:
        a=int(inside_names.get(gene,0));b=int(ring_names.get(gene,0));expected=n_inside*(b+.5)/(n_ring+1);ratio=(a+.5)/(expected+.5)
        p=float(fisher_exact([[a,n_inside-a],[b,n_ring-b]]).pvalue) if a+b else 1.
        rows.append(dict(gene=gene,inside_Q20=a,control_Q20=b,inside_spots=n_inside,control_spots=n_ring,expected_from_control=expected,rate_ratio=ratio,p_value=p))
    return rows


def run(run_dir,ref_file,output=None,preflight=None,readout_dir=None):
    run=Path(run_dir);readout=Path(readout_dir) if readout_dir is not None else run/'readout';ref=Path(ref_file);out=Path(output) if output else run/'observation_checks/post_decode';out.mkdir(parents=True,exist_ok=True)
    if preflight is None:
        pointer=json.loads((run/'observation_checks/current_preflight.json').read_text());receipt=Path(pointer['receipt'])
        if digest(receipt)!=pointer['sha256']:raise ObservationHold('HOLD_CHANGED_PREFLIGHT')
    else:receipt=Path(preflight)
    pre=json.loads(receipt.read_text());image=Path(pre['scan']['source_image'])
    if digest(ref)!=pre['contract']['codebook_sha256'] or [image.stat().st_size,image.stat().st_mtime_ns]!=pre['contract']['image_identity']:raise ObservationHold('HOLD_AUDIT_SOURCE_MISMATCH')
    validity=receipt.parent/'image_validity.npz'
    if digest(validity)!=pre['validity_sha256']:raise ObservationHold('HOLD_CHANGED_IMAGE_VALIDITY')
    masks=np.load(validity);reason=masks['reason'];H,W=masks['full_shape'];footprint=~((reason&4)>0).all(axis=(0,1));bad=(reason>0).any(axis=(0,1))&footprint;labels,n=ndi.label(bad)
    positions=pd.read_csv(readout/'position.csv');calls=pd.read_csv(readout/'mapping_postcode.csv')
    if positions['index'].duplicated().any() or calls['index'].duplicated().any() or set(positions['index'])!=set(calls['index']):raise ObservationHold('HOLD_DECODE_ID_MISMATCH')
    frame=positions.merge(calls[['index','Gene','Probability']],on='index',validate='one_to_one');prob=frame.Probability.to_numpy(float)
    if not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):raise ObservationHold('HOLD_INVALID_DECODE_CONFIDENCE')
    x=frame.X.to_numpy();y=frame.Y.to_numpy()
    if not np.isfinite(x+y).all() or np.any((x<0)|(x>=W)|(y<0)|(y>=H)):raise ObservationHold('HOLD_INVALID_COORDINATES')
    py=np.minimum((y*bad.shape[0]/H).astype(int),bad.shape[0]-1);px=np.minimum((x*bad.shape[1]/W).astype(int),bad.shape[1]-1);point_region=labels[py,px];pop=np.bincount(point_region,minlength=n+1);genes=pd.read_csv(ref).Gene.astype(str).tolist();names=frame.Gene.astype(str).to_numpy();q=(prob>.99)&np.isin(names,genes);rows=[];regions=[]
    objects=ndi.find_objects(labels)
    for comp in np.flatnonzero(pop>=100):
        if comp==0:continue
        sl=objects[comp-1];area=int((labels[sl]==comp).sum())
        if area<64:continue
        pad=max(8,int(np.sqrt(area)));ys=slice(max(0,sl[0].start-pad),min(bad.shape[0],sl[0].stop+pad));xs=slice(max(0,sl[1].start-pad),min(bad.shape[1],sl[1].stop+pad))
        distance=ndi.distance_transform_edt(labels[ys,xs]!=comp);ringmask=(distance>0)&(distance<=pad)&~bad[ys,xs];near=(py>=ys.start)&(py<ys.stop)&(px>=xs.start)&(px<xs.stop);control=np.zeros(len(frame),bool);ix=np.flatnonzero(near);control[ix]=ringmask[py[ix]-ys.start,px[ix]-xs.start];inside=point_region==comp
        component_reason=reason[:,:,labels==comp];cycles=np.flatnonzero((component_reason>0).any(axis=(1,2)))+1
        source_missing=bool(((component_reason&4)>0).any());dual=bool((component_reason>0).all(axis=1).any())
        kind='SOURCE_MISSING' if source_missing else ('IMAGE_FAILURE_HISTORY_UNRESOLVED' if dual else 'OPTICAL_OR_CHANNEL_FAILURE')
        region=dict(region=int(comp),kind=kind,cycles=cycles.tolist(),inside_spots=int(inside.sum()),control_spots=int(control.sum()),inside_Q20=int((inside&q).sum()),bbox_pyramid=[sl[0].start,sl[0].stop,sl[1].start,sl[1].stop],normal_control_adequate=bool(control.sum()>=100));regions.append(region)
        a=pd.Series(names[inside&q]).value_counts();b=pd.Series(names[control&q]).value_counts()
        for row in screen_region(genes,a,b,int(inside.sum()),int(control.sum())):row.update(region=int(comp),control_adequate=bool(control.sum()>=100));rows.append(row)
    table=pd.DataFrame(rows)
    if len(table):
        table['screen_q_value']=bh_adjust(table.p_value);table['flag']=table.control_adequate&(table.control_Q20>=20)&(table.screen_q_value<=.01)&((table.rate_ratio<.75)|(table.rate_ratio>1.5))&(np.abs(table.inside_Q20-table.expected_from_control)>=10)
    else:table=pd.DataFrame(columns=['region','gene','inside_Q20','control_Q20','rate_ratio','screen_q_value','flag'])
    table.to_csv(out/'all_feature_region_audit.csv',index=False)
    counts=pd.Series(names[q]).value_counts().reindex(genes,fill_value=0);counts.rename_axis('gene').rename('Q20').to_csv(out/'all_feature_counts.csv')
    residual_path=readout/'observation_limitations.json';residual=json.loads(residual_path.read_text()) if residual_path.exists() else None
    outside_acquisition_points=int((~footprint[py,px]).sum())
    affected=any(r['inside_Q20']>=20 for r in regions);flagged=table.loc[table.flag.astype(bool)].to_dict('records')
    spatial_nodata=pre['plan'].get('unobservable_feature_patterns',[])
    repair_scope=pre['plan'].get('repair_scope_exceeded',[])
    geometry=pre['contract'].get('registration',{})
    hold=bool(affected or flagged or residual or spatial_nodata or repair_scope or geometry.get('quality_exceptions',0) or outside_acquisition_points)
    actions=[]
    if any(r['kind']=='SOURCE_MISSING' for r in regions):actions.append('VERIFY_ACQUISITION_CENSUS_AND_EXACT_POINT_COVERAGE')
    if outside_acquisition_points:actions.append('RETAIN_POINTS_WITHOUT_SEQUENCING_SUPPORT_AS_NODATA')
    if affected:actions.append('QUALIFY_OPTICAL_VS_REACTION_HISTORY_BEFORE_ACCEPTANCE')
    if pre['plan']['added_cycles']:actions.append('VERIFY_EFFECTIVE_DETECTION_CYCLES_AND_NEW_POINT_RECALL')
    if spatial_nodata:actions.append('KEEP_UNOBSERVABLE_FEATURE_REGIONS_AS_NODATA_EVEN_WITH_ZERO_DETECTED_POINTS')
    if repair_scope:actions.append('HOLD_UNQUALIFIED_HISTORY_PATTERNS_FOR_SEPARATE_RECOVERY')
    if geometry.get('quality_exceptions',0):actions.append('QUALIFY_RECORDED_REGISTRATION_EXCEPTIONS')
    if residual:actions.append('KEEP_UNRESOLVED_FEATURES_FLAGGED_NO_DENSITY_IMPUTATION')
    report=dict(status='HOLD_ARTIFACT_REVIEW_REQUIRED' if hold else 'PANEL_AUDIT_PASS_WITH_REPORTED_COVERAGE_LIMITS',run_id=pre['contract']['run_id'],features_audited=len(genes),outside_acquisition_points=outside_acquisition_points,rows=len(frame),regions=regions,flagged_gene_regions=flagged,known_residual=residual,spatial_nodata=spatial_nodata,repair_scope_exceeded=repair_scope,registration=geometry,next_actions=actions,preflight_receipt=str(receipt),preflight_sha256=digest(receipt),source_hashes={str(f):digest(f) for f in [readout/'position.csv',readout/'mapping_postcode.csv',ref,Path(__file__)]},limitations=['Region/control density differences can be biological; flags are triage, never automatic gene edits.','No universal biological FDR or artifact-free certificate.','All feature rows are emitted, including zero counts and unflagged genes.','Registration and precise source coverage remain independent acquisition/image gates.'])
    (out/'receipt.json').write_text(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--ref-file',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--preflight',type=Path);p.add_argument('--report-only',action='store_true');a=p.parse_args();result=run(a.run_dir,a.ref_file,a.output,a.preflight);print(json.dumps({'status':result['status'],'features_audited':result['features_audited'],'regions':len(result['regions']),'flagged_gene_regions':len(result['flagged_gene_regions'])}),flush=True)
    if result['status'].startswith('HOLD') and not a.report_only:raise SystemExit(2)

if __name__=='__main__':main()
