"""Source-bound exclusion of unqualified registration measurements.

This disposition preserves images and measured numeric values. It excludes only
uncertified correspondence dimensions from inference, never fits a transform or
asserts biological absence. Reason16 is not a chemical-pause hypothesis.
"""
import json
from pathlib import Path
import numpy as np
from sprintseq.observation_repair.__main__ import digest

REGISTRATION_UNQUALIFIED = np.uint8(16)
STATUS = 'REGISTRATION_EXCEPTIONS_CONSERVATIVELY_EXCLUDED'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def case_key(row):
    return int(row['cycle']), row['group'], row['tile']

def require(condition, message):
    if not condition:
        raise ValueError(message)

def native_bounds(position, shift, origin, tile_shape, full_shape, halo):
    """Enclose the renderer pixel centres plus the readout dependency halo."""
    start = np.asarray(position, float) + np.asarray(shift, float) - np.asarray(origin, float)
    low = np.floor(start).astype(int) - halo
    high = np.ceil(start + np.asarray(tile_shape) - 1).astype(int) + 1 + halo
    low = np.maximum(low, 0)
    high = np.minimum(high, full_shape)
    require(np.all(high > low), 'Registration exclusion outside the canvas')
    return [int(low[0]), int(high[0]), int(low[1]), int(high[1])]

def validate(run_dir, path, expected_sources=None):
    """Reject bare passes, omitted cases, stale evidence and edited footprints."""
    import pandas as pd
    from spatial_img_core.shift_table_view import ShiftTableView
    run, path = Path(run_dir), Path(path)
    item = read(path)
    require(item.get('version') == 1 and item.get('status') == STATUS, 'Unqualified registration disposition')
    require(item.get('scientific_accuracy_accepted') is False, 'Exclusion must retain unknown precision')
    require(item.get('run_id') == run.name.removesuffix('_processed'), 'Registration run mismatch')
    report, shift = run/'registration_report.json', run/'shift_table.csv'
    sources = {str(report):digest(report), str(shift):digest(shift)}
    require(item.get('sources') == sources, 'Stale registration disposition')
    if expected_sources is not None:
        require(sources == expected_sources, 'Preflight registration binding differs')
    rr = read(report)
    require(rr.get('shift_table_sha256') == digest(shift), 'Shift/report mismatch')
    require(rr.get('measurement', {}).get('unresolved') == [], 'Unresolved transforms cannot be excluded by this adapter')
    exceptions = rr['measurement']['quality_exceptions']
    require(exceptions and len({case_key(e) for e in exceptions}) == len(exceptions), 'Invalid native exception list')
    evidence = Path(item['evidence_receipt'])
    require(digest(evidence) == item['evidence_receipt_sha256'], 'Changed reviewed source evidence')
    er = read(evidence)
    require(er['run_id'] == item['run_id'], 'Evidence run differs')
    require(sorted(map(case_key, er['exceptions'])) == sorted(map(case_key, exceptions)), 'Evidence does not cover every exception')
    image = run/'stitched/mosaic.ome.tif'
    identity = [image.stat().st_size, image.stat().st_mtime_ns]
    require(identity == item['image_identity'] == er['image_identity'], 'Changed source image')
    for f, h in er['source_hashes'].items():
        require(digest(f) == h, 'Changed evidence source '+f)
    for f, h in er['products'].items():
        require(digest(f) == h, 'Changed evidence product '+f)
    canvas_path, geometry_path = run/'stitched/canvas.json', run/'stitched/geometry_float.csv'
    require(item['geometry_sources'] == {str(p):digest(p) for p in [canvas_path, geometry_path]}, 'Changed placement geometry')
    canvas = read(canvas_path)
    require(item['full_shape'] == canvas['shape'], 'Canvas shape mismatch')
    require(item['native_shape'] == er['native_shape'], 'Native camera extent differs')
    reader = Path(item['readout_source'])
    require(digest(reader) == item['readout_source_sha256'], 'Changed readout kernel contract')
    require(item['readout_contract'] == {'method':'tophat', 'tophat_radius':3, 'search_radius':1}, 'Unqualified readout dependency halo')
    # An opening has two radius3 operations, followed by radius1 max search.
    # One extra pixel covers subpixel rendering / coordinate rounding.
    halo = 2*3 + 1 + 1
    require(item['halo_pixels'] == halo, 'Under-specified readout footprint')
    geo = pd.read_csv(geometry_path).set_index('file')
    shifts = ShiftTableView(pd.read_csv(shift,index_col=0))
    rows = item['cases']
    require(len(rows) == len(exceptions) and sorted(map(case_key, rows)) == sorted(map(case_key, exceptions)), 'Disposition omits/duplicates native cases')
    for row in rows:
        require(row.get('action') == 'EXCLUDE_UNQUALIFIED_MEASUREMENT' and row.get('channels') == ['cy3','cy5'], 'Unsupported disposition')
        require(row['group'] == 'A' and 1 <= int(row['cycle']) <= 10, 'Unsupported registration group/protocol')
        e = next(e for e in exceptions if case_key(e) == case_key(row))
        require(e['status'] == 'INFERRED_NEIGHBORS_PRECISION_UNKNOWN', 'Unexpected exception type')
        tile = row['tile']
        sv = np.fromstring(str(shifts.get(f"cyc_{row['cycle']}_A",tile)),sep=' ')
        require(sv.shape == (2,) and np.isfinite(sv).all(), 'Invalid saved placement')
        require(np.allclose(sv,e['placement_yx'],rtol=0,atol=5.1e-7), 'Saved/native exception placement differs')
        expected = native_bounds([geo.at[tile,'y'],geo.at[tile,'x']],sv,canvas['origin'],item['native_shape'],canvas['shape'],halo)
        require(row['bounds_y0_y1_x0_x1'] == expected, 'Edited/undersized exclusion footprint')
    return item

def apply(reason, point_reason, x, y, full_shape, cases):
    """Registration uncertainty supersedes optical history inference in its scope."""
    image, point = reason.copy(), point_reason.copy()
    h, w = image.shape[-2:]
    H, W = full_shape
    px, py = np.floor(np.asarray(x)+.5), np.floor(np.asarray(y)+.5)
    for row in cases:
        t = int(row['cycle'])-1
        y0,y1,x0,x1 = row['bounds_y0_y1_x0_x1']
        a,b = max(0,int(np.floor(y0*h/H))), min(h,int(np.ceil(y1*h/H)))
        c,d = max(0,int(np.floor(x0*w/W))), min(w,int(np.ceil(x1*w/W)))
        # Preserve physical missing-support and boundary bits. Clear attenuation /
        # texture history eligibility: unknown geometry does not imply a pause.
        image[t,:,a:b,c:d] = (image[t,:,a:b,c:d] & np.uint8(12)) | REGISTRATION_UNQUALIFIED
        inside = (py>=y0)&(py<y1)&(px>=x0)&(px<x1)
        point[inside,t,:] = (point[inside,t,:] & np.uint8(12)) | REGISTRATION_UNQUALIFIED
    return image, point

def prepare_qc(run_dir, base_qc, output, disposition):
    """Compose a new immutable generation; leave the base QC/data untouched."""
    from spatial_img_core.observation_qc.points import run as image_qc
    run, base, out, disposition = map(Path,[run_dir,base_qc,output,disposition])
    item = validate(run,disposition)
    if out.exists():
        r = read(out/'receipt.json')
        require(r['registration_disposition_sha256'] == digest(disposition), 'Changed registration exclusion checkpoint')
        for p,h in {**r['source_code'],**r['coverage_sources']}.items():
            require(digest(p)==h,'Changed validity source '+p)
        require(digest(out/'point_validity.npz')==r['point_validity_sha256'] and digest(out/'image_validity.npz')==r['image_validity_sha256'],'Changed validity masks')
        return r
    if not base.exists():
        image_qc(run,base)
    r = read(base/'receipt.json')
    require(r['run_id']==item['run_id'] and r['source_tiff_identity']==item['image_identity'], 'QC source frame differs')
    require(digest(run/'readout/transcripts.parquet')==r['source_transcripts_sha256'], 'Changed baseline transcripts')
    for p,h in {**r['source_code'],**r['coverage_sources']}.items():
        require(digest(p)==h,'Changed base validity source '+p)
    require(digest(base/'point_validity.npz')==r['point_validity_sha256'] and digest(base/'image_validity.npz')==r['image_validity_sha256'],'Changed base masks')
    with np.load(base/'image_validity.npz',allow_pickle=False) as iv, np.load(base/'point_validity.npz',allow_pickle=False) as pv:
        values={k:pv[k] for k in pv.files}
        reason, point = apply(iv['reason'],values['reason'],values['x_px'],values['y_px'],r['full_shape'],item['cases'])
    out.mkdir(parents=True,exist_ok=False)
    values.update(reason=point,observed=point==0)
    np.savez_compressed(out/'point_validity.npz',**values)
    np.savez_compressed(out/'image_validity.npz',reason=reason,full_shape=np.asarray(r['full_shape']))
    r.update(point_validity_sha256=digest(out/'point_validity.npz'),image_validity_sha256=digest(out/'image_validity.npz'),registration_disposition=str(disposition),registration_disposition_sha256=digest(disposition),base_validity_receipt=str(base/'receipt.json'),base_validity_receipt_sha256=digest(base/'receipt.json'))
    r['source_code'][str(Path(__file__))]=digest(__file__)
    r['coverage_sources'].update({str(disposition):digest(disposition),str(base/'receipt.json'):digest(base/'receipt.json'),str(base/'point_validity.npz'):digest(base/'point_validity.npz'),str(base/'image_validity.npz'):digest(base/'image_validity.npz')})
    r['reason_bits']['REGISTRATION_UNQUALIFIED']=int(REGISTRATION_UNQUALIFIED)
    r['registration_exclusion_cases']=item['cases']
    r['registration_unqualified_transcripts']=int(((point&16)>0).any(axis=(1,2)).sum())
    r['affected_transcripts']=int((point>0).any(axis=(1,2)).sum())
    r['all_unobserved_transcripts']=int((point>0).all(axis=(1,2)).sum())
    r['point_summary']=[{'cycle':t+1,'cy3_invalid':int((point[:,t,0]>0).sum()),'cy5_invalid':int((point[:,t,1]>0).sum()),'both_invalid':int((point[:,t,:]>0).all(axis=1).sum()),'registration_unqualified':int(((point[:,t,:]&16)>0).any(axis=1).sum()),'source_missing':int(((point[:,t,:]&4)>0).any(axis=1).sum())} for t in range(point.shape[1])]
    r['limitations'] += ['Reason16 excludes uncertified cross-round correspondence; it does not prove physical source absence or biological zero.','Raw intensity values remain preserved but excluded dimensions do not enter likelihoods or chemical-pause eligibility.','Footprints conservatively include overlapping measurements and the readout halo; image-level lookup has the recorded pyramid resolution.','All-unobserved/code-collision calls remain UnknownObservation; reliable dimensions alone may support partial-observation calls.']
    (out/'receipt.json').write_text(json.dumps(r,indent=2),encoding='utf8')
    return r

def create_disposition(run_dir, evidence_receipt, output):
    """Account for every native exception by exclusion, never by accuracy waiver."""
    import pandas as pd
    import inspect
    from sprintseq.cli import readout
    from sprintseq.readout import intensity_readout
    from spatial_img_core.shift_table_view import ShiftTableView
    run, evidence, output = map(Path,[run_dir,evidence_receipt,output])
    require(not output.exists(), 'Do not replace an existing disposition')
    er = read(evidence)
    require(er['status']=='REGISTRATION_EXCEPTION_EVIDENCE_REQUIRES_REVIEW' and not er['scientific_gate_cleared'], 'Expected retained unqualified source evidence')
    require(readout.TOPHAT_RADIUS==3 and readout.SEARCH_RADIUS==1, 'Unqualified readout kernel footprint')
    report, shift = run/'registration_report.json', run/'shift_table.csv'
    rr = read(report)
    geometry, canvas_path = run/'stitched/geometry_float.csv', run/'stitched/canvas.json'
    geo, canvas = pd.read_csv(geometry).set_index('file'), read(canvas_path)
    shifts = ShiftTableView(pd.read_csv(shift,index_col=0))
    cases=[]
    for e in rr['measurement']['quality_exceptions']:
        tile=e['tile']; sv=np.fromstring(str(shifts.get(f"cyc_{e['cycle']}_{e['group']}",tile)),sep=' ')
        cases.append(dict(cycle=int(e['cycle']),group=e['group'],tile=tile,channels=['cy3','cy5'],action='EXCLUDE_UNQUALIFIED_MEASUREMENT',bounds_y0_y1_x0_x1=native_bounds([geo.at[tile,'y'],geo.at[tile,'x']],sv,canvas['origin'],er['native_shape'],canvas['shape'],8),precision_measured=False))
    kernel=Path(inspect.getfile(intensity_readout))
    item=dict(version=1,status=STATUS,run_id=run.name.removesuffix('_processed'),scientific_accuracy_accepted=False,artifact_free_claim=False,sources={str(p):digest(p) for p in [report,shift]},geometry_sources={str(p):digest(p) for p in [canvas_path,geometry]},evidence_receipt=str(evidence),evidence_receipt_sha256=digest(evidence),image_identity=er['image_identity'],full_shape=canvas['shape'],native_shape=er['native_shape'],readout_source=str(kernel),readout_source_sha256=digest(kernel),readout_contract={'method':'tophat','tophat_radius':3,'search_radius':1},halo_pixels=8,cases=cases,disposition='Retain the saved inferred image placement and its unknown precision. Exclude its conservative rendered footprint plus the native readout halo from corresponding SBS likelihood dimensions. No image, transform, measured intensity or cell mask is edited.',limitations=['This is a conservative uncertainty disposition, not measured registration qualification.','Overlapping valid donor measurements may also be excluded; no missing expression is filled.','Raw pixels and intensities remain available for audit; no biological zero or chemical pause is inferred.','Molecular controls, all-feature review and final acceptance remain required.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as f:json.dump(item,f,indent=2)
    return validate(run,output)
