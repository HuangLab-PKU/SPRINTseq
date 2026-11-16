"""Mask-independent whole-field feature screening and explicit review coverage.

Screens nominate source-image review. They never infer an artifact mechanism,
change counts, or treat a quiet statistical screen as biological acceptance.
"""
from sprintseq.check_paths import checks_directory
from itertools import product
import hashlib,json
import numpy as np
from scipy import ndimage as ndi

SCREEN_POLICY = 'whole_field_morphology_supported_v1'
REVIEW_STATES = {'NOT_REVIEWED', 'REVIEW_REQUIRED', 'INSUFFICIENT_EVIDENCE',
                 'CHECKED_WITH_LIMITATIONS', 'SUPPORTED_CORRECTION'}


def _box_sum(values, radius):
    width = 2 * radius + 1
    size = (1, width, width) if values.ndim == 3 else (width, width)
    return ndi.uniform_filter(np.asarray(values, dtype=float), size=size,
                              mode='constant') * width * width


def screen(counts, features, morphology_fraction, *, image_failure=None,
           radii=(2, 4, 8), min_expected=15., min_difference=12.):
    """Inspect every feature everywhere with sufficient independent tissue support.

    The morphology plane must be source-bound to the same native grid. Missing
    gene planes are invalid, not zero. Image failure masks annotate findings but
    do not select the spatial search area. Sparse features remain insufficient.
    Thresholds are triage effect sizes, not p-values or biological FDR estimates.
    """
    counts = np.asarray(counts)
    tissue = np.asarray(morphology_fraction, float)
    features = list(features)
    if counts.ndim != 3 or len(features) != len(counts) or len(set(features)) != len(features):
        raise ValueError('Exactly one plane per unique panel feature is required')
    if counts.shape[1:] != tissue.shape or not np.isfinite(tissue).all() or np.any((tissue < 0) | (tissue > 1)):
        raise ValueError('Invalid independent morphology support')
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('Missing/invalid counts require an explicit separate coverage record')
    if image_failure is not None and np.asarray(image_failure).shape != tissue.shape:
        raise ValueError('Image failure grid differs')
    if not radii or any(type(r) is not int or r < 1 for r in radii):
        raise ValueError('Positive integer radii required')
    low = np.zeros(counts.shape, bool); high = np.zeros(counts.shape, bool)
    score = np.zeros(counts.shape, np.float32)
    eligible = np.zeros(counts.shape, bool)
    for radius in radii:
        inner = _box_sum(counts, radius)
        ring = np.maximum(0., _box_sum(counts, 2 * radius) - inner)
        area = _box_sum(tissue, radius)
        ring_area = np.maximum(0., _box_sum(tissue, 2 * radius) - area)
        inner_capacity = (2 * radius + 1) ** 2
        ring_capacity = (4 * radius + 1) ** 2 - inner_capacity
        supported = (area >= .35 * inner_capacity) & (ring_area >= .35 * ring_capacity)
        expected = ring * np.divide(area, ring_area, out=np.zeros_like(area), where=ring_area > 0)
        tested = supported[None] & (ring >= 20) & (expected >= min_expected)
        eligible |= tested
        deficit = expected - inner
        excess = inner - expected
        lo = tested & (inner <= .3 * expected) & (deficit >= min_difference)
        hi = tested & (inner >= 3 * expected) & (excess >= max(20., min_difference))
        low |= lo; high |= hi
        score = np.maximum(score, np.where(lo, deficit, np.where(hi, excess, 0.))).astype(np.float32)
    rows = []; findings = []
    for i, gene in enumerate(features):
        gene_findings = []
        for kind, mask in [('DEPLETION', low[i]), ('ENRICHMENT', high[i])]:
            labels, _ = ndi.label(mask)
            for component, box in enumerate(ndi.find_objects(labels), 1):
                if box is None:continue
                component_mask = labels[box] == component
                candidate_scores = np.where(component_mask, score[i][box], -1)
                y, x = np.unravel_index(np.argmax(candidate_scores), candidate_scores.shape)
                y += box[0].start; x += box[1].start
                item = {'feature': gene, 'kind': kind,
                        'bbox_y0_y1_x0_x1': [box[0].start, box[0].stop, box[1].start, box[1].stop],
                        'peak_yx': [int(y), int(x)], 'max_count_difference': float(score[i, y, x]),
                        'screen_pixels': int(component_mask.sum()),
                        'image_mask_overlap': None if image_failure is None else
                        bool(np.asarray(image_failure)[box][component_mask].any()),
                        'review_state': 'REVIEW_REQUIRED', 'mechanism_inferred': False}
                item['finding_id']='region_'+hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()[:24]
                gene_findings.append(item);findings.append(item)
        n_tested = int(eligible[i].sum())
        rows.append({'feature': gene, 'total_q20': int(counts[i].sum()),
                     'whole_field_visited': True, 'tested_centres': n_tested,
                     'tissue_centres': int((tissue >= .35).sum()),
                     'findings': len(gene_findings),
                     'screen_state': 'ALERT' if gene_findings else
                     ('SCREENED_NO_ALERT' if n_tested else 'INSUFFICIENT_SUPPORT'),
                     'review_state': 'REVIEW_REQUIRED' if gene_findings else
                     ('NOT_REVIEWED' if n_tested else 'INSUFFICIENT_EVIDENCE')})
    total = counts.sum(axis=0)
    tissue_without_molecular_signal = int(((tissue >= .35) & (total == 0)).sum())
    return {'policy': SCREEN_POLICY, 'features': rows, 'findings': findings,
            'tissue_centres_without_any_q20': tissue_without_molecular_signal,
            'all_feature_planes_visited': True, 'release_accepted': False,
            'limitations': ['Morphology-supported density differences may be biological.',
                           'Sparse/zero features and insufficient controls are not clean passes.',
                           'Image masks annotate findings; they never restrict this screen.',
                           'Finite spatial scales cannot prove absence of every artifact.']}


def cohort_screen(feature_counts, section_ids, features, *, ratio_limit=.2, min_peer_count=20):
    """Nominate section-wide gene deficits; adjacent biology is never a correction."""
    x = np.asarray(feature_counts, float)
    if x.shape != (len(section_ids), len(features)) or len(set(section_ids)) != len(section_ids) or len(set(features))!=len(features):
        raise ValueError('Incomplete/duplicate section-feature inventory')
    if not np.isfinite(x).all() or np.any(x < 0):raise ValueError('Invalid cohort counts')
    totals = x.sum(axis=1);shares = np.divide(x, totals[:, None], out=np.zeros_like(x), where=totals[:, None] > 0)
    alerts = []
    if len(x) < 3:return alerts
    for i, section in enumerate(section_ids):
        peers = np.arange(len(x)) != i
        if totals[i]==0:
            alerts.extend({'section':section,'feature':gene,'kind':'WHOLE_SECTION_NO_Q20','count':0,'peer_scaled_expectation':float(np.median(x[peers,j])),'review_state':'REVIEW_REQUIRED','mechanism_inferred':False} for j,gene in enumerate(features))
            continue
        expected = np.median(shares[peers], axis=0) * totals[i]
        adequate = np.median(x[peers], axis=0) >= min_peer_count
        for j in np.flatnonzero(adequate & (expected >= min_peer_count) & (x[i] < ratio_limit * expected)):
            alerts.append({'section': section, 'feature': features[j], 'kind': 'SECTION_WIDE_DEFICIT',
                           'count': int(x[i, j]), 'peer_scaled_expectation': float(expected[j]),
                           'review_state': 'REVIEW_REQUIRED', 'mechanism_inferred': False})
    return alerts


def validate_coverage(rows, section_ids, features, *, require_review=False):
    """Fail closed on missing, duplicate, unknown or unreviewed coverage entries."""
    section_ids=list(section_ids);features=list(features)
    if not section_ids or not features or len(set(section_ids))!=len(section_ids) or len(set(features))!=len(features):raise ValueError('Nonempty unique section and feature inventories required')
    wanted = set(product(section_ids, features));seen = set()
    for row in rows:
        key = (row['section'], row['feature'])
        if key not in wanted or key in seen:raise ValueError('Unexpected or duplicate coverage entry')
        seen.add(key)
        if not row.get('whole_field_visited') or row.get('review_state') not in REVIEW_STATES:
            raise ValueError('Missing screen coverage or unknown review state')
        if require_review:
            if row['review_state'] not in {'CHECKED_WITH_LIMITATIONS', 'SUPPORTED_CORRECTION'}:
                raise ValueError('Unresolved or unreviewed section-feature pair')
            if not row.get('evidence') or not row.get('limitations'):
                raise ValueError('Review requires actual evidence and explicit limitations')
    if seen != wanted:raise ValueError('Missing section-feature pairs')
    return {'pairs': len(seen), 'screen_coverage_complete': True,
            'review_complete': bool(require_review), 'artifact_free_claim': False}


def audit_run(run_dir, ref_file, output, *, readout_dir=None):
    """Read-only fresh audit of native transcripts using bound cell-mask area.

    Caller must own the shared I/O lane. No decode, activation or image mutation.
    Missing independent support produces explicit insufficient evidence, not PASS.
    """
    import json
    from pathlib import Path
    import pandas as pd
    import pyarrow.parquet as pq
    import tifffile
    from spatial_cells.transcripts import read_footer
    from sprintseq.observation_repair.__main__ import digest
    run, ref, out = Path(run_dir), Path(ref_file), Path(output)
    if out.exists():raise ValueError('Immutable full audit already exists')
    readout=Path(readout_dir) if readout_dir is not None else run/'readout'
    source = readout/'transcripts.parquet';footer = read_footer(source)
    reference_image=Path(footer['reference_image']);image_identity=[reference_image.stat().st_size,reference_image.stat().st_mtime_ns]
    observation_contract=run/'readout/observation_contract.json'
    if observation_contract.exists() and json.loads(observation_contract.read_text()).get('source_image_identity')!=image_identity:raise ValueError('Native reference image changed')
    genes = pd.read_csv(ref).Gene.astype(str).tolist()
    if len(set(genes)) != len(genes):raise ValueError('Duplicate panel feature')
    full = tuple(footer['image_shape_px'])
    if len(full)!=2:raise ValueError('Whole-field review currently requires a 2D native frame')
    shape = (full[0]//200+1, full[1]//200+1)
    counts = np.zeros((len(genes),*shape),np.int64);index = pd.Index(genes)
    bindings = {str(source):digest(source),str(ref):digest(ref)}
    for batch in pq.ParquetFile(source).iter_batches(columns=['x_px','y_px','feature_name','is_gene','probability']):
        frame=batch.to_pandas();probability=frame.probability.to_numpy(float);coordinates=frame[['y_px','x_px']].to_numpy(float)
        if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)) or not np.isfinite(coordinates).all() or np.any(coordinates<0) or np.any(coordinates>=np.asarray(full)):raise ValueError('Invalid source probability or coordinates')
        pick=frame.is_gene.to_numpy(bool)&(probability>.99)
        gi=index.get_indexer(frame.loc[pick,'feature_name'].astype(str));x=np.floor(frame.loc[pick,'x_px'].to_numpy()+.5).astype(int)//200;y=np.floor(frame.loc[pick,'y_px'].to_numpy()+.5).astype(int)//200
        if np.any(gi<0) or np.any(x<0) or np.any(y<0) or np.any(x>=shape[1]) or np.any(y>=shape[0]):raise ValueError('Invalid target/frame')
        np.add.at(counts,(gi,y,x),1)
    support = np.zeros(shape,float);support_status='UNAVAILABLE_NOT_ASSUMED_NORMAL'
    active = run/'readout/active_revision.json'
    if active.exists():
        marker=json.loads(active.read_text());receipt=Path(marker['activation_attempt'])/'receipt.json';accepted=json.loads(receipt.read_text());bindings[str(receipt)]=digest(receipt);bindings[str(active)]=digest(active)
        products=accepted['canonical_products'];expected=products['readout']['transcripts.parquet']
        if readout==run/'readout':
            if bindings[str(source)]!=expected:raise ValueError('Active transcript bytes changed')
        else:
            candidate_receipt=readout.parent/'receipt.json';candidate=json.loads(candidate_receipt.read_text())
            if candidate['transcripts_sha256']!=bindings[str(source)]:raise ValueError('Candidate transcript bytes changed')
            bindings[str(candidate_receipt)]=digest(candidate_receipt)
        for name in ['cellsam_mask.tif','cellpose_mask.tif','nuclei_mask.tif']:
            mask_path=run/'segmented'/name
            if name not in products['segmented'] or not mask_path.exists():continue
            if digest(mask_path)!=products['segmented'][name]:raise ValueError('Independent cell-mask binding changed')
            bindings[str(mask_path)]=products['segmented'][name]
            try:mask=tifffile.memmap(mask_path)
            except ValueError:mask=tifffile.imread(mask_path)
            if mask.shape!=full:raise ValueError('Independent cell-mask frame mismatch')
            starts=np.arange(0,full[1],200);widths=np.minimum(200,full[1]-starts)
            for y0 in range(0,full[0],200):
                strip=np.asarray(mask[y0:min(y0+200,full[0])])>0
                totals=np.add.reduceat(strip.sum(axis=0,dtype=np.int64),starts)
                support[y0//200,:len(starts)]=totals/(strip.shape[0]*widths)
            del mask;support_status='CURRENT_SOURCE_BOUND_CELL_MASK_AREA';break
    # New datasets need not be activated to supply independent image support.
    support_receipt=checks_directory(run)/'morphology_support.json'
    if support_status=='UNAVAILABLE_NOT_ASSUMED_NORMAL' and support_receipt.exists():
        record=json.loads(support_receipt.read_text());mask_path=Path(record['mask']);image=Path(record['source_image'])
        if record.get('schema')!='independent_morphology_support.v1' or record.get('run_id')!=footer['run_id'] or tuple(record['full_shape_px'])!=full or float(record['pixel_size_um'])!=float(footer['pixel_size_um']):raise ValueError('Independent morphology metadata differs')
        if Path(footer['reference_image']).resolve()!=image.resolve() or [image.stat().st_size,image.stat().st_mtime_ns]!=record['source_image_identity'] or digest(mask_path)!=record['mask_sha256']:raise ValueError('Independent morphology source differs')
        with tifffile.TiffFile(image) as tf:
            if not any(tuple(series.shape[-2:])==full for series in tf.series):raise ValueError('Morphology image native frame differs')
        try:mask=tifffile.memmap(mask_path)
        except ValueError:mask=tifffile.imread(mask_path)
        if mask.shape!=full:raise ValueError('Independent morphology mask frame differs')
        starts=np.arange(0,full[1],200);widths=np.minimum(200,full[1]-starts)
        for y0 in range(0,full[0],200):
            strip=np.asarray(mask[y0:min(y0+200,full[0])])>0;totals=np.add.reduceat(strip.sum(axis=0,dtype=np.int64),starts);support[y0//200,:len(starts)]=totals/(strip.shape[0]*widths)
        del mask;bindings[str(mask_path)]=record['mask_sha256'];bindings[str(support_receipt)]=digest(support_receipt);support_status='SOURCE_BOUND_MORPHOLOGY_MASK_AREA'
    verified_density=0
    for i,gene in enumerate(genes):
        density=readout/'density_Q20'/(gene+'.tif')
        if density.exists():
            if not np.array_equal(tifffile.imread(density),counts[i]):raise ValueError('Density does not match current transcripts: '+gene)
            verified_density+=1
    result=screen(counts,genes,support)
    for row in result['features']:
        row.update(section=footer['run_id'],evidence=[],limitations=result['limitations'])
    result.update(run_id=footer['run_id'],source_image=str(reference_image),source_image_identity=image_identity,source_transcripts=str(source),source_bindings=bindings,morphology_support_status=support_status,density_planes_verified=verified_density,entry_sha256=digest(__file__),
                  coverage=validate_coverage(result['features'],[footer['run_id']],genes),
                  status='FULL_FIELD_SCREEN_COMPLETE_REVIEW_REQUIRED')
    for f,h in bindings.items():
        if digest(f)!=h:raise ValueError('Source changed during full audit')
    if [reference_image.stat().st_size,reference_image.stat().st_mtime_ns]!=image_identity:raise ValueError('Native image changed during full audit')
    out.mkdir(parents=True);(out/'receipt.json').write_text(json.dumps(result,indent=2));(out/'coverage_ledger.json').write_text(json.dumps(result['features'],indent=2))
    return result
