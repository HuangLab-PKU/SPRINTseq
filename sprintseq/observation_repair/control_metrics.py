"""Per-feature technical-control coverage; aggregate success never fills gaps."""
import numpy as np


def feature_control_summary(truth, predicted, retained, features):
    truth=np.asarray(truth,int);predicted=np.asarray(predicted,int);retained=np.asarray(retained,bool);features=list(features)
    if not features or len(set(features))!=len(features) or truth.shape!=predicted.shape or truth.shape!=retained.shape or truth.ndim!=1:raise ValueError('Invalid control/feature inventory')
    if np.any((truth<0)|(truth>=len(features))):raise ValueError('Reference outside panel')
    rows=[]
    for index,feature in enumerate(features):
        reference=truth==index;called=retained&(predicted==index);correct=called&reference;wrong=called&~reference;n=int(reference.sum());nc=int(called.sum());nr=int(correct.sum());ne=int(wrong.sum());retention=nr/n if n else None;disagreement=ne/nc if nc else None;reasons=[]
        if n<50:reasons.append('INSUFFICIENT_POSITIVE_REFERENCES')
        if nc<50:reasons.append('INSUFFICIENT_RETAINED_CALLS')
        if n>=50 and retention<.5:reasons.append('LOW_RECOVERY_SENSITIVITY')
        if nc and disagreement>.01:reasons.append('GENE_REFERENCE_DISAGREEMENT_ABOVE_AGGREGATE_BAND')
        rows.append({'feature':feature,'reference_rows':n,'retained_correct':nr,'called_rows':nc,'wrong_calls':ne,'retention':retention,'reference_disagreement_fraction':disagreement,'review_required':bool(reasons),'review_reasons':reasons,'biological_fdr_calibrated':False})
    return rows


def feature_review_items(control_receipt):
    items=[]
    for condition in control_receipt.get('conditions',[]):
        if 'per_feature' not in condition:raise ValueError('Control receipt lacks per-feature coverage')
        for row in condition['per_feature']:
            if row['review_required']:items.append({'condition':condition['condition'],'feature':row['feature'],'metrics':row})
    return items


def select_reference_mask(validity, run_dir, exclusion_file=None, *, known_ids=None):
    """Exclude independently flagged source regions from healthy controls only.

    This never changes measurement validity. Exclusions must bind the exact
    current transcript source and be generated before control outcomes are seen.
    """
    import json
    from pathlib import Path
    from sprintseq.observation_repair.__main__ import digest
    run=Path(run_dir);eligible=np.asarray(validity['observed']).all(axis=(1,2)).copy()
    result={'eligible':eligible,'sources':{},'metadata':{'scope':'HEALTHY_CONTROL_SELECTION_ONLY','additional_exclusions':'NOT_PROVIDED','excluded_eligible_rows':0}}
    if exclusion_file is None:return result
    path=Path(exclusion_file);record=json.loads(path.read_text());source=run/'readout/transcripts.parquet'
    if record.get('scope')!='CONTROL_REFERENCE_SELECTION_ONLY' or record.get('run_id')!=run.name.removesuffix('_processed') or digest(source)!=record.get('source_transcripts_sha256'):raise ValueError('Control exclusions do not bind current source')
    ids_path=Path(record['excluded_ids_file'])
    if digest(ids_path)!=record['excluded_ids_sha256']:raise ValueError('Control exclusion IDs changed')
    ids=np.load(ids_path,allow_pickle=False)['transcript_ids']
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(np.unique(ids))!=len(ids):raise ValueError('Invalid control exclusion IDs')
    if known_ids is not None and not np.isin(ids,known_ids).all():raise ValueError('Control exclusion IDs outside current source')
    for file,expected in record.get('selection_sources',{}).items():
        if digest(file)!=expected:raise ValueError('Control exclusion evidence changed')
    remove=np.isin(validity['transcript_ids'],ids);count=int((eligible&remove).sum());eligible&=~remove
    result['sources']={str(path):digest(path),str(ids_path):record['excluded_ids_sha256'],**record.get('selection_sources',{})};result['metadata'].update(additional_exclusions=str(path),excluded_eligible_rows=count,measurement_validity_modified=False)
    return result
