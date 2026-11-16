"""Source-bound all-feature review gate, distinct from numerical reconstruction."""
import hashlib,json
from pathlib import Path
from .preflight import ObservationHold
from .whole_field import SCREEN_POLICY,validate_coverage
from sprintseq.observation_repair.__main__ import digest


def review_items(audit, full):
    items={}
    for item in full.get('findings',[]):
        key=item.get('finding_id')
        if not key or key in items:raise ObservationHold('HOLD_INVALID_FULL_FIELD_FINDING_IDS')
        items[key]=item
    for name in ['flagged_gene_regions','flagged_gene_regions_before','flagged_gene_regions_after','unassessed_small_failure_patterns','repair_scope_exceeded','spatial_nodata','feature_control_review']:
        for item in audit.get(name,[]):
            key=name+'_'+hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()[:24];items[key]=item
    if audit.get('known_residual'):
        item=audit['known_residual'];key='known_residual_'+hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()[:24];items[key]=item
    return items


def require_review(path, run_id, features, audit_path, revision_receipt):
    path=Path(path)
    if not path.is_file():raise ObservationHold('HOLD_FULL_PANEL_REVIEW_REQUIRED: '+str(path))
    proof=json.loads(path.read_text())
    if proof.get('run_id')!=run_id or proof.get('audit_sha256')!=digest(audit_path) or proof.get('revision_receipt_sha256')!=digest(revision_receipt):
        raise ObservationHold('HOLD_STALE_FULL_PANEL_REVIEW')
    audit=json.loads(Path(audit_path).read_text());revision=json.loads(Path(revision_receipt).read_text())
    for file,expected in audit.get('control_review_sources',{}).items():
        if not Path(file).is_file() or digest(file)!=expected:raise ObservationHold('HOLD_CHANGED_FEATURE_CONTROL_EVIDENCE')
    if audit.get('source_post_audit') and digest(audit['source_post_audit'])!=audit.get('source_post_audit_sha256'):raise ObservationHold('HOLD_CHANGED_POST_RECOVERY_AUDIT')
    full_path=Path(proof.get('whole_field_receipt',''))
    if not full_path.is_file() or digest(full_path)!=proof.get('whole_field_receipt_sha256'):
        raise ObservationHold('HOLD_FULL_FIELD_RECEIPT_REQUIRED')
    full=json.loads(full_path.read_text())
    if full.get('run_id')!=run_id or full.get('policy')!=SCREEN_POLICY or full.get('morphology_support_status') not in {'CURRENT_SOURCE_BOUND_CELL_MASK_AREA','SOURCE_BOUND_MORPHOLOGY_MASK_AREA'}:
        raise ObservationHold('HOLD_UNVERIFIED_WHOLE_FIELD_SUPPORT')
    try:
        validate_coverage(full['features'],[run_id],features)
        coverage=validate_coverage(proof['ledger'],[run_id],features,require_review=True)
    except (KeyError,ValueError) as error:raise ObservationHold('HOLD_INCOMPLETE_FULL_PANEL_REVIEW: '+str(error)) from error
    image=Path(full.get('source_image',''))
    if not image.is_file() or [image.stat().st_size,image.stat().st_mtime_ns]!=full.get('source_image_identity'):raise ObservationHold('HOLD_CHANGED_FULL_FIELD_IMAGE')
    bindings=full.get('source_bindings',{});source=full.get('source_transcripts')
    if not source or bindings.get(source)!=revision.get('transcripts_sha256'):
        raise ObservationHold('HOLD_FULL_FIELD_REVISION_SOURCE_MISMATCH')
    for file,expected in bindings.items():
        if not Path(file).is_file() or digest(file)!=expected:raise ObservationHold('HOLD_CHANGED_FULL_FIELD_SOURCE')
    evidence=proof.get('evidence_hashes',{})
    def check_evidence(row):
        if not row.get('evidence') or not row.get('limitations'):raise ObservationHold('HOLD_MISSING_ITEM_REVIEW_EVIDENCE')
        for item in row['evidence']:
            if item not in evidence or not Path(item).is_file() or digest(item)!=evidence[item]:raise ObservationHold('HOLD_CHANGED_REVIEW_EVIDENCE')
    for row in proof['ledger']:check_evidence(row)
    expected_items=review_items(audit,full);dispositions=proof.get('finding_dispositions',{})
    if set(dispositions)!=set(expected_items):raise ObservationHold('HOLD_UNREVIEWED_OR_STALE_FINDINGS')
    for item in dispositions.values():
        if item.get('review_state') not in {'CHECKED_WITH_LIMITATIONS','SUPPORTED_CORRECTION'}:raise ObservationHold('HOLD_UNRESOLVED_FINDING')
        check_evidence(item)
    if proof.get('artifact_free_claim',False):raise ObservationHold('HOLD_UNSUPPORTED_ARTIFACT_FREE_CLAIM')
    return {'status':'ALL_FEATURE_REVIEW_COMPLETE_WITH_EXPLICIT_LIMITATIONS','coverage':coverage,'findings_reviewed':len(dispositions),'whole_field_receipt_sha256':digest(full_path),'receipt_sha256':digest(path),'artifact_free_claim':False}
