import json
import numpy as np
import pytest
from sprintseq.observation_checks.review_gate import require_review,review_items
from sprintseq.observation_checks.whole_field import SCREEN_POLICY
from sprintseq.observation_checks.preflight import ObservationHold,coverage_plan
from sprintseq.observation_repair.__main__ import digest


def test_review_gate_requires_actual_whole_field_and_every_finding(tmp_path):
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'run_id':'run','unassessed_small_failure_patterns':[{'pattern':1,'pixels':63}]}));source=tmp_path/'transcripts';source.write_bytes(b'fixture');revision=tmp_path/'revision.json';revision.write_text(json.dumps({'transcripts_sha256':digest(source)}));evidence=tmp_path/'controls.json';evidence.write_text('{}');review=tmp_path/'review.json';full_path=tmp_path/'full.json'
    with pytest.raises(ObservationHold,match='REQUIRED'):require_review(review,'run',['A','B'],audit,revision)
    rows=[{'section':'run','feature':g,'whole_field_visited':True,'review_state':'CHECKED_WITH_LIMITATIONS','evidence':[str(evidence)],'limitations':['Conditional measurement evidence']} for g in ['A','B']]
    full={'run_id':'run','policy':SCREEN_POLICY,'morphology_support_status':'CURRENT_SOURCE_BOUND_CELL_MASK_AREA','source_image':str(source),'source_image_identity':[source.stat().st_size,source.stat().st_mtime_ns],'source_transcripts':str(source),'source_bindings':{str(source):digest(source)},'features':rows,'findings':[{'feature':'A','finding_id':'region_A'}]};full_path.write_text(json.dumps(full))
    proof={'run_id':'run','audit_sha256':digest(audit),'revision_receipt_sha256':digest(revision),'ledger':rows,'evidence_hashes':{str(evidence):digest(evidence)}};review.write_text(json.dumps(proof))
    with pytest.raises(ObservationHold,match='FULL_FIELD_RECEIPT'):require_review(review,'run',['A','B'],audit,revision)
    proof.update(whole_field_receipt=str(full_path),whole_field_receipt_sha256=digest(full_path));review.write_text(json.dumps(proof))
    with pytest.raises(ObservationHold,match='FINDINGS'):require_review(review,'run',['A','B'],audit,revision)
    proof['finding_dispositions']={key:{'review_state':'CHECKED_WITH_LIMITATIONS','evidence':[str(evidence)],'limitations':['Conditional']} for key in review_items(json.loads(audit.read_text()),full)};review.write_text(json.dumps(proof))
    assert require_review(review,'run',['A','B'],audit,revision)['findings_reviewed']==2
    proof['ledger']=rows[:1];review.write_text(json.dumps(proof))
    with pytest.raises(ObservationHold,match='INCOMPLETE'):require_review(review,'run',['A','B'],audit,revision)
    proof['ledger']=rows;review.write_text(json.dumps(proof));evidence.write_text('changed')
    with pytest.raises(ObservationHold,match='EVIDENCE'):require_review(review,'run',['A','B'],audit,revision)
    audit.write_text('changed')
    with pytest.raises(ObservationHold,match='STALE'):require_review(review,'run',['A','B'],audit,revision)


def test_small_image_patterns_are_recorded_as_unassessed():
    reason=np.zeros((10,2,16,16),np.uint8);reason[0,0,:7,:9]=1
    codes=np.zeros((2,20),int);codes[:,4]=1
    plan=coverage_plan(codes,[0,1,2,3],reason,allow_local_nodata=True)
    assert plan['retained_image_patterns_all_planned'] is False
    assert plan['unassessed_small_failure_patterns']==[{'pattern':1,'pixels':63,'feature_count':2,'assessment':'NOT_ASSESSED_BELOW_PLANNING_AREA'}]


def test_small_pattern_hold_propagates_through_audit_and_recovery(tmp_path):
    import pandas as pd
    from sprintseq.observation_checks.audit import run as audit_run
    from sprintseq.observation_checks.recovery import resolve_audit
    image=tmp_path/'image.tif';image.write_bytes(b'image');ref=tmp_path/'panel.csv';pd.DataFrame({'Gene':['A','B'],'Barcode':['AAAAAAAAAA','TTTTTTTTTT']}).to_csv(ref,index=False)
    reason=np.zeros((10,2,16,16),np.uint8);reason[0,0,:7,:9]=1;codes=np.zeros((2,20),int);codes[:,4]=1;plan=coverage_plan(codes,[0,1,2,3],reason,allow_local_nodata=True)
    qc=tmp_path/'qc';qc.mkdir();np.savez(qc/'image_validity.npz',reason=reason,full_shape=[160,160]);pre={'contract':{'run_id':'run','codebook_sha256':digest(ref),'image_identity':[image.stat().st_size,image.stat().st_mtime_ns]},'scan':{'source_image':str(image)},'validity_sha256':digest(qc/'image_validity.npz'),'plan':plan};receipt=qc/'receipt.json';receipt.write_text(json.dumps(pre));readout=tmp_path/'readout';readout.mkdir();pd.DataFrame({'index':[0],'Y':[100.],'X':[100.]}).to_csv(readout/'position.csv',index=False);pd.DataFrame({'index':[0],'Gene':['A'],'Probability':[.999]}).to_csv(readout/'mapping_postcode.csv',index=False)
    result=audit_run(tmp_path,ref,tmp_path/'audit',receipt)
    assert result['status']=='HOLD_ARTIFACT_REVIEW_REQUIRED' and not result['flagged_gene_regions']
    assert result['unassessed_small_failure_patterns']==plan['unassessed_small_failure_patterns']
    resolved=resolve_audit(result,result,{'status':'STANDARD_OBSERVATION_CANDIDATE_VERIFIED','density_planes_verified':2},{'components':[]})
    assert resolved['unassessed_small_failure_patterns'] and resolved['artifact_review_complete'] is False
