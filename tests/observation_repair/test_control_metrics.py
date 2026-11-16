import numpy as np
from sprintseq.observation_repair.control_metrics import feature_control_summary,feature_review_items


def test_aggregate_pass_cannot_hide_one_gene_error():
    truth=np.r_[np.zeros(1000,int),np.ones(100,int)];pred=truth.copy();pred[:2]=1;keep=np.ones(len(truth),bool)
    assert (pred!=truth).mean()<.01
    rows=feature_control_summary(truth,pred,keep,['A','B','C'])
    assert rows[1]['reference_disagreement_fraction']>.01 and rows[1]['review_required']
    assert rows[2]['reference_rows']==0 and rows[2]['review_required']
    assert {x['feature'] for x in feature_review_items({'conditions':[{'condition':'erasure','per_feature':rows}]})}=={'B','C'}


def test_low_retention_is_explicit_without_changing_Q20():
    truth=np.zeros(200,int);pred=truth.copy();keep=np.arange(200)<20
    row=feature_control_summary(truth,pred,keep,['CCL5'])[0]
    assert row['retention']==.1 and row['wrong_calls']==0 and 'LOW_RECOVERY_SENSITIVITY' in row['review_reasons']


def test_control_exclusions_do_not_rewrite_measurement_validity(tmp_path):
    import json
    from sprintseq.observation_repair.control_metrics import select_reference_mask
    from sprintseq.observation_repair.__main__ import digest
    run=tmp_path/'sample_processed';(run/'readout').mkdir(parents=True);source=run/'readout/transcripts.parquet';source.write_bytes(b'source');ids=tmp_path/'excluded.npz';np.savez(ids,transcript_ids=np.array([2],np.int64));path=tmp_path/'exclusions.json';path.write_text(json.dumps({'scope':'CONTROL_REFERENCE_SELECTION_ONLY','run_id':'sample','source_transcripts_sha256':digest(source),'excluded_ids_file':str(ids),'excluded_ids_sha256':digest(ids)}));validity={'transcript_ids':np.array([1,2,3]),'observed':np.ones((3,10,2),bool)}
    result=select_reference_mask(validity,run,path,known_ids=np.array([1,2,3]));assert result['eligible'].tolist()==[True,False,True] and validity['observed'].all()
    source.write_bytes(b'changed')
    import pytest
    with pytest.raises(ValueError,match='current source'):select_reference_mask(validity,run,path)
