import numpy as np
import pytest
from sprintseq.observation_checks.whole_field import screen,cohort_screen,validate_coverage


def test_density_hole_outside_image_mask_requires_source_review():
    counts=np.full((1,64,64),10,dtype=int);counts[0,25:39,25:39]=0
    result=screen(counts,['CCL5'],np.ones((64,64)),image_failure=np.zeros((64,64),bool))
    assert any(f['kind']=='DEPLETION' and 25<=f['peak_yx'][0]<39 and 25<=f['peak_yx'][1]<39 for f in result['findings'])
    assert all(f['image_mask_overlap'] is False and not f['mechanism_inferred'] for f in result['findings'])
    assert result['features'][0]['review_state']=='REVIEW_REQUIRED'


def test_zero_gene_and_molecular_void_are_explicit_not_pass():
    result=screen(np.zeros((2,20,20),int),['A','B'],np.ones((20,20)))
    assert len(result['features'])==2 and result['tissue_centres_without_any_q20']==400
    assert all(x['review_state']=='INSUFFICIENT_EVIDENCE' for x in result['features'])
    assert result['release_accepted'] is False


def test_missing_plane_is_not_a_zero_gene():
    with pytest.raises(ValueError):screen(np.ones((1,20,20)),['A','B'],np.ones((20,20)))
    with pytest.raises(ValueError):screen(np.full((2,20,20),np.nan),['A','B'],np.ones((20,20)))


def test_section_wide_loss_with_normal_local_ratios_is_detected():
    result=cohort_screen([[1000,100],[1000,0],[1000,100]],[1,2,3],['A','ID4'])
    assert any(x['section']==2 and x['feature']=='ID4' for x in result)
    assert all(x['mechanism_inferred'] is False for x in result)


def test_complete_ledger_is_not_completed_review():
    rows=[{'section':s,'feature':g,'whole_field_visited':True,'review_state':'NOT_REVIEWED'} for s in [1,2] for g in ['A','B']]
    assert validate_coverage(rows,[1,2],['A','B'])['pairs']==4
    with pytest.raises(ValueError):validate_coverage(rows[:-1],[1,2],['A','B'])
    with pytest.raises(ValueError):validate_coverage(rows+[rows[0]],[1,2],['A','B'])
    with pytest.raises(ValueError):validate_coverage(rows,[1,2],['A','B'],require_review=True)
    rows[0]['review_state']='MAGIC_PASS'
    with pytest.raises(ValueError):validate_coverage(rows,[1,2],['A','B'])


def test_review_needs_evidence_and_does_not_claim_artifact_free():
    row={'section':1,'feature':'A','whole_field_visited':True,'review_state':'CHECKED_WITH_LIMITATIONS'}
    with pytest.raises(ValueError):validate_coverage([row],[1],['A'],require_review=True)
    row.update(evidence=['source-bound controls'],limitations=['Not a biological accuracy estimate'])
    assert validate_coverage([row],[1],['A'],require_review=True)['artifact_free_claim'] is False


def test_duplicate_or_empty_expected_inventory_is_rejected():
    row={'section':1,'feature':'A','whole_field_visited':True,'review_state':'NOT_REVIEWED'}
    for sections,features in [([1,1],['A']),([1],['A','A']),([],['A']),([1],[])]:
        with pytest.raises(ValueError):validate_coverage([row] if sections and features else [],sections,features)


def test_whole_section_zero_is_not_hidden_by_zero_normalization():
    alerts=cohort_screen([[1000,100],[0,0],[1000,100]],[1,2,3],['A','B'])
    assert {x['feature'] for x in alerts if x['section']==2 and x['kind']=='WHOLE_SECTION_NO_Q20'}=={'A','B'}


def test_unactivated_run_can_bind_independent_morphology(tmp_path,monkeypatch):
    import json,pandas as pd,pyarrow as pa,pyarrow.parquet as pq,tifffile
    import spatial_cells.transcripts as transcripts
    from sprintseq.observation_checks.whole_field import audit_run
    from sprintseq.observation_repair.__main__ import digest
    (tmp_path/'readout').mkdir();(tmp_path/'observation_checks').mkdir();image=tmp_path/'image.tif';mask=tmp_path/'mask.tif';tifffile.imwrite(image,np.ones((20,20),np.uint16));tifffile.imwrite(mask,np.ones((20,20),np.uint8));ref=tmp_path/'panel.csv';pd.DataFrame({'Gene':['A','B'],'Barcode':['AAAAAAAAAA','TTTTTTTTTT']}).to_csv(ref,index=False)
    source=tmp_path/'readout/transcripts.parquet';pq.write_table(pa.Table.from_pandas(pd.DataFrame({'x_px':[5.],'y_px':[5.],'feature_name':['A'],'is_gene':[True],'probability':[.999]})),source)
    monkeypatch.setattr(transcripts,'read_footer',lambda *a:{'image_shape_px':[20,20],'run_id':'run','pixel_size_um':.1625,'reference_image':str(image)})
    record={'schema':'independent_morphology_support.v1','run_id':'run','mask':str(mask),'mask_sha256':digest(mask),'source_image':str(image),'source_image_identity':[image.stat().st_size,image.stat().st_mtime_ns],'full_shape_px':[20,20],'pixel_size_um':.1625};(tmp_path/'observation_checks/morphology_support.json').write_text(json.dumps(record))
    result=audit_run(tmp_path,ref,tmp_path/'audit');assert result['morphology_support_status']=='SOURCE_BOUND_MORPHOLOGY_MASK_AREA' and result['coverage']['pairs']==2 and result['release_accepted'] is False
    image.touch()
    with pytest.raises(ValueError,match='source differs'):audit_run(tmp_path,ref,tmp_path/'changed_source_audit')
