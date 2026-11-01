import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sprintseq.observation_checks.preflight import coverage_plan,require_finite,ObservationHold


def test_initial_pause_extends_detection_before_any_readout():
    codes=np.zeros((2,20),int);codes[0,6]=1;codes[1,2]=1
    reason=np.zeros((10,2,16,16),np.uint8);reason[0,:,4:12,4:12]=1
    plan=coverage_plan(codes,[0,1,2,3],reason)
    assert plan['added_cycles']==[5] and plan['features_at_risk']==[0]
    assert plan['effective_cycles']==[1,2,3,4,5]


def test_source_missing_does_not_invent_reaction_pause():
    codes=np.zeros((1,20),int);codes[0,6]=1
    reason=np.zeros((10,2,16,16),np.uint8);reason[0,:,4:12,4:12]=4
    plan=coverage_plan(codes,[0,1,2,3],reason)
    assert plan['added_cycles']==[]
    assert all(not h['candidate_pause_cycles'] for h in plan['scenarios'])


def test_unobservable_gene_holds_instead_of_assuming_dark_measurement():
    codes=np.zeros((1,20),int);codes[0,0]=1
    reason=np.zeros((10,2,16,16),np.uint8);reason[0,:,4:12,4:12]=4
    with pytest.raises(ObservationHold,match='UNOBSERVABLE'):coverage_plan(codes,[0,1,2,3],reason)


def test_missing_intensity_is_not_zero_and_real_zero_remains_valid():
    frame=pd.DataFrame({'cyc_1_cy3':[0.,np.nan]})
    with pytest.raises(ObservationHold,match='MISSING_NOT_MEASURED'):require_finite(frame,frame.columns)
    assert np.isnan(frame.iloc[1,0])
    require_finite(frame.iloc[:1],frame.columns)


def test_default_readout_uses_preflight_cycles_and_holds_before_detection(tmp_path,monkeypatch):
    from sprintseq.cli import readout
    from sprintseq.observation_checks import preflight
    run=tmp_path/'sample_processed';(run/'stitched').mkdir(parents=True);seen=[]
    monkeypatch.setattr(preflight,'run',lambda *args: {'plan':{'effective_cycles':[1,2,3,4,5]}})
    def detect(*args,**kwargs):seen.append(kwargs['detection_cycles']);raise RuntimeError('stop after planning')
    monkeypatch.setattr(readout,'detect_all_spots',detect)
    with pytest.raises(RuntimeError,match='stop after planning'):readout.run_pipeline('sample',base_dir=tmp_path,ref_file='exact.csv')
    assert seen==[[1,2,3,4,5]]
    monkeypatch.setattr(preflight,'run',lambda *args: (_ for _ in ()).throw(ObservationHold('HOLD_SOURCE')))
    with pytest.raises(ObservationHold):readout.run_pipeline('sample',base_dir=tmp_path,ref_file='exact.csv')
    assert len(seen)==1


def test_actual_intensity_reader_rejects_absent_planes(tmp_path,monkeypatch):
    from sprintseq.cli import readout
    monkeypatch.setattr(readout,'_has_stitched',lambda *args:False)
    with pytest.raises(ObservationHold,match='MISSING_NOT_MEASURED'):
        readout.read_all_intensities(tmp_path,np.array([[10.,10.]]),channels=['cy3','cy5'],seq_cycle=1,n_workers=1)


def test_worker_exception_is_not_converted_to_an_empty_detection(monkeypatch):
    from sprintseq.cli import readout
    def fail(*a,**k):raise RuntimeError('GPU failure')
    monkeypatch.setattr(readout,'get_spot_coordinates',fail)
    with pytest.raises(RuntimeError,match='GPU failure'):
        readout._detect_spots_in_block(('cy3',np.ones((8,8)),0,0,'spotiflow',{}))


def test_decode_entry_does_not_accept_old_mapping_after_failed_decoder(tmp_path,monkeypatch):
    from sprintseq.cli import main as cli
    from sprintseq.observation_checks import preflight
    run=tmp_path/'sample_processed';(run/'readout').mkdir(parents=True);(run/'readout/mapping_postcode.csv').write_text('stale')
    monkeypatch.setattr(preflight,'validate_readout_binding',lambda *a:None)
    monkeypatch.setattr(cli.gc_mod,'run_pipeline',lambda **kw:None)
    with pytest.raises(ObservationHold,match='NO_FRESH_DECODE'):
        cli.main(['gene-calling','--run-id','sample','--base-dir',str(tmp_path),'--ref-file','exact.csv'])


def test_panel_audit_emits_every_gene_and_flags_image_bound_deficit(tmp_path):
    from sprintseq.observation_checks.audit import run
    from sprintseq.observation_repair.__main__ import digest
    image=tmp_path/'image.tif';image.write_bytes(b'identity');ref=tmp_path/'panel.csv';pd.DataFrame({'Gene':['A','B','C'],'Barcode':['A','T','C']}).to_csv(ref,index=False)
    qc=tmp_path/'preflight';qc.mkdir();reason=np.zeros((10,2,32,32),np.uint8);reason[0,:,12:20,12:20]=1;np.savez(qc/'image_validity.npz',reason=reason,full_shape=[320,320])
    pre={'contract':{'run_id':'sample','codebook_sha256':digest(ref),'image_identity':[image.stat().st_size,image.stat().st_mtime_ns]},'scan':{'source_image':str(image)},'validity_sha256':digest(qc/'image_validity.npz'),'plan':{'added_cycles':[5]}}
    receipt=qc/'receipt.json';receipt.write_text(json.dumps(pre));readout=tmp_path/'readout';readout.mkdir();n=360
    pd.DataFrame({'index':range(n),'Y':[155.]*n,'X':[155.]*160+[105.]*200}).to_csv(readout/'position.csv',index=False)
    pd.DataFrame({'index':range(n),'Gene':['B']*160+['A']*150+['B']*50,'Probability':[.999]*n}).to_csv(readout/'mapping_postcode.csv',index=False)
    result=run(tmp_path,ref,tmp_path/'audit',receipt)
    assert result['status']=='HOLD_ARTIFACT_REVIEW_REQUIRED' and result['features_audited']==3
    assert any(row['gene']=='A' for row in result['flagged_gene_regions'])
    rows=pd.read_csv(tmp_path/'audit/all_feature_region_audit.csv');assert set(rows.gene)=={'A','B','C'}


def test_workflow_holds_after_export_and_never_calls_density(tmp_path,monkeypatch):
    from sprintseq.observation_checks import workflow
    run=tmp_path/'sample_processed';run.mkdir();ref=tmp_path/'panel.csv';ref.write_text('Gene,Barcode\nA,A\n');pre=tmp_path/'pre.json';pre.write_text('{}')
    monkeypatch.setattr(workflow,'preflight',lambda *a:{'receipt':str(pre)})
    seen=[]
    class Child:
        pid=123
        def __init__(self,cmd,**kwargs):
            module=cmd[3];name=cmd[4] if module in ['sprintseq.cli.main','spatial_cells.cli.main'] else 'audit';seen.append(name);self.returncode=2 if name in ['gene-calling','audit'] else 0
            out=run/'readout';out.mkdir(exist_ok=True);audit=run/'observation_checks/post_decode';audit.mkdir(exist_ok=True)
            if name=='readout':
                for p in [out/'position.csv',out/'intensity.csv',run/'observation_checks/readout_binding.json']:p.write_text('{}')
            if name=='gene-calling':(out/'mapping_postcode.csv').write_text('provisional');(audit/'receipt.json').write_text('{"status":"HOLD_ARTIFACT_REVIEW_REQUIRED"}')
            if name=='transcripts':
                for p in [out/'transcripts.parquet',out/'transcript_intensities.parquet']:p.write_bytes(b'fixture')
        def wait(self):return self.returncode
        def poll(self):return self.returncode
    monkeypatch.setattr(workflow.subprocess,'Popen',Child)
    state,code=workflow.run(run,ref,'first',auto_repair=False)
    assert code==2 and state['status']=='HOLD'
    assert seen==['readout','gene-calling','transcripts','audit']
    assert not (run/'observation_checks/workflow.lock').exists()
    seen.clear();state,code=workflow.run(run,ref,'resume',auto_repair=False)
    assert seen==[] and code==2


def test_local_nodata_is_explicit_and_not_fabricated_coverage():
    codes=np.zeros((1,20),int);codes[0,0]=1
    reason=np.zeros((10,2,16,16),np.uint8);reason[0,:,4:12,4:12]=4
    plan=coverage_plan(codes,[0,1,2,3],reason,allow_local_nodata=True)
    assert plan['unobservable_feature_patterns'] and plan['added_cycles']==[]


def test_supplement_roi_filters_worker_coordinates_and_skips_empty_blocks(monkeypatch):
    from sprintseq.cli import readout
    mask=np.zeros((10,10),bool);mask[2:4,2:4]=True
    region=readout._block_region(mask,(100,100),0,0,50,50)
    monkeypatch.setattr(readout,'get_spot_coordinates',lambda *a,**k:np.array([[25.,25.],[45.,45.]]))
    _,coords,error=readout._detect_spots_in_block(('cy3',np.ones((50,50)),0,0,'spotiflow',{},region))
    assert coords.tolist()==[[25.,25.]] and error is None
    assert not readout._block_region(mask,(100,100),50,50,50,50)[0].any()


def test_registration_failure_requires_matching_solved_report(tmp_path):
    from sprintseq.observation_checks.preflight import registration_evidence
    from sprintseq.observation_repair.__main__ import digest
    (tmp_path/'registration_unresolved.json').write_text('{}')
    with pytest.raises(ObservationHold,match='UNRESOLVED'):registration_evidence(tmp_path)
    shift=tmp_path/'shift_table.csv';shift.write_text('solved fixture')
    report=tmp_path/'registration_report.json';report.write_text(json.dumps({'shift_table_sha256':digest(shift),'measurement':{'unresolved':[],'quality_exceptions':[]}}))
    assert registration_evidence(tmp_path)['status'].startswith('NUMERIC_REGISTRATION_COMPLETE')
    shift.write_text('changed')
    with pytest.raises(ObservationHold,match='MISMATCH'):registration_evidence(tmp_path)


def test_detection_planning_does_not_flag_unmeasured_future_histories():
    codes=np.zeros((1,20),int);codes[0,0]=1
    reason=np.zeros((10,2,16,16),np.uint8);reason[7:,:,4:12,4:12]=3
    plan=coverage_plan(codes,[0,1,2,3],reason,allow_local_nodata=True)
    assert plan['repair_scope_exceeded']==[]
    assert all(not x['candidate_pause_cycles'] for x in plan['scenarios'])


def test_empty_canvas_outside_every_rendered_plane_is_not_a_missing_tile_region():
    codes=np.zeros((1,20),int);codes[0,6]=1
    reason=np.zeros((10,2,16,16),np.uint8);reason[:,:,:8,:]=4
    plan=coverage_plan(codes,[0,1,2,3],reason,allow_local_nodata=True)
    assert plan['outside_acquisition_canvas_pixels']==128
    assert plan['unobservable_feature_patterns']==[]
    assert plan['added_cycles']==[]
    reason[:]=4
    with pytest.raises(ObservationHold,match='NO_ACQUIRED'):coverage_plan(codes,[0,1,2,3],reason,allow_local_nodata=True)
