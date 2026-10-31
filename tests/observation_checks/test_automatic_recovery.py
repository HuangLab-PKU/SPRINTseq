import json
import numpy as np
import pandas as pd
import pytest
from sprintseq.observation_repair.discovery_policy import DiscoveryPolicy,early_failure_regions,alignment_gate,fallback_required,normalize_observed,candidate_gate


def test_missing_allspot_and_bad_alignment_choose_later_cycles():
    assert fallback_required(False,{'qualified':True})
    assert fallback_required(True,{'qualified':False})
    assert not fallback_required(True,{'qualified':True})


def test_local_alignment_does_not_relax_a_failed_regional_gate():
    x=np.arange(120)*5.+20
    anchors=np.r_[np.c_[np.ones(120)*20,x],np.c_[np.ones(120)*1100,x]]
    detected=anchors.copy();detected[:120,0]+=.5;detected[120:,0]+=4
    policy=DiscoveryPolicy(minimum_anchors=100)
    take,record=alignment_gate(anchors,detected,np.array([[20.,100.],[1100.,100.]]),policy)
    assert not record['qualified'] and take.tolist()==[True,False]
    assert fallback_required(True,record)


def test_low_anchor_fraction_is_not_good_alignment():
    anchors=np.c_[np.arange(1000)*10.,np.zeros(1000)]
    _,result=alignment_gate(anchors,anchors[:110],anchors[:3])
    assert not result['qualified']


def test_source_missing_and_optical_regions_are_proposed_but_empty_canvas_is_not():
    reason=np.zeros((10,2,32,32),np.uint8);reason[:,:,:8]=4;reason[0,0,12:20,12:20]=4;reason[2,:,22:30,22:30]=3
    labels=early_failure_regions(reason)
    assert not labels[:8].any() and labels[15,15]>0 and labels[25,25]>0


def test_nan_missing_is_marginalized_real_measured_nan_fails():
    model={'log_add':np.ones(4),'data_log_mean':np.zeros(4),'data_log_std':np.ones(4)}
    raw=np.array([[2.,np.nan,0.,3.]]);observed=np.array([[True,False,True,True]])
    result=normalize_observed(raw,observed,model)
    assert np.isnan(result[0,1]) and result[0,2]==0
    with pytest.raises(ValueError):normalize_observed(raw,np.ones_like(observed),model)


def test_high_posterior_alone_never_qualifies_an_added_spot():
    args=[np.zeros(4,int),np.ones(4)*.999,np.ones(4),2,[True,True,True,False],[False,True,True,True],[True,False,True,True],[True]*4]
    assert candidate_gate(*args).tolist()==[False,False,True,False]


def test_completed_acquisition_census_absence_vs_unfinished_and_gap(tmp_path):
    from sprintseq.observation_checks.acquisition import acquisition_check
    from sprintseq.observation_checks.preflight import ObservationHold
    from sprintseq.observation_repair.__main__ import digest
    run=tmp_path/'sample_processed';control=run/'observation_checks';control.mkdir(parents=True);raw=tmp_path/'raw';(raw/'cyc_1').mkdir(parents=True);completion=tmp_path/'finished';completion.write_text('closed acquisition')
    plan={'run_id':'sample','raw_run':str(raw),'acquisition_complete':True,'completion_evidence':[{'path':str(completion),'sha256':digest(completion)}],'expected_observations':[{'cycle':1,'channel':'cy3','tile_id':1,'render_tile':'tile1.tif','z_indices':[0,1]}]}
    path=control/'acquisition_plan.json';path.write_text(json.dumps(plan));result=acquisition_check(run)
    assert len(result['missing_stacks'])==1 and not result['chemistry_pause_inferred']
    (raw/'cyc_1/C001-T0001-cy3-Z000.tif').write_bytes(b'frame')
    with pytest.raises(ObservationHold,match='GAPPED'):acquisition_check(run)
    plan['acquisition_complete']=False;path.write_text(json.dumps(plan))
    with pytest.raises(ObservationHold,match='NOT_COMPLETE'):acquisition_check(run)


def test_workflow_routes_audit_hold_into_recovery_by_default(tmp_path,monkeypatch):
    from sprintseq.observation_checks import workflow,recovery
    run=tmp_path/'sample_processed';run.mkdir();ref=tmp_path/'panel.csv';ref.write_text('Gene,Barcode\nA,A\n');pre=tmp_path/'pre.json';pre.write_text('{}');monkeypatch.setattr(workflow,'preflight',lambda *a:{'receipt':str(pre)})
    seen=[]
    class Child:
        pid=123
        def __init__(self,cmd,**kwargs):
            name=cmd[4] if cmd[3] in ['sprintseq.cli.main','spatial_cells.cli.main'] else 'audit';seen.append(name);self.returncode=2 if name in ['gene-calling','audit'] else 0;out=run/'readout';out.mkdir(exist_ok=True);audit=run/'observation_checks/post_decode';audit.mkdir(exist_ok=True)
            if name=='readout':
                for f in [out/'position.csv',out/'intensity.csv',run/'observation_checks/readout_binding.json']:f.write_text('{}')
            if name=='gene-calling':(out/'mapping_postcode.csv').write_text('provisional');(audit/'receipt.json').write_text('{"status":"HOLD_ARTIFACT_REVIEW_REQUIRED"}')
            if name=='transcripts':
                for f in [out/'transcripts.parquet',out/'transcript_intensities.parquet']:f.write_bytes(b'fixture')
        def wait(self):return self.returncode
        def poll(self):return self.returncode
    monkeypatch.setattr(workflow.subprocess,'Popen',Child)
    def recover(*a,**kw):seen.append('automatic_recovery');return {'status':'COMPLETE_WITH_REPORTED_COVERAGE_LIMITS'}
    monkeypatch.setattr(recovery,'run',recover)
    result,code=workflow.run(run,ref,'auto')
    assert code==0 and result['status'].startswith('COMPLETE') and seen==['readout','gene-calling','transcripts','audit','automatic_recovery']


def test_resolution_preserves_flags_instead_of_marking_image_clean():
    from sprintseq.observation_checks.recovery import resolve_audit
    before={'features_audited':2,'flagged_gene_regions':[{'gene':'A'}]};after={'features_audited':2,'regions':[],'flagged_gene_regions':[{'gene':'B'}]}
    result=resolve_audit(before,after,{'status':'STANDARD_OBSERVATION_CANDIDATE_VERIFIED','density_planes_verified':2},{'components':[]})
    assert result['flagged_gene_regions_after']==[{'gene':'B'}] and not result['scientific_accuracy_accepted']


def test_standard_cli_exposes_automatic_pipeline(capsys):
    from sprintseq.cli.main import main
    with pytest.raises(SystemExit) as error:main(['pipeline','--help'])
    assert error.value.code==0 and '--recover-existing' in capsys.readouterr().out

@pytest.mark.parametrize('allspot_available,bad_allspot',[(False,False),(True,True),(True,False),(True,'empty')])
def test_discovery_orchestrator_uses_later_rounds_when_allspot_missing_or_bad(tmp_path,monkeypatch,allspot_available,bad_allspot):
    import pyarrow.parquet as pq
    from sprintseq.observation_repair import discovery as d
    from sprintseq.readout import mosaic
    run=tmp_path/'sample_processed';root=run/'observation_repair/generation'
    for name in ['readout','stitched','codebook']:(run/name).mkdir(parents=True,exist_ok=True)
    for name in ['repair','image_qc','model']:(root/name).mkdir(parents=True)
    image=run/'stitched/mosaic.ome.tif';image.write_bytes(b'fixture');coords=np.array([[20.+18*(i//15),20.+18*(i%15)] for i in range(120)])
    frame=pd.DataFrame({'transcript_id':np.arange(120),'y_px':coords[:,0],'x_px':coords[:,1],'is_gene':True,'probability':.999});frame.to_parquet(run/'readout/transcripts.parquet');columns=[f'r{t:02d}_{c}' for t in range(1,11) for c in ['cy3','cy5']];raw=pd.DataFrame(np.ones((120,20))*100,columns=columns);raw.insert(0,'transcript_id',np.arange(120));raw.to_parquet(run/'readout/transcript_intensities.parquet');(run/'codebook/codebook.csv').write_text('Gene,Barcode\nA,A\n');(run/'readout/position.csv').write_text('fixture');(run/'readout/intensity.csv').write_text('fixture')
    reason=np.zeros((10,2,32,32),np.uint8);reason[0,0]=4;np.savez(root/'image_qc/image_validity.npz',reason=reason,full_shape=[512,512]);np.savez(root/'repair/repair_calls.npz',transcript_ids=[]);np.savez(root/'model/model.npz',fixture=[1])
    from sprintseq.observation_repair.__main__ import digest
    qr={'source_tiff_identity':[image.stat().st_size,image.stat().st_mtime_ns],'image_validity_sha256':digest(root/'image_qc/image_validity.npz')};(root/'image_qc/receipt.json').write_text(json.dumps(qr));mr={'model_sha256':digest(root/'model/model.npz'),'gene_classes':2,'intensity_order':columns,'input_parquet_sha256':digest(run/'readout/transcript_intensities.parquet')};(root/'model/receipt.json').write_text(json.dumps(mr));rr={'run_id':'sample','model_receipt':str(root/'model/receipt.json'),'source_transcripts_sha256':digest(run/'readout/transcripts.parquet')};(root/'repair/receipt.json').write_text(json.dumps(rr))
    monkeypatch.setattr(d,'ObservableHistoryModel',lambda *a:object());monkeypatch.setattr(mosaic,'list_mosaics',lambda p:[(t,c) for t in range(1,12 if allspot_available else 11) for c in ['cy3','cy5']]);seen=[]
    def detect(run,cycles,*a):
        seen.append(cycles);found=np.r_[coords,np.array([[400.,400.]])]
        if cycles==[11] and bad_allspot=='empty':return np.empty((0,2)),np.empty(0,np.uint16)
        if cycles==[11] and bad_allspot:found=found+[4.,0.]
        return found,np.ones(len(found),np.uint16)
    monkeypatch.setattr(d,'detect',detect)
    monkeypatch.setattr(d,'measurements',lambda run,points,roi,shape,cycles,policy:(np.ones((len(points),20))*100,{(t,c):np.ones(len(points),bool) for t in set(range(1,11))|set(cycles) for c in range(2)}))
    def decode(root,reason,points,shape,raw,peaks,cycles,aligned,*a,**kw):return {'y_px':points[:,0],'x_px':points[:,1],'adoptable':aligned}
    monkeypatch.setattr(d,'decode_candidates',decode)
    result=d.run(run,root,root/'discovery')
    expected=[[11]] if allspot_available else []
    if not allspot_available or bad_allspot:expected+=[list(range(5,11))]
    assert seen==expected and result['adoptable']==1
    assert d.run(run,root,root/'discovery')==result and seen==expected


def test_new_points_use_exact_renderer_support_not_pyramid_centre(tmp_path):
    import zlib
    from sprintseq.observation_repair.discovery import exact_point_reason
    (tmp_path/'stitched').mkdir();support=np.ones((8,8),bool);support[3,4]=False;packed=tmp_path/'support.bin';packed.write_bytes(zlib.compress(np.packbits(support,axis=1,bitorder='little').tobytes()));planes=[{'cycle':t,'channel':c,'support_encoding':'zlib, row-major np.packbits(axis=1, bitorder=little)','support_path':str(packed)} for t in range(1,11) for c in ['cy3','cy5']];(tmp_path/'stitched/finite_stitch_report.json').write_text(json.dumps({'planes':planes}));reason=np.zeros((10,2,2,2),np.uint8);reason[0,0]=4
    actual=exact_point_reason(tmp_path,reason,np.array([[3.,4.],[2.,2.]]),(8,8))
    assert ((actual[0]&4)>0).all() and not ((actual[1]&4)>0).any()


def test_actual_empty_discovery_decode_is_a_valid_checkpoint(tmp_path):
    from types import SimpleNamespace
    from sprintseq.observation_repair.discovery import decode_candidates
    model={'log_add':np.ones(20),'data_log_mean':np.zeros(20),'data_log_std':np.ones(20)}
    result=decode_candidates(tmp_path,np.zeros((10,2,2,2),np.uint8),np.empty((0,2)),(32,32),np.empty((0,20)),{(t,c):np.empty(0,bool) for t in range(1,12) for c in range(2)},[11],np.empty(0,bool),model,SimpleNamespace(K=2),DiscoveryPolicy())
    assert result['intensities'].shape==(0,20) and not result['adoptable'].any()
