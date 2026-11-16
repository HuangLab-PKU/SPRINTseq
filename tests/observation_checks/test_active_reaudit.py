import json
from pathlib import Path
from sprintseq.observation_checks import workflow,whole_field
from sprintseq.observation_repair.__main__ import digest


def test_audit_only_does_not_short_circuit_active_data(tmp_path,monkeypatch):
    import spatial_cells.transcripts as transcripts
    import spatial_cells.observation_reentry as reentry
    run=tmp_path/'sample_processed'
    for name in ['readout','stitched','observation_repair']:(run/name).mkdir(parents=True)
    ref=tmp_path/'panel.csv';ref.write_text('Gene,Barcode\nA,A\n');receipt=tmp_path/'activation.json';receipt.write_text('{}')
    image=run/'stitched/mosaic.ome.tif';image.write_bytes(b'image')
    (run/'readout/observation_contract.json').write_text(json.dumps({'source_image_identity':[image.stat().st_size,image.stat().st_mtime_ns]}))
    (run/'observation_repair/default_activation.json').write_text(json.dumps({'receipt':str(receipt),'receipt_sha256':digest(receipt)}))
    monkeypatch.setattr(transcripts,'read_footer',lambda *a:{'codebook':{'sha256':digest(ref)}})
    monkeypatch.setattr(reentry,'verify_existing',lambda *a:{'bytes_verified':True})
    calls=[]
    monkeypatch.setattr(whole_field,'audit_run',lambda *a:(calls.append(a) or {'status':'FULL_FIELD_SCREEN_COMPLETE_REVIEW_REQUIRED'}))
    monkeypatch.setattr(workflow.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No decode/recovery subprocess allowed')))
    state,code=workflow.run(run,ref,'fresh',auto_repair=False)
    assert code==2 and state['status']=='EXISTING_DEFAULT_REAUDITED_REVIEW_REQUIRED' and len(calls)==1
    assert not (run/'logs/observation_checks/workflow.lock').exists()
    state,code=workflow.run(run,ref,'verify',auto_repair=True)
    assert code==0 and len(calls)==1 and state['artifact_review']['verification_is_not_artifact_clearance']


def test_clean_legacy_mask_cannot_bypass_whole_field_review(tmp_path,monkeypatch):
    run=tmp_path/'fresh_processed';run.mkdir();ref=tmp_path/'panel.csv';ref.write_text('Gene,Barcode\nA,AAAAAAAAAA\n');pre=tmp_path/'pre.json';pre.write_text('{}')
    monkeypatch.setattr(workflow,'preflight',lambda *a:{'receipt':str(pre),'acquired_image_failure_pixels':0})
    calls=[]
    monkeypatch.setattr(whole_field,'audit_run',lambda *a,**kw:(calls.append(a) or {'status':'FULL_FIELD_SCREEN_COMPLETE_REVIEW_REQUIRED','density_planes_verified':1}))
    class Child:
        pid=12
        def __init__(self,cmd,**kw):
            self.returncode=0;out=run/'readout';out.mkdir(exist_ok=True);audit=run/'logs/observation_checks/post_decode';audit.mkdir(exist_ok=True);name=cmd[4] if cmd[3] in ['sprintseq.cli.main','spatial_cells.cli.main'] else 'audit'
            if name=='readout':
                for p in [out/'position.csv',out/'intensity.csv',run/'logs/observation_checks/readout_binding.json']:p.write_text('{}')
            if name in ['gene-calling','audit']:
                (out/'mapping_postcode.csv').write_text('{}');(audit/'receipt.json').write_text(json.dumps({'status':'PANEL_AUDIT_PASS_WITH_REPORTED_COVERAGE_LIMITS','source_hashes':{}}))
            if name=='transcripts':
                for p in [out/'transcripts.parquet',out/'transcript_intensities.parquet']:p.write_bytes(b'fixture')
            if name=='density':
                (out/'density_Q20').mkdir();(out/'density_Q20/A.tif').write_bytes(b'fixture')
        def wait(self):return 0
        def poll(self):return 0
    monkeypatch.setattr(workflow.subprocess,'Popen',Child)
    state,code=workflow.run(run,ref,'new',auto_repair=True)
    assert code==2 and state['status']=='HOLD' and 'FULL_PANEL_REVIEW' in state['reason'] and len(calls)==1
