"""Published defaults remain protected after disposable work directories are removed."""
import json
from types import SimpleNamespace
import pytest
from sprintseq.observation_paths import activation_path,current_path
from spatial_cells.observation_paths import activation_path as cells_activation,current_path as cells_current
from sprintseq.observation_repair.__main__ import digest

@pytest.mark.parametrize('function,peer,root_name,legacy_name',[(activation_path,cells_activation,'observation_default_activation.json','default_activation.json'),(current_path,cells_current,'observation_current.json','current.json')])
def test_root_precedence_legacy_fallback_and_new_writes(tmp_path,function,peer,root_name,legacy_name):
    run=tmp_path/'run';run.mkdir();root=run/root_name;legacy=run/'observation_repair'/legacy_name
    assert function(run)==peer(run)==root
    assert not (run/'observation_repair').exists()
    legacy.parent.mkdir();legacy.write_text('legacy')
    assert function(run)==peer(run)==legacy
    assert function(run,for_write=True)==peer(run,for_write=True)==root
    root.write_text('root')
    assert function(run)==peer(run)==root

@pytest.mark.parametrize('marker_location',['root','legacy'])
def test_readout_and_gene_calling_cannot_overwrite_active_defaults(tmp_path,marker_location):
    from sprintseq.cli import main,readout
    from sprintseq.observation_checks.preflight import ObservationHold
    run=tmp_path/'sample_processed';(run/'stitched').mkdir(parents=True)
    marker=run/'observation_default_activation.json' if marker_location=='root' else run/'observation_repair/default_activation.json'
    marker.parent.mkdir(parents=True,exist_ok=True);marker.write_text('{}')
    with pytest.raises(ObservationHold,match='HOLD_ACTIVE_DEFAULT'):readout.run_pipeline('sample',base_dir=tmp_path)
    with pytest.raises(ObservationHold,match='HOLD_ACTIVE_DEFAULT'):main._run_gene_calling(SimpleNamespace(base_dir=tmp_path,run_id='sample'))
    assert not (run/'readout').exists()

@pytest.mark.parametrize('root_corrupt',[False,True])
def test_workflow_uses_root_marker_and_never_falls_back_on_corruption(tmp_path,monkeypatch,root_corrupt):
    from sprintseq.observation_checks import workflow
    import spatial_cells.transcripts as transcripts
    import spatial_cells.observation_reentry as reentry
    run=tmp_path/'sample_processed'
    for name in ['readout','stitched','observation_repair']:(run/name).mkdir(parents=True)
    ref=tmp_path/'panel.csv';ref.write_text('Gene,Barcode\nA,A\n');receipt=run/'release.json';receipt.write_text('{}');image=run/'stitched/mosaic.ome.tif';image.write_bytes(b'image')
    (run/'readout/observation_contract.json').write_text(json.dumps({'source_image_identity':[image.stat().st_size,image.stat().st_mtime_ns]}))
    entry={'receipt':str(receipt),'receipt_sha256':digest(receipt)}
    (run/'observation_default_activation.json').write_text('invalid' if root_corrupt else json.dumps(entry));(run/'observation_repair/default_activation.json').write_text(json.dumps(entry) if root_corrupt else 'invalid')
    monkeypatch.setattr(transcripts,'read_footer',lambda *a:{'codebook':{'sha256':digest(ref)}})
    calls=[];monkeypatch.setattr(reentry,'verify_existing',lambda *a:(calls.append(a) or {'bytes_verified':True}))
    monkeypatch.setattr(workflow.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No processing allowed')))
    if root_corrupt:
        with pytest.raises(json.JSONDecodeError):workflow.run(run,ref,'root_corrupt')
        assert not calls
    else:
        state,code=workflow.run(run,ref,'root_valid');assert code==0 and state['status']=='EXISTING_DEFAULT_VERIFIED_NOT_REPLAYED' and len(calls)==1
    assert not (run/'observation_checks/workflow.lock').exists()
