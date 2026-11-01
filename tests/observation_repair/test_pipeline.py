import json,sys,pytest
from sprintseq.observation_repair.pipeline import run_stage,check_hashes

def test_real_child_failure_preserves_exit_and_does_not_invent_receipt(tmp_path):
    state={'steps':{}}
    with pytest.raises(RuntimeError,match='actual child exit 7'):
        run_stage('bad',[sys.executable,'-c','raise SystemExit(7)'],tmp_path/'receipt.json',state,tmp_path,lambda r:None)
    assert json.loads((tmp_path/'bad_exit.json').read_text())['exit_code']==7
    assert not (tmp_path/'receipt.json').exists() and not state['steps']

def test_resume_requires_validator_and_does_not_fake_child_exit(tmp_path):
    receipt=tmp_path/'receipt.json';receipt.write_text('{"ok": true}');state={'steps':{}}
    run_stage('ready',['not-run'],receipt,state,tmp_path,lambda r:None)
    assert state['steps']['ready']['actual_child_exit'] is None
    with pytest.raises(ValueError):check_hashes({str(receipt):'wrong'})

def test_recorded_sources_resolve_preserved_baseline_after_activation(tmp_path):
    import hashlib
    from sprintseq.observation_repair.pipeline import resolve_recorded_source
    current=tmp_path/'readout/data';current.parent.mkdir();current.write_bytes(b'new');old=tmp_path/'observation_repair/activation/baseline/readout/data';old.parent.mkdir(parents=True);old.write_bytes(b'old');h=hashlib.sha256(b'old').hexdigest();rel=old.parents[2]/'baseline_relocation.json';rel.write_text(json.dumps({str(current):{'preserved_at':str(old),'sha256':h}}))
    assert resolve_recorded_source(tmp_path,current,h)==old
    assert resolve_recorded_source(tmp_path,current,hashlib.sha256(b'new').hexdigest())==current
    old.write_bytes(b'corrupt')
    with pytest.raises(ValueError):resolve_recorded_source(tmp_path,current,h)

def test_recorded_source_cannot_escape_run(tmp_path):
    from sprintseq.observation_repair.pipeline import resolve_recorded_source
    with pytest.raises(ValueError):resolve_recorded_source(tmp_path,tmp_path.parent/'foreign','irrelevant')
