import json
import pytest
from sprintseq.readout.runtime_resources import resolve_workers

def policy(tmp_path, value, **extra):
    run=tmp_path/'sample_processed';run.mkdir()
    (run/'readout_resources.json').write_text(json.dumps({'version':1,'run_id':'sample','n_workers':value,**extra}))
    return run

def test_default_without_policy(tmp_path):
    assert resolve_workers(tmp_path/'sample_processed')==4

def test_run_policy_limits_workers(tmp_path):
    assert resolve_workers(policy(tmp_path,1))==1

def test_explicit_cli_takes_precedence(tmp_path):
    assert resolve_workers(policy(tmp_path,1),explicit=2)==2

@pytest.mark.parametrize('value',[0,-1,True,1.5,'2'])
def test_invalid_worker_count_is_rejected(tmp_path,value):
    with pytest.raises(ValueError,match='positive integer'):
        resolve_workers(policy(tmp_path,value))

def test_wrong_run_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='does not match'):
        resolve_workers(policy(tmp_path,1,run_id='another'))

def test_wrong_policy_version_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='does not match'):
        resolve_workers(policy(tmp_path,1,version=2))
