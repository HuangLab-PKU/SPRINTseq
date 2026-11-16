import copy,json
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq,pytest
from sprintseq.observation_repair.__main__ import digest
from sprintseq.observation_repair.qualify_observable_history import run as qualify
from sprintseq.observation_repair.history_observability import ObservableHistoryModel
from sprintseq.observation_repair.discovery_policy import DiscoveryPolicy
from sprintseq.observation_checks.recovery_policy import control_capability,adopt_seed


def fixture(tmp_path):
    run=tmp_path/'synthetic_processed';ro=run/'readout';ro.mkdir(parents=True);(run/'codebook').mkdir()
    qc=tmp_path/'qc';modeldir=tmp_path/'model';qc.mkdir();modeldir.mkdir()
    rng=np.random.default_rng(83);codes=rng.integers(0,2,(80,20));n=256;truth=np.arange(n)%78
    names=[f'gene{i}' for i in range(78)];layout=[f'r{t:02d}_{c}' for t in range(1,11) for c in ['cy3','cy5']]
    model=dict(all_codes=codes,means=codes.astype(float)*4,covariance=np.eye(20)*.05,priors=np.ones(80)/80,class_groups=np.arange(80),log_add=np.array(0.),data_log_mean=np.zeros(20),data_log_std=np.ones(20))
    np.savez(modeldir/'model.npz',**model)
    ids=np.arange(n);tr=pa.table({'transcript_id':ids,'feature_name':[names[t] for t in truth],'probability':np.full(n,.999),'is_gene':np.ones(n,bool)});pq.write_table(tr,ro/'transcripts.parquet')
    raw=pd.DataFrame(10**model['means'][truth],columns=layout);raw.insert(0,'transcript_id',ids);pq.write_table(pa.Table.from_pandas(raw),ro/'transcript_intensities.parquet')
    pd.DataFrame({'Gene':names,'Barcode':['A'*10]*78}).to_csv(run/'codebook/codebook.csv',index=False)
    np.savez(qc/'point_validity.npz',transcript_ids=ids,observed=np.ones((n,10,2),bool),reason=np.zeros((n,10,2),np.uint8))
    qr={'run_id':'synthetic','point_validity_sha256':digest(qc/'point_validity.npz')};(qc/'receipt.json').write_text(json.dumps(qr));(modeldir/'receipt.json').write_text(json.dumps({'gene_classes':78,'model_sha256':digest(modeldir/'model.npz'),'intensity_order':layout}))
    return run,qc,modeldir,model


def test_restricted_qualification_uses_accuracy_and_explicit_abstention(tmp_path):
    run,qc,md,model=fixture(tmp_path)
    result=qualify(run,qc,md,tmp_path/'control.json',max_events=0)
    control_capability(result,0)
    assert result['status']=='CONTROL_QUALIFICATION_PASS'
    conditions={c['condition']:c for c in result['conditions']}
    for name in ['injected_first_cycle_pause','image_failure_unknown_history','injected_single_pause','injected_double_pause']:
        assert conditions[name]['qualification_mode']=='REQUIRE_ALL_UNKNOWN' and conditions[name]['retained_Q20']==0
    assert conditions['terminal_optical_loss']['retained_Q20']>=50
    assert conditions['source_missing_single_channel']['disagreement_fraction']<=.01
    missing=copy.deepcopy(result);missing['conditions'].pop()
    with pytest.raises(ValueError,match='omit'):control_capability(missing,0)
    wrong=copy.deepcopy(result);wrong['conditions'][2]['retained_Q20']=1
    with pytest.raises(ValueError):control_capability(wrong,0)
    with pytest.raises(ValueError):control_capability(result,2)


def test_actual_discovery_abstains_for_disabled_relevant_history(tmp_path):
    from sprintseq.observation_repair.discovery import decode_candidates
    run,qc,md,m=fixture(tmp_path);hm=ObservableHistoryModel(m,78)
    rr=np.zeros((2,10,2),np.uint8);rr[0,0]=3
    points=np.array([[2.,2.],[20.,20.]]);raw=10**m['means'][[0,1]]
    peaks={(t,c):np.ones(2,bool) for t in range(1,12) for c in range(2)}
    output=decode_candidates(tmp_path,np.zeros((10,2,4,4),np.uint8),points,(32,32),raw,peaks,[11],np.ones(2,bool),m,hm,DiscoveryPolicy(max_history_events=0),point_reason=rr)
    assert output['candidate_index'][0]==-1 and output['probability'][0]==0 and not output['adoptable'][0]
    assert output['candidate_index'][1]==1 and output['probability'][1]>.99


def test_zero_history_limit_marginalizes_future_and_keeps_valid_prefix(tmp_path):
    run,qc,md,m=fixture(tmp_path);hm=ObservableHistoryModel(m,78);mask=np.ones(20,bool);mask[14:]=False
    actual=hm.infer(m['means'][:10],mask,[7,8,9],max_events=0)
    reference=hm.infer(m['means'][:10],mask,[],max_events=2)
    assert np.array_equal(actual['index'],reference['index']) and np.array_equal(actual['probability'],reference['probability'])
    assert hm.infer(m['means'][:10],mask,[2],max_events=0)['index'].tolist()==[-1]*10
    assert DiscoveryPolicy().max_history_events==2
    with pytest.raises(ValueError):DiscoveryPolicy(max_history_events=True)


def test_accepted_checkpoint_adoption_never_overwrites_or_executes(tmp_path):
    qc,me=tmp_path/'qc',tmp_path/'mechanisms';qc.mkdir();me.mkdir()
    for directory,names in [(qc,['receipt.json','point_validity.npz','image_validity.npz']),(me,['receipt.json','mechanism_policy.npz'])]:
        for name in names:(directory/name).write_bytes((str(directory)+name).encode())
    c=tmp_path/'control.json';c.write_text('source-validated-control');model=tmp_path/'original_model';model.mkdir()
    policy={'seed':{'qc_dir':str(qc),'model_dir':str(model),'mechanisms_dir':str(me),'control_receipt':str(c)},'max_history_events':0}
    target=tmp_path/'next';assert adopt_seed(target,policy)==model
    assert (target/'image_qc/point_validity.npz').read_bytes()==(qc/'point_validity.npz').read_bytes()
    (target/'image_qc/point_validity.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='overwrite'):adopt_seed(target,policy)
