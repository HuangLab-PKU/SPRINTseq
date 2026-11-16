import numpy as np
from sprintseq.observation_repair.model import HistoryModel,acquisition_codes


def fixture():
    rng=np.random.default_rng(944)
    targets=np.array([[1,0,0,0,0,1],[0,1,1,0,0,0]],float)
    codes=np.vstack([targets,np.zeros((1,6)),rng.integers(0,2,(60,6))])
    weights=np.full(len(codes),1e-12);weights[0]=.9998;weights[1]=.0001;weights[2]=.0001;weights/=weights.sum()
    groups=np.r_[0,1,2,np.full(len(codes)-3,3)]
    return {'all_codes':codes,'means':4*codes,'covariance':np.eye(6)*.01,'priors':weights,'class_groups':groups}


def test_unrelated_aliased_history_does_not_delete_unique_current_call():
    model=fixture();hm=HistoryModel(model,2);mask=np.array([False,False,True,True,True,True])
    assert 0 in hm.ambiguous_targets(mask,[(),(0,)])
    result=hm.infer(model['means'][[0]],mask,[0],max_events=1)
    assert result['index'][0]==0 and result['probability'][0]>.99
    assert result['observability_unknown_probability'][0]<.01


def test_supported_collision_remains_unknown_even_with_extreme_gene_prior():
    model=fixture();hm=HistoryModel(model,2);mask=np.array([False,False,True,True,True,True])
    value=4*acquisition_codes(model['all_codes'][[0]],[0],[(0,0)])
    result=hm.infer(value,mask,[0],max_events=1)
    assert result['index'][0]==-1 and result['probability'][0]==0
    assert result['observability_unknown_probability'][0]>.99


def test_unique_pause_state_survives_aliased_normal_state_of_same_gene():
    model=fixture();hm=HistoryModel(model,2);mask=np.array([False,False,True,True,True,True])
    value=4*acquisition_codes(model['all_codes'][[1]],[0],[(0,0)])
    result=hm.infer(value,mask,[0],max_events=1)
    assert result['index'][0]==1 and result['probability'][0]>.99


def test_alias_masks_keep_non_target_states_and_cache_separate_layouts():
    model=fixture();hm=HistoryModel(model,2);mask=np.array([False,False,True,True,True,True]);histories=[(),(0,)]
    aliases=hm.ambiguous_state_masks(mask,histories)
    assert not aliases[0][0] and aliases[0][1]
    assert aliases[1][0] and not aliases[1][1]
    for history,alias in zip(histories,aliases):
        assert not alias[hm.hypothesis(history)[2]>=2].any()
    assert hm.ambiguous_state_masks(mask,histories) is aliases
    assert hm.ambiguous_state_masks(np.ones(6,bool),histories) is not aliases
