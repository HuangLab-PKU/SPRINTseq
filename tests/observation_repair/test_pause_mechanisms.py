import copy
import numpy as np
import pytest
from sprintseq.observation_repair.model import HistoryModel, acquisition_codes
from sprintseq.observation_repair.pause_mechanisms import accepts_pause, infer_qualified_pause


def evidence(n, fraction):
    return dict(n=n, pause_fraction=fraction, halves=[dict(n=n//2, pause_fraction=fraction) for _ in range(2)])


def test_policy_requires_complementary_folds_controls_halves_and_local_support():
    affected=[evidence(200,.95) for _ in range(2)]; normal=[evidence(200,.05) for _ in range(2)]
    blocks=[dict(key=k, **evidence(80,.95)) for k in range(3)]
    assert accepts_pause(affected,normal,blocks)==[0,1,2]
    bad=copy.deepcopy(affected);bad[0]['halves'][0]['pause_fraction']=.60
    assert not accepts_pause(bad,normal,blocks)
    assert not accepts_pause(affected,[evidence(200,.4)]*2,blocks)
    assert not accepts_pause([evidence(99,.99)]*2,normal,blocks)
    assert not accepts_pause(affected,normal,blocks[:2])


def model():
    rng=np.random.default_rng(51);codes=rng.integers(0,2,(80,20))
    m=dict(all_codes=codes,means=codes.astype(float)*5,covariance=np.eye(20)*.02,
           priors=np.full(80,1/80),class_groups=np.arange(80))
    return HistoryModel(m,78)


def test_conditional_pause_recovers_shift_without_imputing_terminal_base():
    hm=model();mask=np.ones(20,bool);mask[:2]=False
    codes=acquisition_codes(hm.m['all_codes'],[0],[(0,0)])
    means=np.column_stack([codes,np.ones(len(codes))])@hm.coef
    idx=next(i for i in range(78) if i not in hm.ambiguous_targets(mask,[(0,)]))
    values=means[[idx]].copy();values[:,:2]=np.nan
    result=infer_qualified_pause(hm,values,mask)
    assert result['index'][0]==idx and result['probability'][0]>.99
    assert not result['veto'][0]
    with pytest.raises(ValueError):infer_qualified_pause(hm,values,np.ones(20,bool))


def test_strong_individual_optical_evidence_vetoes_region_assumption():
    hm=model();mask=np.ones(20,bool);mask[:2]=False
    values=hm.m['means'][:30].copy();values[:,:2]=np.nan
    result=infer_qualified_pause(hm,values,mask);veto=result['veto']
    assert veto.any()
    original=hm.infer(values[veto],mask,[0])
    assert np.array_equal(result['index'][veto],original['index'])
    assert np.allclose(result['probability'][veto],original['probability'])
