import numpy as np
import pytest
from sprintseq.observation_repair.model import HistoryModel
from sprintseq.observation_repair.history_observability import ObservableHistoryModel,relevant_pauses


def fixture():
    rng=np.random.default_rng(83);codes=rng.integers(0,2,(80,20));return dict(all_codes=codes,means=codes.astype(float)*4,covariance=np.eye(20)*.05,priors=np.ones(80)/80,class_groups=np.arange(80))


def test_unmeasured_tail_events_do_not_destroy_measured_prefix():
    m=fixture();mask=np.ones(20,bool);mask[14:]=False;values=m['means'][:20].copy();values[:,14:]=np.nan
    actual=ObservableHistoryModel(m,78).infer(values,mask,[7,8,9]);expected=HistoryModel(m,78).infer(values,mask,[])
    assert np.array_equal(actual['index'],expected['index'])
    assert np.allclose(actual['probability'],expected['probability'],atol=0,rtol=0)
    assert actual['unobserved_future_pauses_marginalized']==[7,8,9]
    assert (actual['probability']>.99).any()


def test_relevant_event_limit_and_empty_evidence_still_hold():
    m=fixture();h=ObservableHistoryModel(m,78);mask=np.ones(20,bool);mask[[0,1,4,5,8,9]]=False
    assert h.infer(m['means'][:2],mask,[0,2,4])['index'].tolist()==[-1,-1]
    assert h.infer(m['means'][:2],np.zeros(20,bool),[0,1,2])['index'].tolist()==[-1,-1]
    assert relevant_pauses(np.r_[np.ones(14,bool),np.zeros(6,bool)],[0,7,8,9])==(0,)
    with pytest.raises(ValueError):relevant_pauses(np.ones(20,bool),[10])
