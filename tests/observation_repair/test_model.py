import numpy as np
from scipy.stats import multivariate_normal
from sprintseq.observation_repair.model import acquisition_codes,log_likelihood,HistoryModel

def test_likelihood_matches_independent_scipy_and_ignores_missing_values():
    r=np.random.default_rng(3);x=r.normal(size=(7,6));m=r.normal(size=(5,6));a=r.normal(size=(6,6));cov=a@a.T+np.eye(6);mask=np.array([1,0,1,1,0,1],bool)
    actual=log_likelihood(x,m,cov,mask);expected=np.stack([multivariate_normal.logpdf(x[:,mask],mean=mu[mask],cov=cov[np.ix_(mask,mask)]) for mu in m],axis=1)
    assert np.allclose(actual,expected,atol=1e-10)
    x[:,~mask]=np.nan;assert np.allclose(actual,log_likelihood(x,m,cov,mask))

def test_two_pauses_and_terminal_nonobservation():
    x=np.arange(20).reshape(1,20);y=acquisition_codes(x,[5,8],[(88,99),(77,66)])
    assert y.tolist()==[[0,1,2,3,4,5,6,7,8,9,88,99,10,11,12,13,77,66,14,15]]
    x[:,-4:]=555;assert np.array_equal(y,acquisition_codes(x,[5,8],[(88,99),(77,66)]))

def test_unknown_without_evidence_and_history_limit():
    r=np.random.default_rng(4);codes=r.integers(0,2,(80,20));m={'all_codes':codes,'means':codes.astype(float)*3,'covariance':np.eye(20),'priors':np.ones(80)/80,'class_groups':np.arange(80)};h=HistoryModel(m,78)
    assert h.infer(np.zeros((2,20)),np.zeros(20,bool),[])['index'].tolist()==[-1,-1]
    assert h.infer(np.zeros((2,20)),np.ones(20,bool),[2,4,7])['index'].tolist()==[-1,-1]

def test_projected_collision_not_resolved_by_prior():
    r=np.random.default_rng(6);codes=r.integers(0,2,(80,20));codes[1]=codes[0];codes[1,-1]=1-codes[0,-1];m={'all_codes':codes,'means':codes.astype(float)*4,'covariance':np.eye(20)*.05,'priors':np.ones(80)/80,'class_groups':np.arange(80)};h=HistoryModel(m,78);mask=np.ones(20,bool);mask[-1]=False
    result=h.infer(m['means'][[0]],mask,[])
    assert result['index'][0]==-1 and result['probability'][0]==0
