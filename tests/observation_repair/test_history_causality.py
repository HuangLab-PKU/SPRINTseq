import numpy as np
import pytest
from sprintseq.observation_repair.model import HistoryModel,log_likelihood,grouped_posterior


def model():
    d=8;codes=np.vstack([np.eye(d),np.zeros((1,d))]);coef=np.zeros((d+1,d));coef[:d]=np.eye(d);coef[-1]=np.linspace(.1,.8,d)
    # Molecular prephasing is allowed; future experimental events cannot alter past images.
    coef[2,0]=.6;coef[4,2]=.4;coef[6,4]=.3
    means=np.c_[codes,np.ones(len(codes))]@coef
    return {'all_codes':codes,'means':means,'covariance':np.eye(d),'priors':np.ones(d+1)/(d+1),'class_groups':np.arange(d+1)}


def test_single_event_and_nuisance_fill_preserve_acquired_prefix():
    m=model();hm=HistoryModel(m,8);mu,_,_=hm.hypothesis((1,));mu=mu.reshape(4,len(m['means']),8)
    assert np.array_equal(mu[:,:,:2],np.broadcast_to(m['means'][None,:,:2],mu[:,:,:2].shape))
    assert np.array_equal(hm.hypothesis(())[0],m['means'])


@pytest.mark.parametrize('first,last',[(0,1),(0,2),(1,2),(1,3),(2,3)])
def test_adding_future_event_preserves_earlier_history_not_normal_baseline(first,last):
    m=model();hm=HistoryModel(m,8);one=hm.hypothesis((first,))[0].reshape(4,len(m['means']),8);both=hm.hypothesis((first,last))[0].reshape(4,4,len(m['means']),8)
    assert np.array_equal(both[:,:,:,:2*last],np.broadcast_to(one[:,None,:,:2*last],both[:,:,:,:2*last].shape))
    assert np.any(one[:,:,2*first:2*last]!=m['means'][None,:,2*first:2*last])
    assert np.isclose(hm.hypothesis((first,last))[1].sum(),1)

def test_future_event_cannot_change_prefix_evidence_or_posteriors():
    m=model();hm=HistoryModel(m,8);observed=np.array([True]*4+[False]*4);values=m['means'][:3]+.1
    mu0,p0,g0=hm.hypothesis(());muf,pf,gf=hm.hypothesis((2,))
    a,ea=grouped_posterior(log_likelihood(values,mu0,m['covariance'],observed)+np.log(p0),g0)
    b,eb=grouped_posterior(log_likelihood(values,muf,m['covariance'],observed)+np.log(pf),gf)
    assert np.allclose(a,b,rtol=0,atol=1e-12) and np.allclose(ea,eb,rtol=0,atol=1e-12)


def test_first_event_preserves_legacy_response_and_background():
    from sprintseq.observation_repair.model import acquisition_codes
    m=model();hm=HistoryModel(m,8);expected=[]
    for fill in [(0,0),(0,1),(1,0),(1,1)]:
        codes=acquisition_codes(m['all_codes'],[0],[fill]);mu=np.c_[codes,np.ones(len(codes))]@hm.coef;mu[m['class_groups']==8]=m['means'][m['class_groups']==8];expected.append(mu)
    assert np.array_equal(hm.hypothesis((0,))[0],np.concatenate(expected))
    assert abs(hm.coef[2,0])>.5
