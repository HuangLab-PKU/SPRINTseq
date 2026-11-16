import numpy as np
from sprintseq.observation_repair.frozen_observed import FrozenObservedModel


def test_missing_dimensions_never_affect_frozen_marginal_calls():
    state={'feature_order':[f'cyc_{t}_{c}' for t in range(1,11) for c in ['cy3','cy5']],'gene_names':['A','B'],'theta':np.array([[0]*20,[1]*20,[-1]*20]),'sigma':np.eye(20),'weights':np.array([.4,.4,.2]),'log_add':np.ones(20),'log_mean':np.zeros(20),'log_std':np.ones(20),'background_indices':[2],'infeasible_indices':[]}
    model=FrozenObservedModel(state);x=np.ones((3,20));mask=np.ones(20,bool);mask[[1,11]]=False;x[:,~mask]=np.nan;a=model.infer(x,mask);x[:,~mask]=1e9;b=model.infer(x,mask)
    assert np.array_equal(a['Gene'],b['Gene']) and np.array_equal(a['Probability'],b['Probability'])
    assert model.infer(x,np.zeros(20,bool))['Gene'].tolist()==['UnknownObservation']*3
