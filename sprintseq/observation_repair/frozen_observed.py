"""Gaussian marginal inference from a verified actual-fit snapshot, without refitting."""
from pathlib import Path
import numpy as np
from .__main__ import digest
from .model import log_likelihood,grouped_posterior


class FrozenObservedModel:
    def __init__(self,state):
        fields=[f'cyc_{t}_{c}' for t in range(1,11) for c in ['cy3','cy5']]
        if state['feature_order']!=fields:raise ValueError('Unsupported fitted feature order')
        def array(key):
            value=state[key]
            return np.asarray(value.detach().cpu().numpy() if hasattr(value,'detach') else value,dtype=float)
        self.fields=fields;self.genes=list(state['gene_names']);self.means=array('theta');self.covariance=array('sigma');self.weights=array('weights').reshape(-1);self.add=array('log_add').reshape(-1);self.mean=array('log_mean').reshape(-1);self.std=array('log_std').reshape(-1)
        if self.means.shape!=(len(self.weights),20) or self.covariance.shape!=(20,20) or any(x.shape!=(20,) for x in [self.add,self.mean,self.std]):raise ValueError('Invalid snapshot dimensions')
        if not all(np.isfinite(x).all() for x in [self.means,self.covariance,self.weights,self.add,self.mean,self.std]) or np.any(self.weights<0) or self.weights.sum()<=0 or np.any(self.std<=0):raise ValueError('Invalid snapshot values')
        np.linalg.cholesky(self.covariance);self.groups=np.full(len(self.weights),-1,int);self.groups[:len(self.genes)]=np.arange(len(self.genes));self.labels=self.genes.copy();seen=list(range(len(self.genes)))
        for name,key in [('Background','background_indices'),('Infeasible','infeasible_indices')]:
            indices=list(state[key])
            if indices:self.groups[indices]=len(self.labels);self.labels.append(name);seen+=indices
        if sorted(seen)!=list(range(len(self.weights))) or len(seen)!=len(set(seen)):raise ValueError('Invalid class collapse')
        self.log_weights=np.full(len(self.weights),-np.inf);positive=self.weights>0;self.log_weights[positive]=np.log(self.weights[positive])

    @classmethod
    def load(cls,path,expected_sha256):
        if digest(path)!=expected_sha256:raise ValueError('Actual-fit snapshot changed')
        import torch
        # Only the source-hash-verified snapshot produced by the user's pipeline is loaded.
        state=torch.load(Path(path),map_location='cpu',weights_only=False)
        if state.get('schema')!='sprintseq.frozen_actual_postcode_fit.v1':raise ValueError('Unknown fit snapshot schema')
        return cls(state)

    def infer(self,raw,observed):
        raw=np.asarray(raw,float);mask=np.asarray(observed)
        if raw.ndim!=2 or raw.shape[1]!=20 or mask.dtype!=bool or mask.shape!=(20,):raise ValueError('Explicit 20-dimensional observations required')
        if not mask.any():return dict(Gene=np.full(len(raw),'UnknownObservation',object),Probability=np.zeros(len(raw)),Probability_2=np.zeros(len(raw)),Entropy=np.zeros(len(raw)))
        selected=raw[:,mask]
        if not np.isfinite(selected).all() or np.any(selected<0) or np.any(selected+self.add[mask]<=0):raise ValueError('Invalid observed measurements')
        normalized=np.full(raw.shape,np.nan);normalized[:,mask]=(np.log10(selected+self.add[mask])-self.mean[mask])/self.std[mask]
        post,_=grouped_posterior(log_likelihood(normalized,self.means,self.covariance,mask)+self.log_weights,self.groups);order=np.argsort(post,axis=1);best=order[:,-1];second=order[:,-2]
        return dict(Gene=np.asarray(self.labels,object)[best],Probability=np.clip(post[np.arange(len(post)),best],0,1),Probability_2=np.clip(post[np.arange(len(post)),second],0,1),Entropy=-(post*np.log(np.maximum(post,1e-10))).sum(axis=1))
