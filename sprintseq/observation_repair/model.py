"""Observed-dimension likelihoods and acquisition-history alternatives.

No pixel interpolation, missing-as-zero symbol or inferred terminal observation.
All target, background and infeasible latent classes remain in competition.
"""
from itertools import combinations,product
import numpy as np
from scipy.special import logsumexp
from scipy.sparse import csr_matrix

def acquisition_codes(codes, pauses, fills):
    x=np.asarray(codes).reshape(len(codes),-1,2);out=x.copy();j=0;lookup=dict(zip(pauses,fills))
    for t in range(x.shape[1]):
        if t in lookup:out[:,t]=lookup[t]
        else:out[:,t]=x[:,j];j+=1
    return out.reshape(len(codes),-1)

def mean_transform(codes,means):
    design=np.column_stack([codes,np.ones(len(codes))]);coef,_,rank,_=np.linalg.lstsq(design,means,rcond=None)
    error=float(np.max(np.abs(design@coef-means)))
    if rank!=design.shape[1] or error>2e-5:raise ValueError('Unidentified/inexact exported mean transform')
    return coef,error

def log_likelihood(values,means,covariance,observed):
    mask=np.asarray(observed,bool)
    if not mask.any():raise ValueError('No measured dimensions')
    x=np.asarray(values,float)[:,mask];m=np.asarray(means,float)[:,mask]
    if not np.isfinite(x).all():raise ValueError('Unmarked missing/nonfinite value')
    cov=np.asarray(covariance,float)[np.ix_(mask,mask)];L=np.linalg.cholesky(cov)
    xw=np.linalg.solve(L,x.T).T;mw=np.linalg.solve(L,m.T).T
    d2=(xw*xw).sum(axis=1)[:,None]+(mw*mw).sum(axis=1)[None]-2*xw@mw.T
    return -.5*(d2+int(mask.sum())*np.log(2*np.pi)+2*np.log(np.diag(L)).sum())

def grouped_posterior(logjoint,groups):
    norm=logsumexp(logjoint,axis=1);p=np.exp(logjoint-norm[:,None]);g=np.asarray(groups,int)
    agg=csr_matrix((np.ones(len(g)),(np.arange(len(g)),g)),shape=(len(g),int(g.max())+1))
    return np.asarray(p@agg),norm

class HistoryModel:
    def __init__(self,model,target_count):
        self.m=model;self.K=target_count;self.groups=np.asarray(model['class_groups'],int)
        self.coef,self.reconstruction_error=mean_transform(model['all_codes'],model['means'])
        if not np.isclose(model['priors'].sum(),1) or np.any(model['priors']<=0):raise ValueError('Invalid priors')
        np.linalg.cholesky(model['covariance']);self.cache={}
    def hypothesis(self,pauses):
        pauses=tuple(pauses)
        if pauses in self.cache:return self.cache[pauses]
        if not pauses:result=(self.m['means'],self.m['priors'],self.groups)
        else:
            means=[];priors=[];groups=[];codes=self.m['all_codes'];count=4**len(pauses)
            for fills in product([(0,0),(0,1),(1,0),(1,1)],repeat=len(pauses)):
                c=acquisition_codes(codes,pauses,fills);mu=np.column_stack([c,np.ones(len(c))])@self.coef
                mu[self.groups==self.K]=self.m['means'][self.groups==self.K]
                means.append(mu);priors.append(self.m['priors']/count);groups.append(self.groups)
            result=(np.concatenate(means),np.concatenate(priors),np.concatenate(groups))
        self.cache[pauses]=result;return result
    def ambiguous_targets(self,observed,histories):
        """Exact projected-code collisions cannot be resolved by class priors."""
        signatures={};codes=self.m['all_codes'][:self.K]
        for history in histories:
            c=acquisition_codes(codes,history,[(0,0)]*len(history))[:,observed]
            for g,row in enumerate(c):signatures.setdefault(tuple(row),set()).add(g)
        return {g for genes in signatures.values() if len(genes)>1 for g in genes}
    def infer(self,values,observed,eligible_pauses, *, max_events=2):
        n=len(values);mask=np.asarray(observed,bool);eligible=tuple(sorted(set(eligible_pauses)))
        G=int(self.groups.max())+1
        if not mask.any() or len(eligible)>max_events:
            return dict(index=np.full(n,-1),probability=np.zeros(n),probability_2=np.zeros(n),entropy=np.zeros(n),history_index=np.full(n,-1),history_probability=np.zeros(n),ambiguous=np.ones(n,bool),status='NO_EVIDENCE_OR_HISTORY_LIMIT',histories=[])
        histories=[tuple(c) for k in range(len(eligible)+1) for c in combinations(eligible,k)]
        posts=[];evidences=[]
        for h in histories:
            means,priors,groups=self.hypothesis(h);post,evidence=grouped_posterior(log_likelihood(values,means,self.m['covariance'],mask)+np.log(priors)[None],groups);posts.append(post);evidences.append(evidence)
        posts=np.stack(posts,axis=1);ev=np.stack(evidences,axis=1);sensitivity=[];historyweights=[]
        for prior_pause in [.1,.5,.9]:
            logprior=np.array([len(h)*np.log(prior_pause)+(len(eligible)-len(h))*np.log1p(-prior_pause) for h in histories]);joint=ev+logprior[None];weights=np.exp(joint-logsumexp(joint,axis=1)[:,None]);sensitivity.append((posts*weights[:,:,None]).sum(axis=1));historyweights.append(weights)
        stack=np.stack(sensitivity);indices=stack.argmax(axis=2);best=indices[1];stable=(indices==best[None]).all(axis=0);prob=np.min(np.take_along_axis(stack,np.broadcast_to(best,(3,n))[:,:,None],axis=2)[:,:,0],axis=0);chosen=np.argmin(np.take_along_axis(stack,np.broadcast_to(best,(3,n))[:,:,None],axis=2)[:,:,0],axis=0);p=stack[chosen,np.arange(n)];ordered=np.sort(p,axis=1);entropy=-(p*np.log(np.maximum(p,1e-300))).sum(axis=1);hw=historyweights[1];hbest=hw.argmax(axis=1)
        collisions=self.ambiguous_targets(mask,histories);ambiguous=~stable|np.isin(best,list(collisions));best[ambiguous]=-1;prob[ambiguous]=0;ordered[ambiguous,-2]=0;entropy[ambiguous]=0
        return dict(index=best,probability=prob,probability_2=ordered[:,-2],entropy=entropy,history_index=hbest,history_probability=hw.max(axis=1),ambiguous=ambiguous,status='OBSERVATION_REPAIRED_WITH_HISTORY_UNCERTAINTY',histories=histories)
