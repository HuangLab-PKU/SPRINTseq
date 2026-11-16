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

HISTORY_MODEL_VERSION = 'causal_acquisition_history_v2'
INFERENCE_MODEL_VERSION = 'state_resolved_observability_v3'


def causal_history_means(codes, coef, baseline_means, pauses, fills):
    """Apply only events already reached at each acquisition time.

    Normal prephasing coefficients are retained. A future experimental event,
    including its nuisance fill, cannot rewrite an already acquired prefix.
    With multiple pauses, each prefix retains its previously applicable history.
    """
    codes=np.asarray(codes);pauses=tuple(pauses);fills=tuple(fills)
    if codes.ndim!=2 or codes.shape[1]%2:raise ValueError('Two-channel code matrix required')
    cycles=codes.shape[1]//2
    if pauses!=tuple(sorted(set(pauses))) or len(fills)!=len(pauses) or any(not isinstance(t,(int,np.integer)) or t<0 or t>=cycles for t in pauses):raise ValueError('Ordered distinct valid acquisition events required')
    if np.asarray(baseline_means).shape!=codes.shape or np.asarray(coef).shape!=(codes.shape[1]+1,codes.shape[1]):raise ValueError('History mean dimensions differ')
    result=np.asarray(baseline_means,dtype=float).copy()
    for number,start in enumerate(pauses):
        stop=pauses[number+1] if number+1<len(pauses) else cycles
        acquired=acquisition_codes(codes,pauses[:number+1],fills[:number+1])
        # Calculate the same full matrix for a shared past, then select its time interval.
        # This preserves earlier-history values exactly when a later event is added.
        current=np.column_stack([acquired,np.ones(len(acquired))])@coef
        result[:,2*start:2*stop]=current[:,2*start:2*stop]
    return result

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
        np.linalg.cholesky(model['covariance']);self.cache={};self._state_alias_cache={}
    def hypothesis(self,pauses):
        pauses=tuple(pauses)
        if pauses in self.cache:return self.cache[pauses]
        if not pauses:result=(self.m['means'],self.m['priors'],self.groups)
        else:
            means=[];priors=[];groups=[];codes=self.m['all_codes'];count=4**len(pauses)
            for fills in product([(0,0),(0,1),(1,0),(1,1)],repeat=len(pauses)):
                mu=causal_history_means(codes,self.coef,self.m['means'],pauses,fills)
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
    def ambiguous_state_masks(self,observed,histories):
        """Mark colliding target states, not every history of an affected gene.

        A signature may occur under multiple histories or nuisance fills of the
        same gene without losing gene identity. Only signatures shared by
        distinct target genes are unidentifiable. Their posterior mass remains
        explicit UnknownObservation mass; it is never redistributed to genes.
        """
        mask=np.asarray(observed,bool);histories=tuple(tuple(h) for h in histories)
        if mask.ndim!=1 or len(mask)!=self.m['all_codes'].shape[1]:raise ValueError('Observed/model dimensions differ')
        key=(tuple(mask.tolist()),histories)
        if key in self._state_alias_cache:return self._state_alias_cache[key]
        state_masks=[np.zeros(len(self.hypothesis(h)[0]),bool) for h in histories];signatures={};total=len(self.m['all_codes'])
        for hi,history in enumerate(histories):
            for variant,fills in enumerate(product([(0,0),(0,1),(1,0),(1,1)],repeat=len(history))):
                codes=acquisition_codes(self.m['all_codes'][:self.K],history,fills)[:,mask]
                for gene,row in enumerate(codes):signatures.setdefault(tuple(row),{}).setdefault(gene,[]).append((hi,variant*total+gene))
        for genes in signatures.values():
            if len(genes)>1:
                for states in genes.values():
                    for hi,index in states:state_masks[hi][index]=True
        result=tuple(state_masks);self._state_alias_cache[key]=result;return result

    def infer(self,values,observed,eligible_pauses, *, max_events=2):
        n=len(values);mask=np.asarray(observed,bool);eligible=tuple(sorted(set(eligible_pauses)))
        G=int(self.groups.max())+1
        if not mask.any() or len(eligible)>max_events:
            return dict(index=np.full(n,-1),probability=np.zeros(n),probability_2=np.zeros(n),entropy=np.zeros(n),history_index=np.full(n,-1),history_probability=np.zeros(n),ambiguous=np.ones(n,bool),status='NO_EVIDENCE_OR_HISTORY_LIMIT',histories=[],inference_model_version=INFERENCE_MODEL_VERSION,observability_unknown_probability=np.ones(n))
        histories=[tuple(c) for k in range(len(eligible)+1) for c in combinations(eligible,k)]
        aliases=self.ambiguous_state_masks(mask,histories);posts=[];evidences=[]
        for history,alias in zip(histories,aliases):
            means,priors,groups=self.hypothesis(history);observable_groups=np.asarray(groups,int).copy();observable_groups[alias]=G
            post,evidence=grouped_posterior(log_likelihood(values,means,self.m['covariance'],mask)+np.log(priors)[None],observable_groups)
            if post.shape[1]==G:post=np.pad(post,((0,0),(0,1)))
            posts.append(post);evidences.append(evidence)
        posts=np.stack(posts,axis=1);ev=np.stack(evidences,axis=1);sensitivity=[];historyweights=[]
        for prior_pause in [.1,.5,.9]:
            logprior=np.array([len(h)*np.log(prior_pause)+(len(eligible)-len(h))*np.log1p(-prior_pause) for h in histories]);joint=ev+logprior[None];weights=np.exp(joint-logsumexp(joint,axis=1)[:,None]);sensitivity.append((posts*weights[:,:,None]).sum(axis=1));historyweights.append(weights)
        stack=np.stack(sensitivity);indices=stack.argmax(axis=2);best=indices[1].copy();stable=(indices==best[None]).all(axis=0);best_values=np.take_along_axis(stack,np.broadcast_to(best,(3,n))[:,:,None],axis=2)[:,:,0];prob=best_values.min(axis=0);chosen=best_values.argmin(axis=0);p=stack[chosen,np.arange(n)];ordered=np.sort(p,axis=1);entropy=-(p*np.log(np.maximum(p,1e-300))).sum(axis=1);hw=historyweights[1];hbest=hw.argmax(axis=1)
        ambiguous=~stable|(best==G);best[ambiguous]=-1;prob[ambiguous]=0;ordered[ambiguous,-2]=0;entropy[ambiguous]=0
        return dict(index=best,probability=prob,probability_2=ordered[:,-2],entropy=entropy,history_index=hbest,history_probability=hw.max(axis=1),ambiguous=ambiguous,status='OBSERVATION_REPAIRED_WITH_HISTORY_UNCERTAINTY',histories=histories,inference_model_version=INFERENCE_MODEL_VERSION,observability_unknown_probability=stack[:,:,G].max(axis=0))
