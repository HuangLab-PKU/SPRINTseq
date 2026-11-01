"""Fixed, label-independent gates for supplemental measured discovery."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class DiscoveryPolicy:
    block_size: int = 1024
    minimum_region_pixels: int = 64
    minimum_anchors: int = 100
    minimum_anchor_fraction: float = .5
    median_offset: float = 1.5
    p90_offset: float = 3.
    exclusion_radius: float = 3.
    peak_minimum: float = 50.
    posterior: float = .99
    history_posterior: float = .99
    allspot_cycle: int = 11
    discovery_window: int = 4

    def as_dict(self):
        return asdict(self)


def early_failure_regions(reason, policy=DiscoveryPolicy()):
    reason=np.asarray(reason)
    if reason.ndim!=4 or reason.shape[1]!=2:
        raise ValueError('Expected cycle/channel/y/x validity')
    footprint=~((reason&4)>0).all(axis=(0,1))
    failed=(reason[:policy.discovery_window]>0).any(axis=(0,1))&footprint
    labels,n=ndi.label(failed)
    counts=np.bincount(labels.ravel())
    allowed=np.flatnonzero(counts>=policy.minimum_region_pixels)
    allowed=allowed[allowed>0]
    return ndi.label(np.isin(labels,allowed))[0]


def alignment_gate(anchors, detected, candidates, policy=DiscoveryPolicy()):
    """Regional gate, then fixed local blocks; no threshold relaxation."""
    anchors=np.asarray(anchors).reshape(-1,2)
    detected=np.asarray(detected).reshape(-1,2)
    candidates=np.asarray(candidates).reshape(-1,2)
    empty=np.zeros(len(candidates),bool)
    if not len(anchors) or not len(detected):
        return empty,{'qualified':False,'reason':'NO_ALIGNMENT_ANCHORS','pairs':0,'blocks':[]}
    distance,near=cKDTree(detected).query(anchors,p=np.inf)
    back=cKDTree(anchors).query(detected,p=np.inf)[1]
    mutual=(back[near]==np.arange(len(anchors)))&(distance<=6)
    delta=np.max(abs(detected[near]-anchors),axis=1)
    def evaluate(pick):
        used=pick&mutual;n=int(used.sum());total=int(pick.sum())
        rec={'anchors':total,'pairs':n,'matched_fraction':n/max(total,1),'qualified':False}
        if n:
            rec.update(median=float(np.median(delta[used])),p90=float(np.quantile(delta[used],.9)))
            rec['qualified']=n>=policy.minimum_anchors and rec['matched_fraction']>=policy.minimum_anchor_fraction and rec['median']<=policy.median_offset and rec['p90']<=policy.p90_offset
        return rec
    regional=evaluate(np.ones(len(anchors),bool));regional['blocks']=[]
    if regional['qualified']:
        return np.ones(len(candidates),bool),regional
    akeys=np.floor(anchors/policy.block_size).astype(int)
    ckeys=np.floor(candidates/policy.block_size).astype(int)
    for key in np.unique(akeys,axis=0):
        rec=evaluate((akeys==key).all(axis=1));rec['block_yx']=key.tolist();regional['blocks'].append(rec)
        if rec['qualified']:empty|=(ckeys==key).all(axis=1)
    return empty,regional


def fallback_required(available_allspot, alignment):
    """Missing, dark, or poorly aligned all-spot data trigger measured late rounds."""
    return not available_allspot or not alignment.get('qualified',False)


def normalize_observed(raw, observed, model):
    raw=np.asarray(raw);observed=np.asarray(observed,bool)
    safe=np.where(observed,raw+model['log_add'],1.)
    if np.any(~np.isfinite(safe)) or np.any(safe<=0):
        raise ValueError('Invalid intensity in a measured dimension')
    result=(np.log10(safe)-model['data_log_mean'])/model['data_log_std']
    result[~observed]=np.nan
    return result


def candidate_gate(index,probability,history_probability,genes,alignment,
                   discovery_peak,sequence_peak,lost_support,policy=DiscoveryPolicy()):
    index=np.asarray(index);p=np.asarray(probability);hp=np.asarray(history_probability)
    return (index>=0)&(index<genes)&np.isfinite(p)&(p>policy.posterior)&np.isfinite(hp)&(hp>policy.history_posterior)&np.asarray(alignment,bool)&np.asarray(discovery_peak,bool)&np.asarray(sequence_peak,bool)&np.asarray(lost_support,bool)
