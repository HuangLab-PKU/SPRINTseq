"""Collapse acquisition events strictly after the last measured cycle.

An unobserved future reaction event cannot alter an already acquired prefix.
Integrate those irrelevant history variables out before applying the qualified
limit on events that can affect measured observations. No missing base is filled.
"""
import numpy as np
from .model import HistoryModel


def relevant_pauses(observed,eligible):
    mask=np.asarray(observed,bool)
    if mask.ndim!=1 or len(mask)%2:raise ValueError('Expected flat two-channel validity')
    cycles=tuple(sorted(set(eligible)))
    if any(t<0 or t>=len(mask)//2 for t in cycles):raise ValueError('Pause cycle outside observed acquisition layout')
    if not mask.any():return ()
    last=int(np.flatnonzero(mask)[-1]//2)
    return tuple(t for t in cycles if t<=last)


class ObservableHistoryModel(HistoryModel):
    def infer(self,values,observed,eligible_pauses,*,max_events=2):
        eligible=tuple(eligible_pauses);relevant=relevant_pauses(observed,eligible)
        result=super().infer(values,observed,relevant,max_events=max_events)
        result['unobserved_future_pauses_marginalized']=[int(t) for t in eligible if t not in relevant]
        return result
