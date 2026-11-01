"""Check detection coverage after a qualified sequencing-history shift."""
import numpy as np
from .model import acquisition_codes


def detection_support(codes, detection_cycles, pauses=()):
    """Count observable bright dimensions per code; cycles and pauses are zero-based."""
    codes=np.asarray(codes);T=codes.shape[1]//2
    if codes.ndim!=2 or codes.shape[1]%2 or not set(detection_cycles)<=set(range(T)) or not set(pauses)<=set(range(T)):
        raise ValueError('Invalid code dimensions/cycle indices')
    acquired=acquisition_codes(codes,pauses,[(0,0)]*len(pauses)).reshape(len(codes),T,2)
    return acquired[:,list(detection_cycles)].sum(axis=(1,2))


def supplement_cycles(codes, detection_cycles, pauses=()):
    """Greedily cover lost detection support without consulting expression counts."""
    selected=list(detection_cycles);extra=[];T=np.asarray(codes).shape[1]//2
    while True:
        missing=detection_support(codes,selected,pauses)==0
        if not missing.any():return extra,[]
        choices=[t for t in range(T) if t not in selected and t not in pauses]
        gain={t:int((detection_support(np.asarray(codes)[missing],[t],pauses)>0).sum()) for t in choices}
        if not gain or max(gain.values())==0:return extra,np.flatnonzero(missing).tolist()
        best=max(choices,key=lambda t:(gain[t],-t));selected.append(best);extra.append(best)
