import numpy as np
import pandas as pd
import pytest
from sprintseq.observation_repair.localize import localize_calls


def test_localize_omits_roundoff_but_rejects_material_unqualified_changes(tmp_path):
    f=pd.DataFrame(dict(transcript_id=[1,2],x_px=[1.,2.],y_px=[3.,4.],feature_name=['A','B'],feature_class=['rna']*2,codeword_category=['gene']*2,codeword_index=[0,1],is_gene=[True]*2,probability=[.999,.999],probability_2=[.001,.001],entropy=[.01,.01],qv=[30.,30.]))
    c=dict(transcript_ids=np.array([1,2]),candidate_index=np.array([-1,1]),probability=np.array([0.,.999+1e-16]),probability_2=np.array([0.,.001]),entropy=np.array([0.,.01]))
    p=tmp_path/'calls.npz';np.savez(p,**c)
    with np.load(p) as calls:
        scoped=localize_calls(f,calls,['A','B'],np.array([1]))
    assert scoped['transcript_ids'].tolist()==[1]
    c['probability'][1]=.9;np.savez(p,**c)
    with np.load(p) as calls,pytest.raises(ValueError,match='Material change'):
        localize_calls(f,calls,['A','B'],np.array([1]))
