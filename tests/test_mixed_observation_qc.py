import numpy as np
import pandas as pd
import pytest
from sprintseq.observation_repair.mixed_qc import observed_summary

def test_missing_is_not_zero_and_empty_is_unknown():
    frame=pd.DataFrame({'a':[2.,np.nan,4.],'b':[np.nan]*3})
    result=observed_summary(frame,['a','b'])
    assert result['a']['mean']==3
    assert result['a']['missing_rows']==1
    assert result['a']['measured_rows']==2
    assert result['b']['mean'] is None
    assert result['b']['missing_rows']==3

def test_infinity_is_not_missing():
    with pytest.raises(ValueError,match='Infinite'):
        observed_summary(pd.DataFrame({'a':[np.inf]}),['a'])
