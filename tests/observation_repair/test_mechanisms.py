from copy import deepcopy
from sprintseq.observation_repair.mechanisms import decide_no_shift

def fixture():
    a={'n':200,'pause_fraction':.04,'halves':[{'n':100,'pause_fraction':.03},{'n':100,'pause_fraction':.05}]};c={'n':100,'pause_fraction':.04,'reference_n':100,'prefix_reference_agreement':1.};b=[{'key':i,'n':50,'pause_fraction':.04,'halves':[{'n':25,'pause_fraction':.04},{'n':25,'pause_fraction':.04}]} for i in range(4)];return a,c,b

def test_optical_region_requires_all_independent_gates():
    a,c,b=fixture();assert decide_no_shift(a,c,b)==[0,1,2,3]
    c['prefix_reference_agreement']=.98;assert decide_no_shift(a,c,b)==[]

def test_pause_and_conflicting_halves_keep_history_mixture():
    a,c,b=fixture();a['pause_fraction']=.9;assert decide_no_shift(a,c,b)==[]
    a,c,b=fixture();a['halves'][1]['pause_fraction']=.4;assert decide_no_shift(a,c,b)==[]

def test_spatially_mixed_block_not_overridden():
    a,c,b=fixture();b[0]['pause_fraction']=.8;assert decide_no_shift(a,c,b)==[1,2,3]
    b[1]['pause_fraction']=.8;assert decide_no_shift(a,c,b)==[]
