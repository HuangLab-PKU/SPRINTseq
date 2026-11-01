import numpy as np
from sprintseq.observation_repair.detection_window import detection_support,supplement_cycles
from sprintseq.observation_repair.recover_detection import novel_candidates


def test_first_pause_moves_only_early_bright_base_outside_detection_window():
    codes=np.zeros((2,12),int);codes[0,6]=1;codes[1,2]=1
    assert detection_support(codes,[0,1,2,3]).tolist()==[1,1]
    assert detection_support(codes,[0,1,2,3],[0]).tolist()==[0,1]
    assert supplement_cycles(codes,[0,1,2,3],[0])==([4],[])


def test_single_late_discovery_remains_eligible_and_existing_spots_never_duplicate():
    coords=np.array([[10.,10.],[10.,11.],[1.,1.],[30.,30.]])
    result,bits=novel_candidates(coords,np.array([4,4,5,8]),np.array([10.,8.,6.,5.]),np.array([[0.,0.]]))
    assert result.tolist()==[[10.,10.],[30.,30.]]
    assert [int(b).bit_count() for b in bits]==[1,1]


def test_source_peak_localization_rejects_shoulders():
    from sprintseq.observation_repair.verify_signal_peaks import peak_supported
    image=np.zeros((25,25),np.float32);image[12,12]=100
    assert peak_supported(image,np.array([[12,12],[12,11],[12,9],[3,3]])).tolist()==[True,True,False,False]


def test_detection_supervisor_preserves_failure_and_releases_lock(tmp_path,monkeypatch):
    import pytest,json
    from sprintseq.observation_repair import detection_pipeline as pipeline
    run=tmp_path/'run_processed';(run/'observation_repair').mkdir(parents=True)
    def fail(*args,**kwargs):raise RuntimeError('measured child failure')
    monkeypatch.setattr(pipeline,'run_stage',fail)
    with pytest.raises(RuntimeError,match='measured child failure'):
        pipeline.run(run,tmp_path/'policy',tmp_path/'model',tmp_path/'output',-6000)
    assert not (run/'observation_repair/.pipeline.lock').exists()
    assert json.loads((tmp_path/'output/detection_pipeline_state.json').read_text())['status']=='FAILED_PRESERVE_CHECKPOINTS'
