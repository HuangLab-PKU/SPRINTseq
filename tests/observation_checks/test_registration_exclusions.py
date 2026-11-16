import copy
import json
from pathlib import Path
import numpy as np
import pytest
from sprintseq.observation_checks.registration_exclusions import apply, native_bounds, validate, prepare_qc, STATUS
from sprintseq.observation_repair.__main__ import digest
from sprintseq.observation_repair.history_observability import ObservableHistoryModel


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value))


@pytest.fixture
def sources(tmp_path):
    run=tmp_path/'test_run_processed'; (run/'stitched').mkdir(parents=True)
    (run/'stitched/mosaic.ome.tif').write_bytes(b'unchanged-image-identity')
    shift=run/'shift_table.csv';shift.write_text('cyc_group,FocalStack_001.tif\ncyc_1_A,1.000000 2.000000\n')
    geometry=run/'stitched/geometry_float.csv';geometry.write_text('file,y,x\nFocalStack_001.tif,10,20\n')
    canvas=run/'stitched/canvas.json';write(canvas,{'origin':[-3,-5],'shape':[100,200]})
    e={'cycle':1,'group':'A','tile':'FocalStack_001.tif','placement_yx':[1.0000004,2.0000004],'status':'INFERRED_NEIGHBORS_PRECISION_UNKNOWN'}
    report=run/'registration_report.json';write(report,{'shift_table_sha256':digest(shift),'measurement':{'unresolved':[],'quality_exceptions':[e]}})
    image=run/'stitched/mosaic.ome.tif'; identity=[image.stat().st_size,image.stat().st_mtime_ns]
    evidence=tmp_path/'evidence.json';write(evidence,{'run_id':'test_run','exceptions':[e],'image_identity':identity,'native_shape':[10,10],'source_hashes':{str(report):digest(report),str(shift):digest(shift)},'products':{}})
    kernel=tmp_path/'kernel.py';kernel.write_text('# known readout kernel')
    row={'cycle':1,'group':'A','tile':e['tile'],'channels':['cy3','cy5'],'action':'EXCLUDE_UNQUALIFIED_MEASUREMENT','bounds_y0_y1_x0_x1':[6,32,19,45]}
    item={'version':1,'status':STATUS,'run_id':'test_run','scientific_accuracy_accepted':False,'sources':{str(report):digest(report),str(shift):digest(shift)},'geometry_sources':{str(p):digest(p) for p in [canvas,geometry]},'evidence_receipt':str(evidence),'evidence_receipt_sha256':digest(evidence),'image_identity':identity,'full_shape':[100,200],'native_shape':[10,10],'readout_source':str(kernel),'readout_source_sha256':digest(kernel),'readout_contract':{'method':'tophat','tophat_radius':3,'search_radius':1},'halo_pixels':8,'cases':[row]}
    manifest=tmp_path/'disposition.json';write(manifest,item)
    return run,manifest,item


def test_reject_bare_pass(tmp_path):
    p=tmp_path/'pass.json';write(p,{'status':'REGISTRATION_EXCEPTIONS_TECHNICALLY_RESOLVED'})
    with pytest.raises(ValueError,match='Unqualified registration'):validate(tmp_path,p)


def test_bound_all_cases_and_rounded_native_shift(sources):
    run,path,item=sources
    assert validate(run,path,item['sources'])['cases']==item['cases']


@pytest.mark.parametrize('change', ['omitted','duplicate','bounds','halo','acceptance','cycle'])
def test_invalid_disposition_holds(sources,change):
    run,path,item=sources
    if change=='omitted':item['cases']=[]
    elif change=='duplicate':item['cases']*=2
    elif change=='bounds':item['cases'][0]['bounds_y0_y1_x0_x1'][0]+=1
    elif change=='halo':item['halo_pixels']=0
    elif change=='acceptance':item['scientific_accuracy_accepted']=True
    elif change=='cycle':item['cases'][0]['cycle']=2
    write(path,item)
    with pytest.raises(ValueError):validate(run,path)


def test_source_drift_holds(sources):
    run,path,item=sources
    (run/'shift_table.csv').write_text('changed')
    with pytest.raises(ValueError,match='Stale'):validate(run,path)


def test_native_centres_negative_origin_and_clipping():
    assert native_bounds([0,0],[.2,-.2],[-1,-2],[10,10],[12,12],8)==[0,12,0,12]


def test_exclusion_locality_pixel_centres_and_nonchemical_semantics():
    reason=np.zeros((10,2,10,20),np.uint8); point=np.zeros((4,10,2),np.uint8)
    point[:,0]=3; point[1,0]=7
    reason[0]=3
    row={'cycle':1,'bounds_y0_y1_x0_x1':[20,40,50,70]}
    image,out=apply(reason,point,[49.49,49.5,69.49,69.5],[20,20,39,39],[100,200],[row])
    assert np.array_equal(out[[0,3]],point[[0,3]])
    assert out[1,0].tolist()==[20,20]  # source support retained; no chemistry bits
    assert out[2,0].tolist()==[16,16]
    assert not ((out[1:3,0]&3)>0).any()
    assert np.array_equal(image[1:],reason[1:])
    assert np.array_equal(point[:,0],np.array([[3,3],[7,7],[3,3],[3,3]],np.uint8))


def test_all_unobserved_stays_unknown_without_history():
    rng=np.random.default_rng(83);codes=rng.integers(0,2,(80,20))
    model=dict(all_codes=codes,means=codes.astype(float)*4,covariance=np.eye(20)*.05,priors=np.ones(80)/80,class_groups=np.arange(80))
    result=ObservableHistoryModel(model,78).infer(np.full((2,20),np.nan),np.zeros(20,bool),[])
    assert result['index'].tolist()==[-1,-1]
    assert result['probability'].tolist()==[0,0]


def test_composed_generation_preserves_base_and_unaffected_rows(sources,tmp_path):
    run,path,item=sources
    (run/'readout').mkdir(); transcript=run/'readout/transcripts.parquet';transcript.write_bytes(b'baseline-transcripts')
    base=tmp_path/'base';base.mkdir()
    points={'transcript_ids':np.array([1,2]),'x_px':np.array([25.,100.]),'y_px':np.array([15.,80.]),'reason':np.zeros((2,10,2),np.uint8),'observed':np.ones((2,10,2),bool)}
    np.savez_compressed(base/'point_validity.npz',**points);np.savez_compressed(base/'image_validity.npz',reason=np.zeros((10,2,10,20),np.uint8),full_shape=np.array([100,200]))
    qr={'run_id':'test_run','source_tiff_identity':item['image_identity'],'source_transcripts_sha256':digest(transcript),'source_code':{},'coverage_sources':{},'point_validity_sha256':digest(base/'point_validity.npz'),'image_validity_sha256':digest(base/'image_validity.npz'),'full_shape':[100,200],'reason_bits':{'SOURCE_MISSING':4},'limitations':[]}
    write(base/'receipt.json',qr); before={p.name:digest(p) for p in base.iterdir()}
    output=tmp_path/'combined';result=prepare_qc(run,base,output,path)
    assert {p.name:digest(p) for p in base.iterdir()}==before
    with np.load(output/'point_validity.npz') as v:
        assert (v['reason'][0,0]==16).all()
        assert np.array_equal(v['reason'][1],points['reason'][1])
        assert np.array_equal(v['transcript_ids'],points['transcript_ids'])
    assert result['registration_unqualified_transcripts']==1
    assert prepare_qc(run,base,output,path)['point_validity_sha256']==result['point_validity_sha256']
