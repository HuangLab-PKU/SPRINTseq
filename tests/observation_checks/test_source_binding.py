"""A cached preflight must not outlive registration, coverage or code inputs."""
import json
import pytest
from sprintseq.observation_checks.preflight import validate_readout_binding,ObservationHold
from sprintseq.observation_repair.__main__ import digest


@pytest.mark.parametrize('source_kind',['registration','acquisition','source_code','coverage'])
def test_changed_external_preflight_source_blocks_decode(tmp_path,source_kind):
    checks=tmp_path/'observation_checks';checks.mkdir();readout=tmp_path/'readout';readout.mkdir()
    image=tmp_path/'image';image.write_bytes(b'image');ref=tmp_path/'panel';ref.write_bytes(b'panel')
    source=tmp_path/'bound_input';source.write_bytes(b'original')
    sources={str(source):digest(source)}
    contract={'codebook_sha256':digest(ref),'image_identity':[image.stat().st_size,image.stat().st_mtime_ns]}
    scan={'source_image':str(image)}
    if source_kind in ['registration','acquisition']:contract[source_kind]={'sources':sources}
    elif source_kind=='source_code':contract['source_code']=sources
    else:scan['coverage_sources']=sources
    pre=checks/'preflight.json';pre.write_text(json.dumps({'contract':contract,'scan':scan}))
    (checks/'readout_binding.json').write_text(json.dumps({'preflight_receipt':str(pre),'preflight_sha256':digest(pre),'products':{}}))
    (readout/'intensity.csv').write_text('index,cyc_1_cy3\n1,100\n')
    validate_readout_binding(tmp_path,ref)
    source.write_bytes(b'changed')
    with pytest.raises(ObservationHold,match='CHANGED_PREFLIGHT_SOURCE'):validate_readout_binding(tmp_path,ref)
