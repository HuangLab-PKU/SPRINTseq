"""Opt-in pipeline: automatic validity -> repair -> versioned downstream revision.

No stage overwrites baseline data. Checkpoint reuse requires source/receipt
validation. The process owns actual child exits and a single-writer lock.
"""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from .__main__ import digest

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def check_hashes(mapping):
    for f,h in mapping.items():
        if digest(f)!=h:raise ValueError('Changed source '+f)

def resolve_recorded_source(run,path,expected_hash):
    """Resolve an immutable source after authorized default-directory activation."""
    run=Path(run).resolve();path=Path(path).resolve()
    if run not in path.parents:raise ValueError('Source is outside the recorded run')
    if path.is_file() and digest(path)==expected_hash:return path
    for relocation in sorted((run/'observation_repair').glob('*/baseline_relocation.json')):
        item=read(relocation).get(str(path))
        if item and item['sha256']==expected_hash:
            preserved=Path(item['preserved_at']).resolve()
            if run not in preserved.parents:raise ValueError('Snapshot source escapes run')
            if preserved.is_file() and digest(preserved)==expected_hash:return preserved
    raise ValueError('Recorded source no longer matches and has no verified baseline snapshot: '+str(path))

def run_stage(name,command,receipt,state,root,validator):
    receipt=Path(receipt)
    if receipt.exists():
        result=read(receipt);validator(result);state['steps'][name]={'action':'REUSED_SOURCE_VALIDATED_RECEIPT','receipt':str(receipt),'receipt_sha256':digest(receipt),'actual_child_exit':None};return result
    launch={'argv':command,'started_at':time.time()};(root/f'{name}_launch.json').write_text(json.dumps(launch,indent=2))
    with (root/f'{name}.log').open('ab') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait()
        finally:
            if child.poll() is None:child.wait()
            exit_record={'argv':command,'pid':child.pid,'actual_wait':True,'exit_code':child.returncode,'ended_at':time.time()};(root/f'{name}_exit.json').write_text(json.dumps(exit_record,indent=2))
    if code:raise RuntimeError(f'{name} failed with actual child exit {code}; inspect {root/name}.log')
    result=read(receipt);validator(result);state['steps'][name]={'action':'EXECUTED','receipt':str(receipt),'receipt_sha256':digest(receipt),'actual_child_exit':code};return result

def run(run_dir,root,model_dir=None,*,publish=False,fit_device='cpu',qc_dir=None,qualify_first_pause=False,localize_active=False,molecular_only=False):
    p=Path(run_dir).resolve();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);md=Path(model_dir).resolve() if model_dir else root/'model';rid=p.name.removesuffix('_processed');qroot=Path(qc_dir).resolve() if qc_dir else root/'image_qc';lock=p/'observation_repair/.pipeline.lock'
    if localize_active and not qualify_first_pause:raise ValueError('Localized extension requires first-pause qualification')
    revision_input=root/('localized_repair' if localize_active else 'repair')
    state={'status':'RUNNING','run_id':rid,'pid':os.getpid(),'steps':{}};sp=root/'pipeline_state.json';source=p/'readout/transcripts.parquet';intensity=p/'readout/transcript_intensities.parquet';image=p/'stitched/mosaic.ome.tif';imageid=[image.stat().st_size,image.stat().st_mtime_ns]
    lock.parent.mkdir(exist_ok=True);fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w') as handle:json.dump({'pid':os.getpid(),'run_id':rid,'output':str(root)},handle)
    def save():sp.write_text(json.dumps(state,indent=2))
    def model_valid(r):
        if r['run_id']!=rid or digest(md/'model.npz')!=r['model_sha256'] or digest(intensity)!=r['input_parquet_sha256'] or digest(p/'codebook/codebook.csv')!=r['panel_sha256']:raise ValueError('Model receipt/source mismatch')
        coverage=r.get('training_coverage')
        if coverage is not None and coverage['qc_receipt_sha256']!=digest(qroot/'receipt.json'):raise ValueError('Model training validity binding differs')
    def qc_valid(r):
        if r.get('axis_binding',{}).get('requested_channels')!=['cy3','cy5']:raise ValueError('Legacy validity lacks verified OME axis binding; regenerate with spatial_img_core.observation_qc.points')
        if r['run_id']!=rid or r['source_tiff_identity']!=imageid or digest(resolve_recorded_source(p,source,r['source_transcripts_sha256']))!=r['source_transcripts_sha256']:raise ValueError('QC receipt/source mismatch')
        check_hashes(r['source_code']);check_hashes(r['coverage_sources']);assert digest(qroot/'point_validity.npz')==r['point_validity_sha256'];assert digest(qroot/'image_validity.npz')==r['image_validity_sha256']
    def mechanism_valid(r):
        if r['run_id']!=rid or r['status']!='REGION_MECHANISM_POLICY_QUALIFIED':raise ValueError('Mechanism policy not qualified')
        check_hashes(r['source_code']);assert digest(qroot/'receipt.json')==r['qc_receipt_sha256'];assert digest(md/'receipt.json')==r['model_receipt_sha256'];assert digest(root/'mechanisms/mechanism_policy.npz')==r['policy_sha256']
    def control_valid(r):
        if r['run_id']!=rid or r['status']!='CONTROL_QUALIFICATION_PASS':raise ValueError('Control qualification failed')
        check_hashes(r['sources'])
    def repair_valid(r):
        if r['run_id']!=rid or digest(root/'repair/repair_calls.npz')!=r['repair_calls_sha256'] or digest(resolve_recorded_source(p,source,r['source_transcripts_sha256']))!=r['source_transcripts_sha256'] or digest(intensity)!=r['source_intensities_sha256']:raise ValueError('Repair source mismatch')
        check_hashes(r['source_code']);assert digest(root/'mechanisms/receipt.json')==r['mechanism_receipt_sha256'];assert digest(r['qc_receipt'])==r['qc_receipt_sha256'];assert digest(r['model_receipt'])==r['model_receipt_sha256']
    def revision_valid(r):
        if r['run_id']!=rid or r['status']!='MOLECULAR_REVISION_TECHNICALLY_VALIDATED_WITH_UNCERTAINTY':raise ValueError('Revision not accepted technically')
        assert digest(root/'revision/readout/transcripts.parquet')==r['transcripts_sha256'];assert digest(root/'revision/segmented/cell_gene_matrix_Q20_panel202.csv')==r['matrix_sha256'];assert digest(revision_input/'receipt.json')==r['source_repair_receipt_sha256'];assert digest(p/'segmented/cellsam_mask.tif')==r['mask_sha256'];assert digest(resolve_recorded_source(p,p/'segmented/transcript_assignment_cellsam.parquet',r['assignment_sha256']))==r['assignment_sha256']
        from spatial_cells.transcripts import validate_file,read_footer
        footer=read_footer(root/'revision/readout/transcripts.parquet')
        for f in [root/'revision/readout/transcripts.parquet',root/'revision/segmented/transcript_assignment_cellsam.parquet']:
            errors=validate_file(f,transcripts_footer=footer if 'assignment' in f.name else None)
            if errors:raise ValueError(errors)
    try:
        save();run_stage('image_qc',[sys.executable,'-B','-m','spatial_img_core.observation_qc.points','--run-dir',str(p),'--output',str(qroot)],qroot/'receipt.json',state,root,qc_valid);save()
        save();envold=os.environ.get('CUDA_VISIBLE_DEVICES')
        if fit_device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
        try:run_stage('model',[sys.executable,'-B','-m','sprintseq.observation_repair.export_model','--run-dir',str(p),'--output',str(md),'--qc-dir',str(qroot)],md/'receipt.json',state,root,model_valid)
        finally:
            if envold is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=envold

        run_stage('mechanisms',[sys.executable,'-B','-m','sprintseq.observation_repair.mechanisms','--run-dir',str(p),'--qc-dir',str(qroot),'--model-dir',str(md),'--output',str(root/'mechanisms')],root/'mechanisms/receipt.json',state,root,mechanism_valid);save()
        run_stage('control_qualification',[sys.executable,'-B','-m','sprintseq.observation_repair.qualify_observable_history','--run-dir',str(p),'--qc-dir',str(qroot),'--model-dir',str(md),'--output',str(root/'control_qualification_final.json')],root/'control_qualification_final.json',state,root,control_valid);save()
        repair_module='sprintseq.observation_repair.observable_base_repair';pause_args=[]
        if qualify_first_pause:
            def pause_valid(r):
                assert r['status']=='FIRST_PAUSE_REGION_POLICY_QUALIFIED' and r['run_id']==rid
                assert r['qc_receipt_sha256']==digest(qroot/'receipt.json') and r['model_receipt_sha256']==digest(md/'receipt.json')
                assert r['policy_sha256']==digest(root/'pause_mechanisms/pause_policy.npz');check_hashes(r['source_code'])
            run_stage('pause_mechanisms',[sys.executable,'-B','-m','sprintseq.observation_repair.pause_mechanisms','--run-dir',str(p),'--qc-dir',str(qroot),'--model-dir',str(md),'--output',str(root/'pause_mechanisms')],root/'pause_mechanisms/receipt.json',state,root,pause_valid);save()
            run_stage('pause_control_qualification',[sys.executable,'-B','-m','sprintseq.observation_repair.qualify_pause','--run-dir',str(p),'--qc-dir',str(qroot),'--model-dir',str(md),'--output',str(root/'pause_control_qualification.json')],root/'pause_control_qualification.json',state,root,control_valid);save()
            repair_module='sprintseq.observation_repair.observable_repair';pause_args=['--pause-mechanism-dir',str(root/'pause_mechanisms')]
        run_stage('repair',[sys.executable,'-B','-m',repair_module,'--run-dir',str(p),'--qc-dir',str(qroot),'--model-dir',str(md),'--mechanism-dir',str(root/'mechanisms'),'--output',str(root/'repair')]+pause_args,root/'repair/receipt.json',state,root,repair_valid);save()
        if localize_active:
            def local_valid(r):
                assert r['outside_policy_preserved_from_active_baseline'] and r['run_id']==rid
                assert digest(revision_input/'repair_calls.npz')==r['repair_calls_sha256']
                assert digest(root/'repair/receipt.json')==r['parent_repair_receipt_sha256'];check_hashes(r['source_code'])
            run_stage('localize',[sys.executable,'-B','-m','sprintseq.observation_repair.localize','--run-dir',str(p),'--repair-dir',str(root/'repair'),'--output',str(revision_input)],revision_input/'receipt.json',state,root,local_valid);save()
        if molecular_only:
            model_valid(read(md/'receipt.json'));qc_valid(read(qroot/'receipt.json'));repair_valid(read(root/'repair/receipt.json'));control_valid(read(root/'control_qualification_final.json'));mechanism_valid(read(root/'mechanisms/receipt.json'))
            if qualify_first_pause:
                pause_valid(read(root/'pause_mechanisms/receipt.json'));control_valid(read(root/'pause_control_qualification.json'))
            state.update(status='MOLECULAR_REPAIR_QUALIFIED',repair=str(revision_input),scientific_accuracy_accepted=False);save();return state
        revision=run_stage('revision',[sys.executable,'-B','-m','spatial_cells.observation_revision_current_assignment','--run-dir',str(p),'--repair-dir',str(revision_input),'--output',str(root/'revision')],root/'revision/receipt.json',state,root,revision_valid);save()
        model_valid(read(md/'receipt.json'));qc_valid(read(qroot/'receipt.json'));repair_valid(read(root/'repair/receipt.json'));control_valid(read(root/'control_qualification_final.json'));mechanism_valid(read(root/'mechanisms/receipt.json'))
        if qualify_first_pause:
            pause_valid(read(root/'pause_mechanisms/receipt.json'));control_valid(read(root/'pause_control_qualification.json'))
            assert read(root/'repair/receipt.json')['pause_policy_receipt_sha256']==digest(root/'pause_mechanisms/receipt.json')
        products={str(f):digest(f) for f in sorted((root/'revision').rglob('*')) if f.is_file()};manifest={'status':'REVISION_READY_WITH_REPORTED_UNCERTAINTY','run_id':rid,'revision':str(root/'revision'),'revision_receipt_sha256':digest(root/'revision/receipt.json'),'control_receipt_sha256':digest(root/'control_qualification_final.json'),'products':products,'baseline_preserved':True,'scientific_accuracy_accepted':False};(root/'publication_manifest.json').write_text(json.dumps(manifest,indent=2))
        if publish:
            pointer=p/'observation_repair/current.json'
            if pointer.exists():
                previous=read(pointer);history=pointer.parent/'pointer_history';history.mkdir(exist_ok=True);saved=history/(digest(pointer)+'.json')
                if not saved.exists():saved.write_text(json.dumps(previous,indent=2))
            entry={'run_id':rid,'revision':str(root/'revision'),'manifest':str(root/'publication_manifest.json'),'manifest_sha256':digest(root/'publication_manifest.json'),'status':manifest['status'],'baseline_preserved':True}
            activation=p/'observation_repair/default_activation.json'
            if activation.exists():
                ar=read(activation);assert digest(ar['receipt'])==ar['receipt_sha256'];active=read(ar['receipt'])
                if Path(active['revision']).resolve()==(root/'revision').resolve() and digest(p/'readout/transcripts.parquet')==revision['transcripts_sha256'] and digest(p/'segmented/cell_gene_matrix_Q20_panel202.csv')==revision['matrix_sha256']:
                    entry.update(default_entrypoint_active=True,default_activation=str(activation),canonical_readout=str(p/'readout'),canonical_segmented=str(p/'segmented'))
            temp=pointer.with_name(f'current_{os.getpid()}.tmp');temp.write_text(json.dumps(entry,indent=2));os.replace(temp,pointer)
        state.update(status='COMPLETE',published_current_pointer=publish,revision=str(root/'revision'));save();print(json.dumps(state),flush=True);return state
    except BaseException:
        state.update(status='FAILED_PRESERVE_CHECKPOINTS',error=traceback.format_exc());save();raise
    finally:lock.unlink()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--model-dir',type=Path);p.add_argument('--qc-dir',type=Path);p.add_argument('--publish',action='store_true');p.add_argument('--molecular-only',action='store_true');p.add_argument('--qualify-first-pause',action='store_true');p.add_argument('--localize-active',action='store_true');p.add_argument('--fit-device',choices=['cpu','auto'],default='cpu');a=p.parse_args();run(a.run_dir,a.output_root,a.model_dir,publish=a.publish,fit_device=a.fit_device,qc_dir=a.qc_dir,qualify_first_pause=a.qualify_first_pause,localize_active=a.localize_active,molecular_only=a.molecular_only)
if __name__=='__main__':main()
