"""Explicit source-bound recovery capabilities and accepted-checkpoint adoption."""
import json,shutil
from pathlib import Path
from sprintseq.check_paths import checks_directory
from sprintseq.observation_repair.__main__ import digest

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(ok,message):
    if not ok:raise ValueError(message)
def hashes(mapping):
    for p,h in mapping.items():require(digest(p)==h,'Changed recovery source '+str(p))

def control_capability(receipt, maximum):
    require(receipt.get('status')=='CONTROL_QUALIFICATION_PASS' and receipt.get('max_history_events',2)==maximum,'Control is not qualified for the declared capability')
    require(receipt.get('conditions') and all(c.get('pass_reference_gate') for c in receipt['conditions']),'An enabled control gate failed')
    if maximum==0:
        expected={'terminal_optical_loss','terminal_source_loss','injected_first_cycle_pause','source_missing_single_channel','optical_missing_both','image_failure_unknown_history','injected_single_pause','injected_double_pause'}
        require(len(receipt['conditions'])==len(expected) and {c['condition'] for c in receipt['conditions']}==expected,'Restricted controls omit/duplicate a required condition')
        for c in receipt['conditions']:
            if c.get('qualification_mode')=='REQUIRE_ALL_UNKNOWN':
                require(c['retained_Q20']==0 and c['retained_disagreements']==0,'Unsupported history made a retained call')
            else:
                require(c['retained_Q20']>=50 and c['disagreement_fraction'] is not None and c['disagreement_fraction']<=.01,'Retained accuracy gate relaxed')

def load_policy(run_dir):
    run=Path(run_dir);path=checks_directory(run)/'recovery_policy.json'
    if not path.exists():return {'max_history_events':2,'seed':None,'source_bindings':{}}
    p=read(path);rid=run.name.removesuffix('_processed')
    require(p.get('version')==1 and p.get('run_id')==rid and p.get('max_history_events')==0 and p.get('unsupported_history_action')=='UNKNOWN_OBSERVATION_Q0','Unqualified capability policy')
    failure=Path(p['failed_control_receipt']);require(digest(failure)==p['failed_control_sha256'],'Changed failed control evidence')
    failed=read(failure);require(failed.get('run_id')==rid and failed.get('status')=='CONTROL_QUALIFICATION_HOLD','Missing real prior failed qualification')
    failures={c['condition'] for c in failed['conditions'] if not c['pass_reference_gate']}
    require(failures and failures<={'injected_first_cycle_pause','injected_single_pause','injected_double_pause','image_failure_unknown_history'},'Failure exceeds disabled history capability')
    snapshots=p.get('historical_source_snapshots',{})
    for source,h in failed['sources'].items():
        if Path(source).exists() and digest(source)==h:continue
        require(source in snapshots and digest(snapshots[source])==h,'Historical failed-control implementation is not preserved')
    native_failure=read(p['failed_stage_exit'])
    require(digest(p['failed_stage_exit'])==p['failed_stage_exit_sha256'] and native_failure.get('actual_wait') and native_failure.get('exit_code')==2,'Prior control native failure missing')
    seed=p['seed'];qc,model,mechanisms=map(Path,[seed['qc_dir'],seed['model_dir'],seed['mechanisms_dir']]);control=Path(seed['control_receipt'])
    bindings={str(path):digest(path),str(failure):digest(failure),str(Path(p['failed_stage_exit'])):digest(p['failed_stage_exit'])}
    for directory,key in [(qc,'qc_receipt_sha256'),(model,'model_receipt_sha256'),(mechanisms,'mechanism_receipt_sha256')]:
        require(digest(directory/'receipt.json')==seed[key],'Changed accepted seed receipt');bindings[str(directory/'receipt.json')]=seed[key]
    qr,mr,me=read(qc/'receipt.json'),read(model/'receipt.json'),read(mechanisms/'receipt.json')
    require(qr['run_id']==mr['run_id']==me['run_id']==rid,'Seed belongs to another run')
    image=run/'stitched/mosaic.ome.tif'
    require(qr['source_tiff_identity']==[image.stat().st_size,image.stat().st_mtime_ns] and qr['source_transcripts_sha256']==digest(run/'readout/transcripts.parquet'),'Seed image/transcripts changed')
    require(mr['status']=='NEW_FIT_EXPORTED_FULL_OBSERVATION_PARITY_PASS' and mr['input_parquet_sha256']==digest(run/'readout/transcript_intensities.parquet') and mr['panel_sha256']==digest(run/'codebook/codebook.csv'),'Seed fitted model inputs changed')
    require(mr['training_coverage']['qc_receipt_sha256']==seed['qc_receipt_sha256'] and mr['training_coverage']['invalid_dimensions_used_for_training'] is False,'Seed training validity differs')
    hashes(qr['source_code']);hashes(qr['coverage_sources']);hashes(mr['source_modules']);hashes(me['source_code'])
    require(digest(qc/'point_validity.npz')==qr['point_validity_sha256'] and digest(qc/'image_validity.npz')==qr['image_validity_sha256'] and digest(model/'model.npz')==mr['model_sha256'],'Accepted seed arrays changed')
    require(me['status']=='REGION_MECHANISM_POLICY_QUALIFIED' and me['qc_receipt_sha256']==seed['qc_receipt_sha256'] and me['model_receipt_sha256']==seed['model_receipt_sha256'] and digest(mechanisms/'mechanism_policy.npz')==me['policy_sha256'],'Seed mechanism policy differs')
    require(digest(control)==seed['control_receipt_sha256'],'New restricted control receipt changed')
    cr=read(control);require(cr['run_id']==rid,'Control run differs');control_capability(cr,0);hashes(cr['sources'])
    native=read(seed['control_exit_receipt']);require(digest(seed['control_exit_receipt'])==seed['control_exit_sha256'] and native.get('actual_wait') and native.get('exit_code')==0,'Restricted control lacks real native success')
    require(cr['sources'].get(str(qc/'receipt.json'))==seed['qc_receipt_sha256'] and cr['sources'].get(str(model/'receipt.json'))==seed['model_receipt_sha256'],'Restricted controls do not bind the accepted model/QC')
    bindings.update({str(control):seed['control_receipt_sha256'],str(Path(seed['control_exit_receipt'])):seed['control_exit_sha256']})
    p['source_bindings']=bindings;p['policy_file']=str(path)
    return p

def adopt_seed(root,policy):
    """Copy only identical accepted receipts/arrays; reuse the original model."""
    seed=policy['seed'];root=Path(root)
    pairs=[]
    for source,dest,names in [(Path(seed['qc_dir']),root/'image_qc',['receipt.json','point_validity.npz','image_validity.npz']),(Path(seed['mechanisms_dir']),root/'mechanisms',['receipt.json','mechanism_policy.npz'])]:
        pairs.extend((source/name,dest/name) for name in names)
    pairs.append((Path(seed['control_receipt']),root/'control_qualification_final.json'))
    recorded={}
    for source,target in pairs:
        expected=digest(source);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(digest(target)==expected,'Do not overwrite a changed accepted checkpoint')
        else:shutil.copy2(source,target)
        require(digest(target)==expected,'Checkpoint adoption copy mismatch')
        recorded[str(target)]={'adopted_from':str(source),'sha256':expected,'numerical_execution_repeated':False}
    (root/'accepted_checkpoint_adoption.json').write_text(json.dumps({'products':recorded,'model_reused_at':seed['model_dir'],'max_history_events':policy['max_history_events'],'unsupported_history_action':'UNKNOWN_OBSERVATION_Q0'},indent=2))
    return Path(seed['model_dir'])
