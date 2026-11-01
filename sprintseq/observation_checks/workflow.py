"""Standard supervised SPRINTseq flow with source gates and panel-wide acceptance.

Exit 2 means a recorded scientific/observation HOLD, not successful completion.
Subprocess exit codes are read natively; the outer mamba launcher's exit is not proof.
"""
import argparse,json,os,subprocess,sys,time,traceback,uuid
from pathlib import Path
from .preflight import run as preflight,ObservationHold
from sprintseq.observation_repair.__main__ import digest


def run(run_dir,ref_file,invocation=None,*,auto_repair=True,recover_existing=False,activate=True,allspot_cycle=11):
    run=Path(run_dir).resolve();ref=Path(ref_file).resolve();rid=run.name.removesuffix('_processed');invocation=invocation or uuid.uuid4().hex
    root=run/'observation_checks';root.mkdir(parents=True,exist_ok=True);statepath=root/'workflow_state.json';state={'status':'RUNNING','run_id':rid,'invocation':invocation,'steps':{}};lock=root/'workflow.lock'
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'invocation':invocation},f)
    def save():statepath.write_text(json.dumps(state,indent=2))
    try:
        previous=json.loads(statepath.read_text()) if statepath.exists() else {}
        active=run/'observation_repair/default_activation.json'
        if active.exists():
            entry=json.loads(active.read_text());receipt=Path(entry['receipt'])
            if digest(receipt)!=entry['receipt_sha256']:raise ObservationHold('HOLD_CHANGED_ACTIVE_RECEIPT')
            from spatial_cells.transcripts import read_footer
            footer=read_footer(run/'readout/transcripts.parquet')
            if (footer.get('codebook') or {}).get('sha256')!=digest(ref):raise ObservationHold('HOLD_ACTIVE_CODEBOOK_CHANGED')
            observation=run/'readout/observation_contract.json'
            if not observation.exists():raise ObservationHold('HOLD_ACTIVE_IMAGE_BINDING_UNAVAILABLE')
            binding=json.loads(observation.read_text());image=run/'stitched/mosaic.ome.tif'
            if [image.stat().st_size,image.stat().st_mtime_ns]!=binding['source_image_identity']:raise ObservationHold('HOLD_ACTIVE_IMAGE_CHANGED')

            from spatial_cells.observation_reentry import verify_existing
            state['verification']=verify_existing(run,receipt)
            state.update(status='EXISTING_DEFAULT_VERIFIED_NOT_REPLAYED',activation_receipt=str(receipt),scientific_accuracy_accepted=False);save();return state,0
        plan=preflight(run,ref);contract={'preflight_sha256':digest(plan['receipt']),'workflow_sha256':digest(__file__)};state['contract']=contract
        if previous.get('contract')==contract:state['steps']=previous.get('steps',{})
        attempts=root/'attempts'/invocation;attempts.mkdir(parents=True);save()
        def step(name,command,products,allow_hold=False):
            old=state['steps'].get(name)
            if old and old.get('products') and all(Path(p).is_file() and digest(p)==h for p,h in old['products'].items()):
                if old['exit_code']==0 or (allow_hold and old['exit_code']==2):return old['exit_code']
            with (attempts/(name+'.log')).open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);launch={'argv':command,'pid':child.pid,'started_at':time.time()};(attempts/(name+'_launch.json')).write_text(json.dumps(launch,indent=2))
                try:code=child.wait()
                finally:
                    if child.poll() is None:child.wait()
                    (attempts/(name+'_exit.json')).write_text(json.dumps({'actual_wait':True,'exit_code':child.returncode,'ended_at':time.time()},indent=2))
            if code not in ([0,2] if allow_hold else [0]):
                if code==2:raise ObservationHold('HOLD_'+name.upper()+': see '+str(attempts/(name+'.log')))
                raise RuntimeError(f'{name} actual child exit {code}; see {attempts/(name+".log")}')
            paths=products()
            if not paths or not all(Path(p).is_file() for p in paths):raise RuntimeError('Missing '+name+' products despite child exit '+str(code))
            state['steps'][name]={'exit_code':code,'products':{str(p):digest(p) for p in paths},'attempt':str(attempts)};save();return code
        common=['--run-id',rid,'--base-dir',str(run.parent)]
        snapshot=run/'codebook/codebook.csv'
        if snapshot.exists() and digest(snapshot)!=digest(ref):raise ObservationHold('HOLD_CODEBOOK_SNAPSHOT_MISMATCH')
        if not snapshot.exists():
            import shutil
            snapshot.parent.mkdir(exist_ok=True);shutil.copy2(ref,snapshot)
        if not recover_existing:
            step('readout',[sys.executable,'-B','-m','sprintseq.cli.main','readout',*common,'--ref-file',str(ref)],lambda:[run/'readout/position.csv',run/'readout/intensity.csv',root/'readout_binding.json'])
            step('decode',[sys.executable,'-B','-m','sprintseq.cli.main','gene-calling',*common,'--ref-file',str(ref)],lambda:[run/'readout/mapping_postcode.csv',root/'post_decode/receipt.json'],allow_hold=True)
        elif not all((run/'readout'/name).exists() for name in ['position.csv','intensity.csv','mapping_postcode.csv']):
            raise ObservationHold('HOLD_HISTORICAL_READOUT_INCOMPLETE')
        # A valid but artifact-flagged decode may be exported for repair; density is still gated.
        if not recover_existing or not all((run/'readout'/name).exists() for name in ['transcripts.parquet','transcript_intensities.parquet']):
            step('transcripts',[sys.executable,'-B','-m','spatial_cells.cli.main','transcripts','--platform','sprintseq',*common],lambda:[run/'readout/transcripts.parquet',run/'readout/transcript_intensities.parquet'])
        step('audit',[sys.executable,'-B','-m','sprintseq.observation_checks.audit','--run-dir',str(run),'--ref-file',str(ref)],lambda:[root/'post_decode/receipt.json'],allow_hold=True)
        audit=json.loads((root/'post_decode/receipt.json').read_text())
        if not auto_repair:
            if audit['status']!='PANEL_AUDIT_PASS_WITH_REPORTED_COVERAGE_LIMITS':raise ObservationHold(audit['status'])
            state.update(status='AUDIT_COMPLETE_NO_RECOVERY',scientific_accuracy_accepted=False);save();return state,0
        if audit['status']!='PANEL_AUDIT_PASS_WITH_REPORTED_COVERAGE_LIMITS' or plan.get('acquired_image_failure_pixels',plan.get('image_failure_pixels',0)):
            if not auto_repair:raise ObservationHold(audit['status'])
            from .recovery import run as recover
            state['status']='AUTOMATIC_RECOVERY_RUNNING';save()
            recovery=recover(run,ref,plan,audit,activate=activate,allspot_cycle=allspot_cycle)
            state.update(status=recovery['status'],recovery=recovery,scientific_accuracy_accepted=False);save();return state,0
        for f,h in audit['source_hashes'].items():
            if digest(f)!=h:raise ObservationHold('HOLD_CHANGED_AUDITED_SOURCES')
        import pandas as pd
        genes=pd.read_csv(ref).Gene.astype(str).tolist()
        step('density',[sys.executable,'-B','-m','spatial_cells.cli.main','density',*common,'--ref-file',str(ref)],lambda:[run/'readout/density_Q20'/f'{g}.tif' for g in genes])
        state['status']='COMPLETE_WITH_REPORTED_COVERAGE_LIMITS';state['scientific_accuracy_accepted']=False;save();return state,0
    except ObservationHold as error:
        state.update(status='HOLD',reason=str(error),recovery={'audit':str(root/'post_decode/receipt.json'),'readout_binding':str(root/'readout_binding.json'),'image_and_missing_data':'Repair/qualify acquisition and registration sources before resuming. Missing measurements are never zeros.','molecular_artifacts':'Use source-bound observation_repair after standard transcript export; keep failed/ambiguous regions unknown.','missed_detection':'Use qualified-history detection_pipeline for local measured additions; do not rerun the full accepted readout.','resume':'Rerun the same workflow after addressing the recorded cause; only source-matching checkpoints are reused.'});save();return state,2
    except BaseException:
        state.update(status='FAILED_PRESERVE_CHECKPOINTS',error=traceback.format_exc());save();raise
    finally:lock.unlink()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--base-dir',type=Path);p.add_argument('--ref-file',type=Path,required=True);p.add_argument('--invocation',default=None);p.add_argument('--exit-receipt',type=Path);p.add_argument('--audit-only',action='store_true');p.add_argument('--recover-existing',action='store_true');p.add_argument('--candidate-only',action='store_true');p.add_argument('--allspot-cycle',type=int,default=11);a=p.parse_args(argv)
    from sprintseq.cli.readout import BASE_DEST_DIRECTORY
    invocation=a.invocation or uuid.uuid4().hex;code=1
    try:
        state,code=run((a.base_dir or Path(BASE_DEST_DIRECTORY))/(a.run_id+'_processed'),a.ref_file,invocation,auto_repair=not a.audit_only,recover_existing=a.recover_existing,activate=not a.candidate_only,allspot_cycle=a.allspot_cycle);print(json.dumps(state),flush=True)
    finally:
        if a.exit_receipt:
            a.exit_receipt.parent.mkdir(parents=True,exist_ok=True);a.exit_receipt.write_text(json.dumps({'invocation':invocation,'native_workflow_exit':code,'ended_at':time.time()},indent=2))
    raise SystemExit(code)

if __name__=='__main__':main()
