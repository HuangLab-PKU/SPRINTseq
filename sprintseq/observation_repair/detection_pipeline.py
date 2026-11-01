"""Supervise measured spot recovery after qualified reaction-history repair.

This stage emits a versioned candidate and matched normal-region control. It
never silently publishes, activates, imputes density or modifies old spot IDs.
"""
import argparse,json,os,sys,traceback
from pathlib import Path
from .pipeline import run_stage,resolve_recorded_source
from .__main__ import digest


def run(run_dir,policy_dir,model_dir,output_root,control_offset_x):
    run,policy,model,root=map(lambda p:Path(p).resolve(),[run_dir,policy_dir,model_dir,output_root])
    if not control_offset_x:raise ValueError('An independent normal-region control offset is required')
    root.mkdir(parents=True,exist_ok=True);lock=run/'observation_repair/.pipeline.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'run_id':run.name,'output':str(root)},f)
    state={'status':'RUNNING','steps':{}};statefile=root/'detection_pipeline_state.json'
    def save():statefile.write_text(json.dumps(state,indent=2))
    def validate_sources(receipt):
        for f,h in receipt['source_hashes'].items():
            path=Path(f)
            if run in path.resolve().parents:resolve_recorded_source(run,path,h)
            elif digest(path)!=h:raise ValueError('Detection source changed: '+f)
    try:
        save()
        for label,offset in [('affected',0),('normal',control_offset_x)]:
            dest=root/label
            def validate_detection(r,dest=dest):
                validate_sources(r)
                if digest(dest/'new_spots.npz')!=r['new_spots_sha256'] or r['intensity_reread_max_difference']!=0:raise ValueError('Invalid measured detection checkpoint')
            run_stage(label,[sys.executable,'-B','-m','sprintseq.observation_repair.recover_detection','--run-dir',str(run),'--policy-dir',str(policy),'--model-dir',str(model),'--output',str(dest),'--control-offset-x',str(offset)],dest/'receipt.json',state,root,validate_detection);save()
            peaked=root/(label+'_peaks')
            def validate_peaks(r,peaked=peaked):
                validate_sources(r)
                if digest(peaked/'new_spots.npz')!=r['new_spots_sha256']:raise ValueError('Changed peak checkpoint')
            run_stage(label+'_peaks',[sys.executable,'-B','-m','sprintseq.observation_repair.verify_signal_peaks','--run-dir',str(run),'--detection-dir',str(dest),'--output',str(peaked)],peaked/'receipt.json',state,root,validate_peaks);save()
        def validate_revision(r):
            if digest(root/'revision/readout/transcripts.parquet')!=r['transcripts_sha256'] or digest(root/'revision/segmented/cell_gene_matrix_Q20_panel202.csv')!=r['matrix_sha256']:raise ValueError('Changed extension products')
            if digest(root/'affected_peaks/receipt.json')!=r['source_detection_receipt_sha256']:raise ValueError('Changed extension source')
        run_stage('revision',[sys.executable,'-B','-m','spatial_cells.observation_extension','--run-dir',str(run),'--detection-dir',str(root/'affected_peaks'),'--output',str(root/'revision')],root/'revision/receipt.json',state,root,validate_revision)
        state['status']='MEASURED_EXTENSION_READY_FOR_INDEPENDENT_REVIEW';save();return state
    except BaseException:
        state.update(status='FAILED_PRESERVE_CHECKPOINTS',error=traceback.format_exc());save();raise
    finally:lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir','policy-dir','model-dir','output-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--control-offset-x',type=int,required=True);a=p.parse_args();run(a.run_dir,a.policy_dir,a.model_dir,a.output_root,a.control_offset_x)

if __name__=='__main__':main()
