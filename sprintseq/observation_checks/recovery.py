"""Automatic post-audit recovery through qualified measurement and default readback."""
import argparse,json,hashlib,time,os,traceback
from pathlib import Path
from .preflight import ObservationHold
from .audit import run as audit
from sprintseq.observation_repair.__main__ import digest


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def resolve_audit(before,after,revision,discovery):
    """Resolve technical recovery, retaining every statistical triage flag and unknown."""
    if revision.get('status')!='STANDARD_OBSERVATION_CANDIDATE_VERIFIED':raise ValueError('Unverified molecular candidate')
    if before['features_audited']!=after['features_audited'] or revision['density_planes_verified']!=after['features_audited']:raise ValueError('Panel-wide verification incomplete')
    return {'status':'TECHNICAL_RECOVERY_COMPLETE_WITH_RESIDUAL_UNCERTAINTY','features_audited':after['features_audited'],'regions':after['regions'],'flagged_gene_regions_before':before.get('flagged_gene_regions',[]),'flagged_gene_regions_after':after.get('flagged_gene_regions',[]),'known_residual':after.get('known_residual'),'spatial_nodata':after.get('spatial_nodata',[]),'repair_scope_exceeded':after.get('repair_scope_exceeded',[]),'discovery_components':discovery['components'],'scientific_accuracy_accepted':False,'interpretation':'Statistical density flags remain triage, not automatic gene edits. Acceptance is source/measurement/control/locality/count consistency, not density smoothness.'}


def run(run_dir,ref_file,preflight,before,*,activate=True,device='cpu',allspot_cycle=11):
    from sprintseq.observation_repair.pipeline import run as molecular
    from sprintseq.observation_repair.discovery import run as discover
    from sprintseq.observation_repair.discovery_policy import DiscoveryPolicy
    from spatial_cells.observation_candidate import build
    from spatial_cells.observation_publish import publish
    run=Path(run_dir).resolve();ref=Path(ref_file).resolve();image=run/'stitched/mosaic.ome.tif'
    if before.get('registration',{}).get('quality_exceptions',0):
        evidence=run/'observation_checks/registration_resolution.json'
        if not evidence.exists():raise ObservationHold('HOLD_REGISTRATION_REQUIRES_SOURCE_BOUND_RESOLUTION: '+str(evidence))
        evidence=json.loads(evidence.read_text())
        if evidence.get('status')!='REGISTRATION_EXCEPTIONS_TECHNICALLY_RESOLVED' or evidence.get('sources')!=preflight['contract']['registration']['sources']:raise ObservationHold('HOLD_STALE_REGISTRATION_RESOLUTION')
    from sprintseq.observation_repair import discovery,discovery_policy,pipeline
    from spatial_cells import observation_candidate,observation_publish
    modules=[Path(__file__),*sorted(Path(pipeline.__file__).parent.glob('*.py')),*sorted(Path(__file__).parent.glob('*.py')),Path(observation_candidate.__file__),Path(observation_publish.__file__)]
    contract={'run_id':run.name.removesuffix('_processed'),'transcripts_sha256':digest(run/'readout/transcripts.parquet'),'intensities_sha256':digest(run/'readout/transcript_intensities.parquet'),'codebook_sha256':digest(ref),'preflight_sha256':digest(preflight['receipt']),'image_identity':[image.stat().st_size,image.stat().st_mtime_ns],'source_code':{str(p):digest(p) for p in modules},'allspot_cycle':allspot_cycle}
    key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()[:20];root=run/'observation_repair'/('automatic_'+key);root.mkdir(parents=True,exist_ok=True);state={'status':'RUNNING','run_id':contract['run_id'],'contract':contract,'started_at':time.time(),'steps':{}};statepath=root/'automatic_state.json';lock=root/'automatic.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    def stage(name):state['stage']=name;write(statepath,state)
    try:
        stage('molecular');molecular(run,root/'molecular',fit_device=device,qualify_first_pause=True,molecular_only=True);state['steps']['molecular']='QUALIFIED'
        stage('supplemental_discovery');dr=discover(run,root/'molecular',root/'discovery',policy=DiscoveryPolicy(allspot_cycle=allspot_cycle),device=device);state['steps']['discovery']=dr['status']
        stage('candidate');revision=root/'revision'
        if (revision/'receipt.json').exists():
            result=json.loads((revision/'receipt.json').read_text())
            if digest(revision/'readout/transcripts.parquet')!=result['transcripts_sha256'] or result['source_transcripts_sha256']!=contract['transcripts_sha256']:raise ValueError('Candidate checkpoint changed')
        else:result=build(run,root/'molecular',root/'discovery',revision)
        state['steps']['candidate']=result['status'];stage('panel_audit');after=audit(run,ref,root/'post_recovery_audit',preflight['receipt'],readout_dir=revision/'readout');resolution=resolve_audit(before,after,result,dr);write(root/'resolution.json',resolution)
        for p,h in contract['source_code'].items():
            if digest(p)!=h:raise ValueError('Recovery implementation changed while running')
        if activate:
            stage('default_activation');accepted=publish(run,revision,root/'activation');state['activation_receipt']=str(root/'activation/receipt.json');state['steps']['activation']=accepted['status']
        state.update(status='COMPLETE_WITH_REPORTED_COVERAGE_LIMITS' if activate else 'CANDIDATE_VERIFIED_NOT_ACTIVATED',resolution=str(root/'resolution.json'),revision=str(revision),scientific_accuracy_accepted=False,finished_at=time.time());write(statepath,state);return state
    except ObservationHold as error:
        state.update(status='HOLD_SOURCE_OR_UNQUALIFIED_GEOMETRY',reason=str(error));write(statepath,state);raise
    except BaseException:
        state.update(status='FAILED_PRESERVE_CHECKPOINTS',error=traceback.format_exc());write(statepath,state);raise
    finally:lock.unlink()
