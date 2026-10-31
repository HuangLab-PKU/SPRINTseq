"""Read completed acquisition plans before readout; absence is never inferred from dark pixels."""
import json,re,os
from pathlib import Path
from sprintseq.observation_repair.__main__ import digest
from .preflight import ObservationHold


def acquisition_check(run_dir):
    run=Path(run_dir);planpath=run/'observation_checks/acquisition_plan.json'
    if not planpath.exists():
        return {'status':'RAW_PLAN_UNAVAILABLE_RENDER_SUPPORT_REQUIRED','sources':{},'limits':'No raw completeness claim. Exact rendered support and optical validity remain mandatory.'}
    plan=json.loads(planpath.read_text());rid=run.name.removesuffix('_processed')
    if plan.get('run_id')!=rid or plan.get('acquisition_complete') is not True or not plan.get('completion_evidence'):
        raise ObservationHold('HOLD_ACQUISITION_NOT_COMPLETE: finalized plan and completion evidence required')
    sources={str(planpath):digest(planpath)}
    for entry in plan['completion_evidence']:
        if digest(entry['path'])!=entry['sha256']:raise ObservationHold('HOLD_ACQUISITION_COMPLETION_SOURCE_CHANGED')
        sources[entry['path']]=entry['sha256']
    raw=Path(plan['raw_run'])
    if not raw.is_dir():raise ObservationHold('HOLD_RAW_SOURCE_UNAVAILABLE: '+str(raw))
    expected=plan['expected_observations'];seen=set();index={};empty=[]
    for cycle in sorted({int(row['cycle']) for row in expected}):
        directory=raw/f'cyc_{cycle}'
        if not directory.is_dir():continue
        with os.scandir(directory) as files:
            for f in files:
                match=re.fullmatch(r'C(\d+)-T(\d+)-(.+)-Z(\d+)\.tif',f.name)
                if not match or not f.is_file():continue
                key=(int(match[1]),match[3],int(match[2]));stat=f.stat()
                if stat.st_size<=0:empty.append(f.path)
                index.setdefault(key,[]).append(int(match[4]))
    if empty:raise ObservationHold('HOLD_EMPTY_RAW_FRAME: '+repr(empty[:8]))
    missing=[];observed=0
    for row in expected:
        key=(int(row['cycle']),row['channel'],int(row['tile_id']))
        if not row['z_indices'] or len(set(row['z_indices']))!=len(row['z_indices']) or min(row['z_indices'])<0:raise ObservationHold('HOLD_INVALID_EXPECTED_Z_INDICES')
        if key in seen:raise ObservationHold('HOLD_DUPLICATE_ACQUISITION_PLAN')
        seen.add(key);zs=index.get(key,[])
        if not zs:missing.append({'cycle':key[0],'channel':key[1],'tile_id':key[2],'render_tile':row['render_tile'],'reason':'SOURCE_STACK_ABSENT_AFTER_CONFIRMED_COMPLETION'})
        elif sorted(zs)!=sorted(row['z_indices']):raise ObservationHold('HOLD_PARTIAL_OR_GAPPED_Z_STACK: '+repr(key))
        else:observed+=1
    result={'status':'COMPLETED_ACQUISITION_CHECKED','observed_stacks':observed,'missing_stacks':missing,'sources':sources,'chemistry_pause_inferred':False,'raw_run':str(raw)}
    out=run/'observation_checks/acquisition';out.mkdir(parents=True,exist_ok=True);path=out/(digest(planpath)+'.json');path.write_text(json.dumps(result,indent=2));result['receipt']=str(path);result['receipt_sha256']=digest(path);return result
