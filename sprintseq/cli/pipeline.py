"""Standard automatic source-QC, readout, recovery and synchronized output entry."""
from pathlib import Path


def build_parser(sub):
    p=sub.add_parser('pipeline',help='Source checks, readout, automatic artifact recovery and verified outputs.')
    p.add_argument('--run-id',required=True);p.add_argument('--ref-file',type=Path,required=True);p.add_argument('--base-dir',type=Path)
    p.add_argument('--audit-only',action='store_true');p.add_argument('--recover-existing',action='store_true');p.add_argument('--candidate-only',action='store_true');p.add_argument('--allspot-cycle',type=int,default=11)
    p.set_defaults(_func=dispatch)


def dispatch(args):
    from sprintseq.observation_checks.workflow import main
    argv=['--run-id',args.run_id,'--ref-file',str(args.ref_file),'--allspot-cycle',str(args.allspot_cycle)]
    if args.base_dir is not None:argv+=['--base-dir',str(args.base_dir)]
    for flag in ['audit_only','recover_existing','candidate_only']:
        if getattr(args,flag):argv.append('--'+flag.replace('_','-'))
    return main(argv)
