"""Per-run execution resources; scientific detection parameters are unchanged."""
import json
from pathlib import Path

def resolve_workers(run_dir, explicit=None, default=4):
    """Explicit CLI value wins; otherwise use readout_resources.json, then default."""
    def positive(value):
        if type(value) is not int or value < 1:
            raise ValueError("Readout n_workers must be a positive integer")
        return value
    if explicit is not None:
        return positive(explicit)
    run_dir=Path(run_dir)
    policy=run_dir/'readout_resources.json'
    if not policy.exists():
        return positive(default)
    data=json.loads(policy.read_text(encoding='utf-8-sig'))
    if data.get('version')!=1 or data.get('run_id')!=run_dir.name.removesuffix('_processed'):
        raise ValueError("Readout resource policy does not match this run/version")
    return positive(data['n_workers'])
