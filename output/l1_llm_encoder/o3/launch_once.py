"""Dispatch the approved O3 once, detached from SSH, after exact preflight checks."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/XT/nuist/pythoncode/TimeDARTFirst_M7_29108b4')
DATA = Path('/home/XT/nuist/pythoncode/TimeDARTFirst')
RUN = DATA / 'outputs/logs/LLM_L1/l1_29108b4_o3'
PYTHON = '/home/XT/miniconda3/envs/timedart/bin/python'
EXPECTED = {
    'scripts/llm_encoder_search_l1.py': 'f00006a58c86967acaff061ad2f71adada31d878531cddd2566f8e6dabf4474f',
    'models/ProgramEncoder.py': '7078eaac7976a6a9d6333721b6f8421dd835b5405847195882c0ad16c87b3aff',
    'configs/l1_llm_encoder_protocol.json': '495a5a8dc22e074efe6656f7a330064f664909f0c8aa06860ca0670985488e83',
}
for name, digest in EXPECTED.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
check = json.loads((RUN / 'preflight.json').read_text())
assert check['passed'] and check['candidate_count'] == 21 and check['llm_calls'] == 0
assert not any((RUN / name).exists() for name in ('dispatch.json', 'run.pid', 'formal_instance.lock'))
for proc in Path('/proc').glob('[0-9]*'):
    try:
        args = (proc / 'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if len(args) > 1 and b'python' in args[0] and any(a.endswith(b'llm_encoder_search_l1.py') for a in args[1:]):
        raise RuntimeError(f'An L1 Python process already exists: {proc.name}')
with (RUN / 'dispatch.lock').open('x') as f:
    f.write(dt.datetime.now(dt.timezone.utc).isoformat())
command = [PYTHON, '-u', 'scripts/llm_encoder_search_l1.py', '--data-root', str(DATA),
           '--out', str(RUN), '--device', 'cuda']
env = {**os.environ, 'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'}
with (RUN / 'run.log').open('xb') as log:
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = {'pid': process.pid, 'command': command, 'cwd': str(ROOT), 'code': '29108b4',
          'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'source_sha256': EXPECTED}
(RUN / 'run.pid').write_text(str(process.pid) + '\n')
with (RUN / 'dispatch.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
