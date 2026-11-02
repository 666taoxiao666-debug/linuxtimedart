"""Execution audit only: one approved K1b instance, with inherited exclusive lock."""
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/XT/nuist/pythoncode/TimeDARTFirst')
WORK = Path('/home/XT/nuist/pythoncode/TimeDARTFirst_K1b_20261101T003221Z')
AUDIT = ROOT / 'outputs/audits/arrival_v33_k1b/20261101T003221Z'
RUN = ROOT / 'outputs/logs/Kelmarsh/v33_k1b_20261101T003221Z'
COMMIT = 'afbffeeab3f2862d70ace4ad644f6523fd16951a'
PYTHON = '/home/XT/miniconda3/envs/timedart/bin/python'
ENTRY = 'scripts/train_kelmarsh_arrival_v33_k1b.py'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(name, value):
    (AUDIT / name).write_text(json.dumps(value, indent=2) + '\n')

def git(*args):
    return subprocess.check_output(['git', '-C', str(WORK), *args], text=True).strip()

if '--supervise' in sys.argv:
    held_lock_fd = int(sys.argv[-1])
    begin = time.monotonic()
    command = [PYTHON, '-u', ENTRY, '--device', 'cuda', '--out', str(RUN)]
    child = subprocess.Popen(command, cwd=WORK, stdin=subprocess.DEVNULL, pass_fds=(held_lock_fd,))
    (AUDIT / 'run.pid').write_text(str(child.pid) + '\n')
    write('process.json', {'supervisor_pid': os.getpid(), 'python_pid': child.pid, 'command': command, 'started_utc': now()})
    rc = child.wait()
    write('completion.json', {'exit_code': rc, 'seconds': time.monotonic() - begin, 'completed_utc': now(), 'commit': COMMIT})
    sys.exit(rc)

os.chdir(WORK)
sys.path.insert(0, str(WORK))
assert git('branch', '--show-current') == 'exp/K1b'
assert git('rev-parse', 'HEAD') == COMMIT
assert not git('status', '--porcelain', '--untracked-files=no'), 'tracked files differ from commit'
assert not RUN.exists(), 'never reuse a real run directory in this launch'
assert not (AUDIT / 'launch.json').exists(), 'this instance was already launched'
assert (AUDIT / 'smoke.exit').read_text().strip() == '0'
test_log = (AUDIT / 'tests.log').read_text()
assert 'Ran 17 tests' in test_log and '\nOK\n' in test_log and 'FAILED (' not in test_log
lock = open(AUDIT.parent / 'K1b.lock', 'a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
for proc in Path('/proc').glob('[0-9]*'):
    try:
        args = (proc / 'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(Path(a.decode(errors='replace')).name == Path(ENTRY).name for a in args if a):
        raise RuntimeError(f'K1b is already running: {proc.name}')
gpu_jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
assert not gpu_jobs, f'GPU is occupied: {gpu_jobs}'

import torch
from scripts import train_kelmarsh_arrival_v33_k1b as k
from models.UpstreamArrivalResidual import ArrivalResidual

k.v26run.configure_determinism()
cfg = json.loads(k.k1.CONFIG.read_text())
p = json.loads((WORK / cfg['data_source_protocol']).read_text())
checks = []
for family, indices in [('T9', p['data']['channel_sets']['last10']), ('T21', list(range(21)))]:
    models = [k.build(arm, p, cfg, indices)[0] for arm in [family, family+'+B2', family+'+A']]
    reference = models[0].base.state_dict()
    assert all(all(torch.equal(reference[n], m.base.state_dict()[n]) for n in reference) for m in models[1:])
    checks.append(f'{family}: equal base initialization across all three arms')
    for m in models[1:]:
        m.train()
        assert m.base.training and all(q.requires_grad for q in m.base.parameters())
        m.eval()
        assert not m.base.training
    joint = models[-1]
    frozen = ArrivalResidual(copy.deepcopy(joint.base), prior='arrival', kappa='learned')
    with torch.no_grad():
        joint.head[-1].weight.fill_(0.15)
    frozen.load_state_dict(joint.state_dict())
    joint.eval()
    frozen.eval()
    x = torch.randn(3, 24, len(indices))
    content = torch.randn(3, 8, 12)
    phys = torch.zeros(3, 8, 6)
    phys[..., 0] = torch.linspace(30, 70, 8)
    phys[..., 2:4] = torch.tensor([1., 5.])
    phys[..., 4:] = 1.
    valid = torch.ones(3, 8, dtype=torch.bool)
    assert torch.allclose(joint(x, content, phys, valid), frozen(x, content, phys, valid), atol=1e-7, rtol=1e-6)
    joint.zero_grad()
    joint(x, content, phys, valid).square().mean().backward()
    assert torch.isfinite(joint.kappa_theta.grad) and joint.kappa_theta.grad.abs() > 0
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in joint.base.parameters())
    assert torch.allclose(joint(x, content, phys, valid & False), joint.base(x))
    checks.append(f'{family}: frozen-branch eval parity, base/branch train mode, base and kappa gradients, empty-mask identity')

files = git('ls-files', '*.py', 'configs/*.json', 'docs/STAGE_PLAN.md').splitlines()
hashes = {f: hashlib.sha256((WORK / f).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for f in files}
expected_frozen = {
    'scripts/train_kelmarsh_arrival_v33.py': 'dd7e03247394f8a1d4cc27259f82659c33ce56219492727c234457bfc3643462',
    'models/UpstreamArrivalResidual.py': '976176a0fec396cf1c1638c054cd08553c86864a9fa47e431876086357bc3217',
    'utils/kelmarsh_station_history_v33.py': '6da5805829c27cb82597ba869d1ce045d550c07003f41fe47236e3ff0531995b',
    'configs/kelmarsh_arrival_v33_k1.json': '318038be8813f73fd6f72b53bc8bceca664ede42cc6b0e842ced42249a1384d6',
}
assert k.k1.code_digest() == expected_frozen
write('preflight.json', {'checks': checks, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                       'device': torch.cuda.get_device_name(), 'utc': now(), 'frozen_code': k.k1.code_digest()})
RUN.mkdir()
env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
with open(AUDIT / 'train.log', 'xb') as log:
    supervisor = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--supervise', str(lock.fileno())],
                                  cwd=WORK, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, pass_fds=(lock.fileno(),))
write('launch.json', {'commit': COMMIT, 'branch': git('branch', '--show-current'), 'worktree': str(WORK),
                      'run': str(RUN), 'supervisor_pid': supervisor.pid, 'utc': now(), 'source_sha256_lf': hashes,
                      'command': [PYTHON, '-u', ENTRY, '--device', 'cuda', '--out', str(RUN)],
                      'budget': 'one real instance, six arms, seed 2024', 'env': {'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'}})
print(json.dumps({'launched': True, 'supervisor_pid': supervisor.pid, 'run': str(RUN), 'preflight': checks}))
