"""Approved X1 queue, single instance, no automatic retries or installations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_x1 import digest, read, save, verify_chronos


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--a1', type=Path, required=True)
    args = p.parse_args()
    lock = (args.logs/'X1/gpu_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'run.pid').write_text(str(os.getpid())+'\n')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    phase = 'preflight'
    try:
        import lightgbm
        if Path(sys.prefix).name != 'timedart' or lightgbm.__version__ != '4.5.0':
            raise ValueError('Queue must run in timedart with LightGBM 4.5.0')
        model = verify_chronos(args.receipt, args.manifest)
        source = {str(f.relative_to(ROOT)): digest(f) for folder in ('scripts', 'models', 'utils')
                  for f in (ROOT/folder).rglob('*.py')}
        protocol_sha = digest(ROOT/'configs/x1_protocol.json')
        save(args.output/'launch.json', dict(pid=os.getpid(), model=model, source_sha256=source,
             protocol_sha256=protocol_sha, queue=['A3', 'B2', 'B1', 'C1'], retry=False,
             gpu_memory_fraction=.2, cpu_threads=2))
        for phase in ('A3', 'B2', 'B1', 'C1'):
            if any(digest(ROOT/name) != sha for name, sha in source.items()):
                raise ValueError('Queue source changed after launch')
            if digest(ROOT/'configs/x1_protocol.json') != protocol_sha:
                raise ValueError('Queue protocol changed after launch')
            save(args.output/'status.json', dict(status='RUNNING', phase=phase))
            command = [sys.executable, '-u', str(ROOT/'scripts/run_x1.py'), '--arm', phase,
                '--logs', str(args.logs), '--output', str(args.output/phase), '--protocol-sha256', protocol_sha,
                '--chronos-receipt', str(args.receipt), '--chronos-manifest', str(args.manifest),
                '--a1-results', str(args.a1), '--a3-results', str(args.output/'A3')]
            with (args.output/(phase+'.log')).open('x') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = read(args.output/phase/'completed.json')
            if result['status'] != 'COMPLETED' or (args.output/phase/'failed.json').exists():
                raise ValueError('Arm did not complete: '+phase)
            save(args.output/(phase+'_receipt.json'), dict(completed_sha256=digest(args.output/phase/'completed.json')))
        save(args.output/'completed.json', dict(status='COMPLETED', phases=['A3','B2','B1','C1']))
        save(args.output/'status.json', dict(status='COMPLETED'))
    except Exception as exc:
        save(args.output/'failed.json', dict(phase=phase, type=type(exc).__name__, message=str(exc), retry=False))
        raise


if __name__ == '__main__':
    main()
