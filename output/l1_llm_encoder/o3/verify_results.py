"""Reuse the O2 read-only array audit, adding O3 tuning/selection checks."""
import contextlib
import io
import json
from pathlib import Path

reference = Path(__file__).with_name('verify_o2_reference.py')
if not reference.exists():
    reference = Path(__file__).parent.parent / 'o2' / 'verify_results.py'
base = reference.read_text()
base = base.replace('TimeDARTFirst_M6_ba7bca7', 'TimeDARTFirst_M7_29108b4')
base = base.replace('l1_ba7bca7_o2', 'l1_29108b4_o3')
namespace = {}
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    exec(compile(base, 'verify_o2_reference.py', 'exec'), namespace)
report = json.loads(buffer.getvalue())
read, run = namespace['read'], namespace['RUN']
rerank, grid = read('o3_rerank.json'), read('o3_multiplier_grid.json')
assert len(rerank) == 21 and len(grid) == 10
assert len(list((run / 'tuning').glob('*.json'))) == 31
assert len(read('o3_candidate_pool_freeze.json')['candidates']) == 21
selection = read('search_selection_freeze.json')
freeze = read('formal_selection_freeze.json')
rank = lambda row: (row['val_mae'], row['candidate_id'], row['p_lr_multiplier'])
for selected, arm in [('P', 'P'), ('L_star', 'L'), ('R_star', 'R')]:
    best = min([row for row in rerank + grid if row['arm'] == arm], key=rank)
    assert selection[selected] == best
    assert freeze['p_lr_multipliers'][selected] == best['p_lr_multiplier']
for path in (run / 'training').rglob('*.json'):
    record = json.loads(path.read_text())
    arm = path.stem.rsplit('_s', 1)[0]
    assert record['p_lr_multiplier'] == freeze['p_lr_multipliers'][arm]
report.update(total_fits=55, tuning_fits=31, tuning_selection_verified=True,
              selected_multipliers=freeze['p_lr_multipliers'])
print(json.dumps(report, indent=2))
