"""Compact, read-only report from an existing Wiki calibration result."""
import json
import math
import sys
from pathlib import Path


def formatted(value, precision=2):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 'n/a'
    return f'{value:.{precision}f}'


def percentage(details, key):
    value = details.get(key)
    return formatted(100 * value if isinstance(value, (int, float)) else None)


def report(directory):
    directory = Path(directory)
    metrics = json.loads((directory / 'validation_metrics.json').read_text(encoding='utf-8'))
    audit = json.loads((directory / 'gain_calibration.json').read_text(encoding='utf-8'))
    names = audit['candidate_names']
    for version, label in [('calibrated', 'new_policy'),
                           ('previous_neural_last', 'old_neural_gate')]:
        values = metrics[version]
        details = values.get('diagnostics', {})
        print(f'[DIAG] {label}: mae_kw={formatted(values.get("original_mae"), 3)} '
              f'oracle_mae_kw={formatted(details.get("utility_oracle_mae_kw"), 3)} '
              f'abstain_pct={percentage(details, "utility_abstention_fraction")}')
        for name in names:
            key = f'factor_{name}_'
            print(f'[DIAG] {label} {name}: selected={details.get(key + "count", "n/a")} '
                  f'candidate_gain_pct={formatted(details.get(key + "candidate_gain"))} '
                  f'selected_gain_pct={formatted(details.get(key + "selected_gain"))} '
                  f'harm_pct={formatted(details.get(key + "harm"))} '
                  f'available_pct={percentage(details, key + "available")}')
        for name in ('stable', 'up', 'down'):
            key = f'utility_trend_{name}_'
            print(f'[DIAG] {label} trend_{name}: samples={details.get(key + "samples", "n/a")} '
                  f'gain_pct={formatted(details.get(key + "gain_vs_base_pct"))} '
                  f'intervene_pct={percentage(details, key + "intervention_fraction")}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python scripts/report_gain_calibration.py RESULT_DIR')
    report(sys.argv[1])
