"""Read-only C0b acceptance: hashes + metrics from saved arrays, no inference.

Usage: python verify_saved.py RUN_DIRECTORY CODE_WORKTREE DATA_ROOT
The JSON printed to stdout is the acceptance record; no run file is modified.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def metric(radius, error, capped):
    assert np.isfinite(radius).all() and (radius >= 0).all()
    score = 2*radius+20*np.maximum(error-radius, 0)
    assert np.isfinite(score).all()
    return dict(coverage=float((error <= radius).mean()), width_kw=float((2*radius).mean()),
                winkler_kw=float(score.mean()), cap_fraction=float(capped.mean()))


def main(run, code, data):
    assert read(run/'completed.json')['status'] == 'COMPLETED'
    assert not (run/'failed.json').exists()
    launch = read(run/'launch.json')
    assert sha(code/'scripts/pilot_conformal_c0b.py') == launch['script_sha256']
    assert sha(code/'scripts/pilot_conformal_c0.py') == launch['c0_sha256']
    assert hashlib.sha256((code/'docs/STAGE_PLAN.md').read_bytes().replace(b'\r\n', b'\n')).hexdigest() == launch['plan_sha256_lf']
    freeze = read(run/'selection_freeze.json')
    started = read(run/'test_evaluation_started.json')
    assert started['selection_freeze_sha256'] == sha(run/'selection_freeze.json')
    assert freeze['utc'] < started['utc']
    assert freeze['selection_result_sha256'] == sha(run/'selection_2022/result.json')
    checked_inputs, checked_code, checked_arrays, equal_old = set(), set(), 0, []
    means_by_year = {}
    for year, folder in [(2022, 'selection_2022'), (2023, 'evaluation_2023')]:
        phase = run/folder
        audit, result = read(phase/'calibration_freeze.json'), read(phase/'result.json')
        for name, digest in audit['input_hashes'].items():
            assert sha(name) == digest, name
            checked_inputs.add(name)
        for name, digest in audit['source_code'].items():
            assert sha(code/name) == digest, name
            checked_code.add(name)
        for name, digest in result['artifacts'].items():
            assert sha(phase/name) == digest, name
            checked_arrays += 1
        assert result['seeds'] == [2024, 2025, 2026]
        seeds = []
        for seed in result['seeds']:
            report = read(phase/f'seed_{seed}.json')
            with np.load(phase/f'evaluation_s{seed}.npz') as p:
                error = np.abs(p['prediction']-p['truth'])
                assert np.isfinite(error).all()
                assert np.array_equal(p['times'], p['issue'][:, None]+np.arange(1, 5)*np.timedelta64(1, 'h'))
                first_issue = p['issue'].min()
            medians = []
            for name in ('source', 'target_month'):
                with np.load(phase/f'{name}_s{seed}.npz') as p:
                    assert p['times'].max() <= np.datetime64(f'{year-1}-12-31T23:00')
                    assert p['times'].max() <= first_issue
                    medians.append(np.median(np.abs(p['truth']-p['prediction']), axis=0))
            np.testing.assert_allclose(medians[1]/medians[0], report['kappa'], rtol=0, atol=0)
            seed_metrics = {}
            with np.load(phase/f'intervals_s{seed}.npz') as radii, np.load(phase/f'caps_s{seed}.npz') as caps:
                assert set(radii.files) == set(report['methods']) == set(caps.files)
                if year == 2023:
                    assert set(radii.files) == {'B0', 'B1', 'B2-A', 'M-'+freeze['selected'], 'B3-'+freeze['selected']}
                for method in radii.files:
                    radius, capped = radii[method], caps[method]
                    assert radius.shape == error.shape == capped.shape and capped.dtype == bool
                    computed = metric(radius, error, capped)
                    for key, value in computed.items():
                        np.testing.assert_allclose(value, report['methods'][method]['pooled'][key], rtol=1e-12, atol=1e-9)
                    for h in range(4):
                        for key, value in metric(radius[:, h], error[:, h], capped[:, h]).items():
                            np.testing.assert_allclose(value, report['methods'][method]['horizons'][h][key], rtol=1e-12, atol=1e-9)
                    seed_metrics[method] = computed
            seeds.append(seed_metrics)
            if year == 2022:
                old = data/'outputs/logs/Conformal/c0_9e98cbd_r1'
                for name in ('source', 'target_month', 'evaluation'):
                    with np.load(phase/f'{name}_s{seed}.npz') as new, np.load(old/f'{name}_s{seed}.npz') as previous:
                        for field in ('prediction', 'truth', 'times', 'issue', 'turbines'):
                            np.testing.assert_array_equal(new[field], previous[field])
                    equal_old.append(f'{name}_s{seed}')
        means = {m: {k: float(np.mean([s[m][k] for s in seeds])) for k in seeds[0][m]} for m in seeds[0]}
        for m in means:
            for k in means[m]:
                np.testing.assert_allclose(means[m][k], result['mean_90'][m][k], rtol=1e-12, atol=1e-9)
        means_by_year[year] = means
    variants = ['S', 'A', 'S+A']
    selection = means_by_year[2022]
    eligible = [v for v in variants if .88 <= selection['M-'+v]['coverage'] <= .92]
    chosen = min(eligible, key=lambda v: (selection['M-'+v]['winkler_kw'], variants.index(v))) if eligible else min(
        variants, key=lambda v: (abs(selection['M-'+v]['coverage']-.9), variants.index(v)))
    assert chosen == freeze['selected']
    means = means_by_year[2023]
    comparators = ['B0', 'B1', 'B2-A', 'B3-'+chosen]
    eligible = [m for m in comparators if means[m]['coverage'] >= .88] or comparators
    m = means['M-'+chosen]
    flags = dict(C1=.88 <= m['coverage'] <= .92,
                 C2=m['winkler_kw'] <= .95*min(means[x]['winkler_kw'] for x in eligible),
                 C3=m['winkler_kw'] <= .97*means['B3-'+chosen]['winkler_kw'])
    final = read(run/'result.json')
    assert all(final[k] == v for k, v in flags.items())
    assert final['decision'] == ('continue' if all(flags.values()) else 'stop')
    assert final['mean_90'] == means
    print(json.dumps(dict(status='PASSED', arrays_hashed=checked_arrays, input_files_hashed=len(checked_inputs),
                         source_files_hashed=len(checked_code), old_C0_packs_exactly_equal=equal_old,
                         selected=chosen, flags=flags, means=means, no_inference=True), indent=2))


if __name__ == '__main__':
    main(*(Path(p) for p in sys.argv[1:]))
