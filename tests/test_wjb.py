"""Static tests for scripts/run_wjb.py (numpy only; Cabin/TSG torch checks are skipped without torch).

The TSG routing rule is tested with the AUTHOR'S apply/predict: the author module is executed with stub torch / sklearn
modules (its tree methods are pure numpy), so the controlled-tree acceptance runs here too."""
import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_x1 as x1
from scripts import run_x2 as x2
from scripts import run_x5 as x5
from scripts import run_wjb as w
from tests.test_x5 import statistics_imports

MEAN, SCALE = 4.65, 2.2
HAS_TORCH = importlib.util.find_spec('torch') is not None
PROTOCOL = w.read(w.PROTOCOL)


def synthetic_cache(start='2021-03-01T00', days=6, nsites=4, seed=0):
    rng = np.random.default_rng(seed)
    cycles = np.arange(np.datetime64(start, 'h'), np.datetime64(start, 'h') + np.timedelta64(24 * days, 'h'),
                       np.timedelta64(6, 'h'))
    return {x5.cycle_key(c): rng.uniform(1, 15, (6, nsites)) for c in cycles}


def synthetic_pack(hours=60, turbines=(3, 7, 9), hot=False, delay=6, seed=0):
    """x50 with GFS columns 20..48 built by run_x5.gfs_block (real validity structure) at the given delay."""
    rng = np.random.default_rng(seed)
    t = np.arange(np.datetime64('2021-03-03T00', 'h'), np.datetime64('2021-03-03T00', 'h') + np.timedelta64(hours, 'h'))
    issue = np.repeat(t, len(turbines)).astype('datetime64[ns]')
    n = len(issue)
    x50 = rng.normal(size=(n, 24, 50)).astype(np.float32)
    x50[:, -1, 20:49], _ = x5.gfs_block(issue, synthetic_cache(), delay, MEAN, SCALE, '2024-01-01', 4)
    pack = dict(x50=x50, issue=issue, turbines=np.tile(np.asarray(turbines), hours),
                truth=rng.uniform(0, 2000, (n, 4)),
                times=issue[:, None] + (np.arange(1, 5) * np.timedelta64(1, 'h'))[None, :])
    if hot:
        pack['retrofit'] = (rng.uniform(size=n) > .5).astype(np.float32)
    return pack


def author_tree_module():
    """The vendored author module with stub torch / sklearn / layers modules: class definitions and the numpy-only
    apply / predict are the author's code."""
    stubs = {}
    torch = types.ModuleType('torch')
    torch.nn = types.ModuleType('torch.nn')
    torch.utils = types.ModuleType('torch.utils')
    torch.utils.data = types.ModuleType('torch.utils.data')
    torch.utils.data.Dataset = torch.utils.data.DataLoader = object
    sk = types.ModuleType('sklearn')
    sk.linear_model = types.ModuleType('sklearn.linear_model')
    sk.linear_model.LinearRegression = object
    layers = types.ModuleType('clean_torch_custom_layers')
    for name, mod in (('torch', torch), ('torch.nn', torch.nn), ('torch.utils', torch.utils),
                      ('torch.utils.data', torch.utils.data), ('sklearn', sk), ('sklearn.linear_model', sk.linear_model),
                      ('clean_torch_custom_layers', layers)):
        stubs[name] = mod
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location('tsg_author_stubbed', w.VENDOR_TSG / 'clean_finite_adaptability_functions.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class Vendor(unittest.TestCase):
    def test_fingerprint_matches_protocol(self):
        self.assertEqual(len(w.vendor_fingerprint(PROTOCOL)), 9)

    def test_tsg_edit_is_exactly_two_unused_imports(self):
        rec = PROTOCOL['vendor']['tsg']['files']['models/vendor_tsg/clean_torch_custom_layers.py']
        text = (w.ROOT / 'models/vendor_tsg/clean_torch_custom_layers.py').read_bytes().replace(b'\r\n', b'\n')
        restored = text.replace(b'# [wjb vendoring] removed unused import: import cvxpy as cp\n', b'import cvxpy as cp\n') \
                       .replace(b'# [wjb vendoring] removed unused import: from cvxpylayers.torch import CvxpyLayer\n',
                                b'from cvxpylayers.torch import CvxpyLayer\n')
        self.assertEqual(hashlib.sha1(b'blob %d\0' % len(restored) + restored).hexdigest(), rec['upstream_blob'])
        code = [ln for ln in text.decode().splitlines() if not ln.lstrip().startswith('#')]
        self.assertFalse(any('cp.' in ln or 'CvxpyLayer' in ln or 'cvxpy' in ln for ln in code))

    def test_cabin_has_the_author_sample_softmax_switch(self):
        src = (w.VENDOR_CABIN / 'cabin.py').read_text(encoding='utf-8')
        self.assertIn('self.use_arm0 = use_sample_softmax', src)
        self.assertIn('if self.use_arm0:', src)


class Layout(unittest.TestCase):
    def test_tree_columns_match_tree_matrix(self):
        for hot in (False, True):
            pack = synthetic_pack(hot=hot)
            for info in ('L0', 'LU', 'LG', 'LUG'):
                self.assertEqual(len(w.tree_columns(info, hot)), x2.tree_matrix(pack, info, hot).shape[1])

    def test_registered_positions_both_arms(self):
        for hot in (False, True):
            self.assertEqual(w.tsg_layout('LG', hot)[1], list(range(10, 23)))
            self.assertEqual(w.tsg_layout('LUG', hot)[1], list(range(21, 34)))
            for info in ('LG', 'LUG'):
                keep, _ = w.tsg_layout(info, hot)
                self.assertEqual(len(keep), len(w.tree_columns(info, hot)) - 13)

    def test_view_maps_values_and_bits(self):
        for hot in (False, True):
            pack = synthetic_pack(hot=hot)
            units = np.unique(pack['turbines'])
            for info, start in (('LG', 10), ('LUG', 21)):
                X, valid, pos = w.tsg_view(pack, info, hot, units, start)
                np.testing.assert_array_equal(X[:, pos], pack['x50'][:, -1, 21:34])
                np.testing.assert_array_equal(valid, pack['x50'][:, -1, 34:47] > 0)
                self.assertTrue((~valid).any() and valid.any())
                np.testing.assert_array_equal(X[:, -len(units):].sum(1), 1)      # one-hot turbine block
                with self.assertRaises(ValueError):
                    w.tsg_view(pack, info, hot, units, start + 1)

    def test_scaling_rezero_and_mask(self):
        pack = synthetic_pack(delay=12)
        units = np.unique(pack['turbines'])
        X, valid, pos = w.tsg_view(pack, 'LUG', False, units, 21)
        lo, scale = w.tsg_scaler(X, valid, pos)
        Z, A = w.tsg_inputs(X, valid, pos, lo, scale)
        np.testing.assert_array_equal(A[:, pos].astype(bool), ~valid)
        self.assertFalse(np.delete(A, pos, axis=1).any())
        self.assertTrue(np.all(Z[:, pos][~valid] == 0))
        zv = Z[:, pos][valid]
        self.assertGreaterEqual(zv.min(), -1e-6)
        self.assertLessEqual(zv.max(), 1 + 1e-6)

    def test_6h_and_18h_masks_differ(self):
        a = {d: synthetic_pack(delay=d) for d in (6, 18)}
        units = np.unique(a[6]['turbines'])
        bits = {d: w.tsg_view(a[d], 'LG', False, units, 10)[1] for d in a}
        self.assertFalse(np.array_equal(bits[6], bits[18]))
        self.assertFalse(bits[18][:, 6:10].any())              # t0+1..t0+4 beyond f018 at 18 h


class AuthorRouting(unittest.TestCase):
    def test_controlled_tree_with_author_apply_predict(self):
        fa = author_tree_module()
        packs = {d: synthetic_pack(delay=d) for d in (6, 18)}
        units = np.unique(packs[6]['turbines'])
        X6, V6, pos = w.tsg_view(packs[6], 'LG', False, units, 10)
        lo, scale = w.tsg_scaler(X6, V6, pos)
        v2 = {}
        for d, p in packs.items():
            X, V, _ = w.tsg_view(p, 'LG', False, units, 10)
            v2[d] = (*w.tsg_inputs(X, V, pos, lo, scale), V)
        acc = w.tsg_acceptance(fa, v2, pos)
        r = acc['routing']
        self.assertGreater(min(r['root'], r['left'], r['right_nominal'], r['right_robust']), 0)
        self.assertGreater(acc['windows_differing'], 0)

    def test_routing_check_rejects_a_swapped_tree(self):
        fa = author_tree_module()

        class Swapped(fa.Learn_FiniteAdapt_Robust_Reg):
            def apply(self, X, missing_mask):
                return 3 - super().apply(X, missing_mask)

        fake = types.SimpleNamespace(Learn_FiniteAdapt_Robust_Reg=Swapped)
        pack = synthetic_pack(delay=12)
        units = np.unique(pack['turbines'])
        X, V, pos = w.tsg_view(pack, 'LG', False, units, 10)
        Z, A = w.tsg_inputs(X, V, pos, *w.tsg_scaler(X, V, pos))
        with self.assertRaises(ValueError):
            w.routing_check(fake, Z, A, pos)

    def test_acceptance_rejects_identical_delays(self):
        fa = author_tree_module()
        pack = synthetic_pack(delay=6)
        units = np.unique(pack['turbines'])
        X, V, pos = w.tsg_view(pack, 'LG', False, units, 10)
        Z, A = w.tsg_inputs(X, V, pos, *w.tsg_scaler(X, V, pos))
        with self.assertRaises(ValueError):
            w.tsg_acceptance(fa, {6: (Z, A, V), 18: (Z, A, V)}, pos)


class CabinHist(unittest.TestCase):
    def test_columns_roundtrip_and_shared_digests(self):
        for hot in (False, True):
            pack = synthetic_pack(hot=hot)
            units = np.unique(pack['turbines'])
            for info, cols in (('L0', 218), ('LU', 506)):
                scaler = w.fit_history_scaler(pack, info, hot, units, chunk=40)
                H, dh = w.paired_inputs(pack, info, hot, units, cols, cabin=False, scaler=scaler, chunk=50)
                (x, e), dc = w.paired_inputs(pack, info, hot, units, cols, cabin=True, scaler=scaler, chunk=70)
                self.assertEqual(H.shape[1], cols)
                self.assertEqual(dh, dc)                       # the two lines see byte-identical arrays
                rx, re_ = w.hist_restore(H, len(w.CABIN_DYN[info]))
                np.testing.assert_array_equal(rx, x)
                np.testing.assert_array_equal(re_, e)
                for actual, raw in zip((x, e), w.cabin_raw(pack, info, hot, units)):
                    lo = raw.min(axis=(0, 1))
                    span = raw.max(axis=(0, 1)) - lo
                    np.testing.assert_array_equal(actual, (raw - lo) / np.where(span == 0, 1, span))
                self.assertEqual(dc['x'], hashlib.sha256(x.tobytes()).hexdigest())
                self.assertEqual(dc['e'], hashlib.sha256(e.tobytes()).hexdigest())
                with self.assertRaises(ValueError):
                    w.paired_inputs(pack, info, hot, units, cols + 1, cabin=False, scaler=scaler)

    def test_scaler_is_shared_at_model_inputs_and_fit_on_train_only(self):
        train = synthetic_pack(hours=3, turbines=(1,))
        validation = synthetic_pack(hours=2, turbines=(1,))
        train['x50'][:, :, 49] = np.array([0, 500, 1000])[:, None]
        validation['x50'][:, :, 49] = np.array([500, 2000])[:, None]
        args = ('L0', False, np.array([1]))
        scaler = w.fit_history_scaler(train, *args, chunk=1)
        H, dh = w.paired_inputs(validation, *args, 218, cabin=False, scaler=scaler)
        (x, _), dc = w.paired_inputs(validation, *args, 218, cabin=True, scaler=scaler)
        rx, _ = w.hist_restore(H, len(w.CABIN_DYN['L0']))
        np.testing.assert_array_equal(rx[:, 0, 0], [0.5, 2.0])
        np.testing.assert_array_equal(x, rx)
        self.assertEqual(dc, dh)

    def test_constant_channels(self):
        pack = synthetic_pack(hot=False)
        units = np.unique(pack['turbines'])
        _, e = w.cabin_raw(pack, 'L0', False, units)
        self.assertTrue(np.all(e[:, :, 8] == 0))                            # no retrofit outside HoT
        np.testing.assert_allclose(e[:, 0, 9], np.searchsorted(units, pack['turbines']) / len(units), rtol=1e-6)
        self.assertTrue(np.all(e[:, :, 9] == e[:, :1, 9]))                 # constant over the window
        hot = synthetic_pack(hot=True)
        _, eh = w.cabin_raw(hot, 'LU', True, units)
        np.testing.assert_array_equal(eh[:, 5, 20], hot['retrofit'])


class KRR(unittest.TestCase):
    def test_solver_matches_closed_form(self):
        rng = np.random.default_rng(1)
        Z, Y = rng.normal(size=(53, 5)), rng.normal(size=(53, 4))
        C = Z[rng.choice(53, 9, replace=False)]
        sig, lam = 1.7, 1e-2
        got = w.krr_fit(Z, Y, C, [sig], [lam], chunk=7, jitter=1e-8)[(sig, lam)]
        K = np.exp(-w.sq_dist(Z, C) / (2 * sig ** 2))
        Kmm = np.exp(-w.sq_dist(C, C) / (2 * sig ** 2))
        want = np.linalg.solve(K.T @ K + 53 * lam * Kmm + 1e-8 * np.eye(9), K.T @ Y)
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(w.krr_predict(Z, C, sig, got, chunk=5), K @ got, rtol=1e-10)

    def test_standardizer_and_sigma(self):
        Z = np.column_stack([np.arange(10.0), np.zeros(10)])
        mean, scale = w.standardizer(Z)
        self.assertEqual(scale[1], 1.0)
        with self.assertRaises(ValueError):
            w.sigma_median(np.ones((5, 3)))

    def test_saturation_flag(self):
        rng = np.random.default_rng(2)
        Z = rng.normal(size=(40, 6))
        cfg = PROTOCOL['krr']
        rows = np.arange(5, 40)                                             # sample rows that are not centres
        out = w.kernel_summary(Z, Z[:5], [1e-3, 1e5], rows, cfg)
        self.assertTrue(out[1e-3]['saturated'] and out[1e-3]['below'] == 1.0)
        self.assertTrue(out[1e5]["saturated"] and out[1e5]["above"] == 1.0)
        self.assertFalse(w.kernel_summary(Z, Z[:5], [2.0], rows, cfg)[2.0]['saturated'])

    def test_view_replaces_turbine_by_one_hot(self):
        pack = synthetic_pack(hot=True)
        units = np.unique(pack['turbines'])
        v = w.krr_view(pack, 'LUG', True, units)
        m = x2.tree_matrix(pack, 'LUG', True)
        np.testing.assert_array_equal(v[:, :m.shape[1] - 1], m[:, :-1])
        self.assertEqual(v.shape[1], m.shape[1] - 1 + len(units))


class DevOnly(unittest.TestCase):
    def test_cache_boundary_members_never_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            folder = logs / 'G2/hot_dev_f58f2ce_r5/prepared'
            folder.mkdir(parents=True)
            # Object members cannot be deserialised with allow_pickle=False; touching either fails this test.
            sentinel = np.array(['excluded'], dtype=object)
            dev = '2021123118'
            np.savez(folder / 'points_0.npz', **{dev: [1], '2022010100': sentinel})
            np.savez(folder / 'points_1.npz', **{dev: [2], '2022010106': sentinel})
            hashes = {p.name: w.digest(p) for p in folder.glob('points_*.npz')}
            metadata = [dict(nwp_mean=0, nwp_scale=1), dict(artifacts=hashes)]
            with patch.object(w, 'read', side_effect=metadata):
                kept, _, _, _, audit = w.dev_caches(logs, 'HoT', '2022-01-01')
            self.assertEqual(set(kept), {dev})
            np.testing.assert_array_equal(kept[dev], [2])
            self.assertEqual((audit['cycles'], audit['cycles_kept']), (3, 1))

    def test_guard_refuses_and_restores(self):
        before = (x1.paths, x1.load_pack, x2.load_farm, x5.farm_caches)
        with patch.object(x1, 'load_pack', lambda path: {'ok': str(path)}):
            inner = x1.load_pack
            with w.dev_only():
                self.assertEqual(set(w.guard_controls().values()), {'refused'})
                self.assertEqual(x1.load_pack(Path('G1/prepared/Kelmarsh/points_0.npz'))['ok'],
                                 str(Path('G1/prepared/Kelmarsh/points_0.npz')))
                with self.assertRaises(RuntimeError):
                    x1.load_pack(Path('G1/prepared/Kelmarsh/test.npz'))
            self.assertIs(x1.load_pack, inner)
        self.assertEqual((x1.paths, x1.load_pack, x2.load_farm, x5.farm_caches), before)

    def test_mix_rebuild_matches_x5_counts(self):
        """X5's assignment (default_rng(2024), full train then full V1) reproduces the registered counts."""
        ref = PROTOCOL['references']
        for farm in ('Kelmarsh', 'Penmanshiel', 'HoT'):
            coh = w.read(w.checked(w.ROOT / ref['x2_cohorts']['dir'] / f'{farm}_cohorts.json',
                                   ref['x2_cohorts']['sha256'][farm]))
            rng = np.random.default_rng(w.SEED)
            delays = ref['x5']['delays']
            a = {s: x5.mix_assignment(n, delays, rng) for s, n in (('train', coh['train']), ('V1', coh['counts']['V1']))}
            counts = {s: np.bincount(np.searchsorted(delays, v), minlength=5).tolist() for s, v in a.items()}
            mix = w.read(w.checked(w.ROOT / ref['x5']['local_dir'] / 'X5' / f'{farm}_mix.json', ref['x5']['mix_sha256'][farm]))
            self.assertEqual(counts, mix)

    def test_pinned_references_exist(self):
        r5 = PROTOCOL['references']['x5']
        done = w.read(w.checked(w.ROOT / r5['local_dir'] / 'completed.json', r5['completed_sha256']))
        reg = w.read(w.checked(w.ROOT / r5['local_dir'] / 'X5' / 'registry.json', done['registry_sha256']))
        for farm in ('Kelmarsh', 'Penmanshiel', 'HoT'):
            for i in ('LG', 'LUG'):
                for d in (6, 18):
                    self.assertIn(f'D{d}/{farm}/{i}_d{d}/2024/V2.npy', reg)
                    self.assertIn(f'X5/{farm}/{i}_mix@{d}/V2.npy', reg)

    def test_splits_equal_x2(self):
        self.assertEqual(PROTOCOL['splits'], w.read(w.ROOT / 'configs/x2_protocol.json')['splits'])


class Selection(unittest.TestCase):
    def test_ties_keep_earlier(self):
        self.assertEqual(w.choose([dict(v1_mae=2.0), dict(v1_mae=1.0), dict(v1_mae=1.0)]), 1)

    def test_compare_identical_and_rise(self):
        rng = np.random.default_rng(3)
        n = 400
        issue = np.repeat(np.arange(np.datetime64('2021-07-01T00', 'h'), np.datetime64('2021-07-01T00', 'h')
                                    + np.timedelta64(n // 4, 'h')), 4).astype('datetime64[ns]')
        y = rng.uniform(0, 2000, (n, 4))
        p = {k: y + rng.normal(0, s, (n, 4)) for k, s in (('a', 100), ('b', 120), ('c', 90), ('d', 150))}
        st = dict(block_length=8, replicates=200, seed=2024)          # the bootstrap needs >= 100 replicates
        with statistics_imports():
            same = w.compare(st, p['a'], p['a'], y, issue)
            c = w.compare(st, p['a'], p['b'], y, issue)
            r = w.rise(st, p['a'], p['b'], p['c'], p['d'], y, issue)
        self.assertTrue(same['identical'] and same['gain'] == 0.0)
        self.assertAlmostEqual(c['gain'], np.abs(p['b'] - y).mean() - np.abs(p['a'] - y).mean(), places=9)
        u = lambda lg, lug: np.abs(lg - y).mean() - np.abs(lug - y).mean()  # noqa: E731
        self.assertAlmostEqual(r['gain'], u(p['c'], p['d']) - u(p['a'], p['b']), places=9)

    def test_summary_renders(self):
        cmp = dict(gain=1.0, ci95_low=0.5, ci95_high=1.5)
        met = dict(overall=dict(mae=100.0))
        side = lambda k: {k: met, 'x5_lightgbm': met, 'krr_vs_x5_lightgbm': cmp}  # noqa: E731
        krr = {f'd{d}': dict(LG=side('krr'), LUG=side('krr'), U_krr=cmp, U_x5_lightgbm=cmp) for d in (6, 18)}
        krr.update(rise_krr=cmp, rise_x5_lightgbm=cmp)
        cab = {i: dict(cabin=met, lightgbm_hist=met, x2_lightgbm=met, cabin_vs_lightgbm_hist=cmp,
                       lightgbm_hist_vs_x2_lightgbm=cmp) for i in ('L0', 'LU')}
        cab.update(U_cabin=cmp, U_lightgbm_hist=cmp, U_x2_lightgbm=cmp)
        tsg = {f'd{d}': dict(LG=dict(tsg=met, x5_mix_lightgbm=met, tsg_vs_x5_mix_lightgbm=cmp),
                             LUG=dict(tsg=met, x5_mix_lightgbm=met, tsg_vs_x5_mix_lightgbm=cmp),
                             U_tsg=cmp, U_x5_mix_lightgbm=cmp) for d in (6, 18)}
        tsg.update(rise_tsg=cmp, rise_x5_mix_lightgbm=cmp)
        res = dict(note='n', farms=dict(Kelmarsh=dict(krr=krr, cabin=cab, tsg=tsg),
                                        HoT=dict(krr=krr, cabin=cab, tsg=dict(status='TECHNICAL_LIMIT'))))
        text = w.summary_md(res)
        self.assertIn('technical limit', text)
        self.assertIn('1.00 [0.50, 1.50]', text)


@patch.object(x2, 'git_identity', lambda: ('0' * 40, ''))       # git status on a slow mount is not under test
class Gates(unittest.TestCase):
    def test_formal_needs_smoke_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(protocol_sha256=w.digest(w.PROTOCOL), base=Path(tmp), smoke=False,
                                         allow_dirty=True)
            with self.assertRaises(ValueError):
                w.start_run(args, 'krr')
            args.smoke = True
            out, protocol, x1p = w.start_run(args, 'krr')
            self.assertTrue((out / 'launch.json').exists() and out.name == 'smoke_krr')
            with self.assertRaises(FileExistsError):                        # never reuse a directory
                w.start_run(args, 'krr')

    def test_wrong_protocol_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(protocol_sha256='0' * 64, base=Path(tmp), smoke=True, allow_dirty=True)
            with self.assertRaises(ValueError):
                w.start_run(args, 'krr')


H = 3600.0
LIMIT = PROTOCOL['tsg']['time_limit_hours'] * H


def simulate_limit(durations, kinds, workers, limit=LIMIT):
    """List scheduling in submission order (as the pool dispatches); after each completion the registered rule
    (elapsed + remaining_lower_bound > limit) is evaluated.  Returns (stop time or None, makespan)."""
    import heapq
    free, start, end = [0.0] * workers, {}, {}
    heapq.heapify(free)
    for name in durations:
        t = heapq.heappop(free)
        start[name], end[name] = t, t + durations[name]
        heapq.heappush(free, end[name])
    for name in sorted(end, key=lambda n: (end[n], n)):
        t = end[name]
        finished = {n: durations[n] for n in end if end[n] <= t}
        begun = {n: start[n] for n in start if start[n] <= t and n not in finished}
        if t + w.remaining_lower_bound(t, kinds, finished, begun, workers) > limit:
            return t, max(end.values())
    return None, max(end.values())


def two_kinds(slow_first, n_each, slow, fast):
    first, second = (('NN', slow), ('LR', fast)) if slow_first else (('LR', fast), ('NN', slow))
    durations = {f'{k}{i:02d}': s * H for k, s in (first, second) for i in range(n_each)}
    return durations, {n: n[:2] for n in durations}


def fake_tree_job(job):
    """Stand-in for tsg_fit_job: marks its start and reports the registered wall time."""
    Path(job['folder']).mkdir(parents=True)
    w.mark_started(job['folder'])
    return dict(name=job['name'], wall_seconds=job['seconds'])


class _ThreadPool:
    """ProcessPoolExecutor stand-in (threads, same constructor arguments) so run_pool is tested as written."""

    def __init__(self, max_workers, mp_context=None, initializer=None, initargs=()):
        import concurrent.futures as cf
        self._pool = cf.ThreadPoolExecutor(max_workers=max_workers)
        self._processes = None

    def submit(self, fn, job):
        return self._pool.submit(fn, job)

    def shutdown(self, wait=True, cancel_futures=False):
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)


class TSGTimeLimit(unittest.TestCase):
    def test_slow_first_then_fast_is_not_stopped(self):
        """R1-B4 counterexample: 8 workers, 48 NN trees x 1.3 h dispatched first, then 48 LR trees x 0.1 h.
        Real makespan 8.4 h; the old first-round extrapolation said 15.6 h and stopped."""
        durations, kinds = two_kinds(True, 48, 1.3, 0.1)
        stop, makespan = simulate_limit(durations, kinds, 8)
        self.assertIsNone(stop)
        self.assertAlmostEqual(makespan / H, 8.4)
        self.assertGreater(1.3 * 96 / 8, PROTOCOL['tsg']['time_limit_hours'])          # what the old rule projected

    def test_fast_kind_late_and_slow_kind_late_under_budget(self):
        for slow_first in (True, False):
            durations, kinds = two_kinds(slow_first, 48, 1.5, 0.5)                    # makespan 12 h < 15 h
            stop, makespan = simulate_limit(durations, kinds, 8)
            self.assertIsNone(stop, slow_first)
            self.assertAlmostEqual(makespan / H, 12.0)

    def test_real_overrun_stops_early(self):
        durations = {f'NN{i:02d}': 1.5 * H for i in range(96)}                         # makespan 18 h
        stop, makespan = simulate_limit(durations, {n: 'NN' for n in durations}, 8)
        self.assertEqual((stop / H, makespan / H), (1.5, 18.0))                       # stops at the first completions

    def test_overrun_revealed_by_the_later_kind_stops(self):
        durations, kinds = two_kinds(False, 48, 2.5, 0.5)                             # LR 0.5 h first, NN 2.5 h: 18 h
        stop, makespan = simulate_limit(durations, kinds, 8)
        self.assertEqual(makespan / H, 18.0)
        self.assertIsNotNone(stop)
        self.assertLess(stop, makespan)
        self.assertGreaterEqual(stop / H, 3.0 + 2.5)                                  # only once a slow tree finished

    def test_bound_counts_running_time_and_unseen_kinds_as_free(self):
        kinds = dict(a='NN', b='NN', c='NN', d='LR')
        bound = w.remaining_lower_bound(10.0, kinds, dict(a=8.0), dict(b=4.0), workers=2)
        self.assertEqual(bound, 8.0)                    # b: 8 - 6 = 2, c: 8, d (unseen kind): 0 -> max(10 / 2, 8)

    def test_run_pool_wiring_stops_and_records(self):
        jobs = lambda root: [dict(name=f'j{i}', kind='NN', seconds=4 * H, folder=str(root / f'j{i}'))  # noqa: E731
                             for i in range(16)]
        args = types.SimpleNamespace(workers=2, worker_threads=1)
        with tempfile.TemporaryDirectory() as tmp, patch('concurrent.futures.ProcessPoolExecutor', _ThreadPool), \
                patch.dict('os.environ'):
            root = Path(tmp)
            with self.assertRaises(w.TechnicalLimit):                                 # 16 x 4 h / 2 = 32 h > 15 h
                w.run_pool(args, jobs(root / 'a'), fake_tree_job, root, LIMIT, w.time.time())
            rec = w.read(root / 'technical_limit.json')
            self.assertGreater(rec['projected_hours'], PROTOCOL['tsg']['time_limit_hours'])
            (root / 'technical_limit.json').unlink()
            res = w.run_pool(args, jobs(root / 'b'), fake_tree_job, root, 100 * H, w.time.time())
            self.assertEqual(len(res), 16)
            self.assertFalse((root / 'technical_limit.json').exists())
            self.assertEqual(len(w.run_pool(args, jobs(root / 'c'), fake_tree_job, root)), 16)   # smoke: no limit


@unittest.skipUnless(HAS_TORCH, 'torch not installed here (runs on the server)')
class CabinTorch(unittest.TestCase):
    def test_independence_check_exposes_singleton_only_effect(self):
        import torch

        class SingletonBias(torch.nn.Module):
            def forward(self, x, e):
                return x.new_full((len(x), 4, 1), float(len(x) == 1))

        x = np.zeros((100, 24, 1), np.float32)
        issue = np.arange(len(x)).astype('datetime64[h]')
        self.assertEqual(w.independence_gap(torch, SingletonBias(), x, x, issue, PROTOCOL['cabin']['independence']), 1.0)

    def test_independence_and_negative_control(self):
        import torch
        Cabin = w.load_cabin()
        rng = np.random.default_rng(0)
        x = rng.uniform(size=(120, 24, 1)).astype(np.float32)
        e = rng.uniform(size=(120, 24, 10)).astype(np.float32)
        issue = np.arange(120).astype('datetime64[h]').astype('datetime64[ns]')
        cfg = PROTOCOL['cabin']['independence']
        torch.manual_seed(0)
        self.assertLessEqual(w.independence_gap(torch, w.make_cabin(Cabin, 10, 64), x, e, issue, cfg), cfg['tolerance'])
        torch.manual_seed(0)
        self.assertGreater(w.independence_gap(torch, w.make_cabin(Cabin, 10, 64, True), x, e, issue, cfg),
                           cfg['tolerance'])


if __name__ == '__main__':
    unittest.main()
