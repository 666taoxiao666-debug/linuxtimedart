"""Multi-year Kelmarsh preprocessing: provider-format parsing, causal end-of-hour features, frozen operating-state flags,
curve-year isolation, available-capacity decisions, and the multi-year window provider (splits, cohorts, scaler)."""
import hashlib
import copy
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries
from scripts import prepare_kelmarsh_multiyear as prep
from scripts.acquire_hourly_wind_datasets import KEL_COLUMNS
from utils import kelmarsh_enrichment as enrich

RATED = 2050.
SOURCE = {v: k for k, v in KEL_COLUMNS.items()}


def curve(wind):
    """MM92-like shape: ~110 kW at 4 m/s, ~1650 kW at 10 m/s, rated near 13 m/s."""
    return RATED / (1 + np.exp(-(np.asarray(wind, dtype=float) - 8.) / 1.4))


def ticks(start, end, seed, with_capacity=True):
    times = pd.date_range(start, end, freq="10min", inclusive="left", tz="UTC")
    rng = np.random.default_rng(seed)
    t = np.arange(len(times))
    wind = np.clip(8 + 4 * np.sin(t / 37.) + rng.normal(0, .4, len(t)), 0, None)
    frame = pd.DataFrame({"timestamp": times, "wind_speed_mps": wind, "power_kw": curve(wind) + rng.normal(0, 10, len(t)),
                          "wind_direction_deg": (200 + 5 * np.sin(t / 50.)) % 360, "nacelle_position_deg": 198.,
                          "ambient_temp_c": 12., "rotor_speed_rpm": 8 + wind / 2,
                          "pitch_a_deg": 1., "pitch_b_deg": 1.2, "pitch_c_deg": .8, "yaw_bearing_deg": 198.,
                          "power_setpoint_kw": np.nan, "available_capacity_kw": RATED if with_capacity else np.nan,
                          "data_availability_source": 1.})
    return frame


def inject(frame):
    """Known operating events (all in 2016, turbine-independent)."""
    def at(stamp):
        return frame["timestamp"].dt.floor("h") == pd.Timestamp(stamp, tz="UTC")
    stop = at("2016-05-10 10:00")
    frame.loc[stop, ["wind_speed_mps", "power_kw"]] = [8.0, 0.0]             # -> label 11:00 stopped
    derate = frame["timestamp"] == pd.Timestamp("2016-05-12 14:20", tz="UTC")
    frame.loc[derate, "available_capacity_kw"] = 1000.0                      # -> label 15:00 derated
    under = at("2016-05-15 08:00")
    frame.loc[under, ["wind_speed_mps", "power_kw"]] = [10.0, 200.0]         # -> label 09:00 underperforming
    return frame


def provider_member(frame, drop=()):
    columns = [SOURCE[c] for c in ["timestamp"] + [c for c in frame.columns if c != "timestamp"] if SOURCE[c] not in drop]
    out = io.StringIO()
    out.write("# This file was exported by a synthetic test\n# Turbine: test\n")
    out.write("# " + ",".join(columns) + "\n")
    data = frame.rename(columns=SOURCE)[columns].copy()
    data[SOURCE["timestamp"]] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    data.to_csv(out, header=False, index=False, na_rep="NaN")
    return out.getvalue().encode("utf-8")


def build_source(root):
    raw = Path(root) / prep.SUBDIR / "raw"
    raw.mkdir(parents=True)
    files = []
    for year, (start, end) in {2016: ("2016-04-20", "2016-06-20"), 2017: ("2017-05-01", "2017-06-15")}.items():
        key = f"Kelmarsh_SCADA_{year}_{3000 + year - 2016}.zip"
        with zipfile.ZipFile(raw / key, "w") as archive:
            for unit in range(1, 7):
                frame = ticks(start, end, seed=unit + year, with_capacity=(year == 2016))
                if year == 2016:
                    frame = inject(frame)
                drop = () if year == 2016 else ("Available Capacity for Production (kW)",)
                archive.writestr(f"Turbine_Data_Kelmarsh_{unit}_{start}_-_{end}_{unit}.csv", provider_member(frame, drop))
            archive.writestr(f"Status_Kelmarsh_1_{start}.csv", "# status file ignored by design\n")
        files.append(key)
    static = pd.DataFrame({"Title": [f"Kelmarsh {u}" for u in range(1, 7)], "Rated power (kW)": RATED,
                           "Commercial Operations Date": "2016-04-15"})
    static.to_csv(raw / "Kelmarsh_WT_static.csv", index=False)
    files.append("Kelmarsh_WT_static.csv")
    record = {"files": [{"key": key, "size": (raw / key).stat().st_size,
                         "checksum": "md5:" + hashlib.md5((raw / key).read_bytes()).hexdigest(),
                         "links": {"self": "http://not-used.invalid"}} for key in files]}
    return record


class EnrichmentTests(unittest.TestCase):
    def test_end_of_hour_tick_and_strict_causality(self):
        frame = ticks("2016-05-01", "2016-05-03", 1).set_index("timestamp")
        extra = enrich.hourly_enrichment(frame, RATED)
        label = pd.Timestamp("2016-05-01 06:00", tz="UTC")
        self.assertAlmostEqual(extra.loc[label, "power_last10_kw"], frame.loc[pd.Timestamp("2016-05-01 05:50", tz="UTC"), "power_kw"])
        hour = frame.loc[pd.Timestamp("2016-05-01 05:00", tz="UTC"):pd.Timestamp("2016-05-01 05:50", tz="UTC"), "power_kw"]
        self.assertAlmostEqual(extra.loc[label, "power_std10_kw"], hour.std(ddof=0))
        changed = frame.copy()
        changed.loc[changed.index >= label, ["power_kw", "wind_speed_mps"]] = 9999.
        again = enrich.hourly_enrichment(changed, RATED)
        pd.testing.assert_frame_equal(extra.loc[:label], again.loc[:label])

    def test_curve_uses_only_curve_years_and_flags_underperformance(self):
        rng = np.random.default_rng(0)
        wind = rng.uniform(0, 20, 4000)
        years = np.repeat([2016, 2017], 2000)
        power = curve(wind) * np.where(years == 2017, 0.3, 1.0)
        none = np.zeros(4000, bool)
        fitted = enrich.fit_power_curve(wind, power, years, (2016,), none, none, RATED)
        self.assertEqual(fitted["hours_used"], 2000)
        self.assertAlmostEqual(enrich.expected_power(fitted, 12.)[()] / RATED, 1.0, delta=.08)
        flag = enrich.underperformance_flag([10., 10., 2.], [900., 200., 0.], fitted, RATED)
        self.assertEqual(flag.tolist(), [False, True, False])
        with self.assertRaises(ValueError):
            enrich.fit_power_curve(wind[:50], power[:50], years[:50], (2016,), none[:50], none[:50], RATED)

    def test_window_cohort_requires_all_target_hours_normal(self):
        normal = np.ones(40, bool)
        normal[30] = False
        starts = np.array([0, 3, 6])  # targets 24..27, 27..30, 30..33
        self.assertEqual(enrich.window_cohort(normal, starts, 24, 4).tolist(), [True, False, False])


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        record = build_source(cls.root)
        cls.meta = prep.prepare(cls.root, years=[2016, 2017], curve_years=(2016,), download=False, record=record)
        cls.frame = pd.read_csv(cls.root / prep.SUBDIR / "hourly" / prep.OUTPUT_NAME, parse_dates=["timestamp_hour_end_utc"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def row(self, unit, stamp):
        f = self.frame
        return f[(f.turbine_id == unit) & (f.timestamp_hour_end_utc == pd.Timestamp(stamp, tz="UTC"))].iloc[0]

    def test_start_rule_base_columns_and_end_tick(self):
        self.assertEqual(self.frame.timestamp_hour_end_utc.min(), pd.Timestamp("2016-05-01 01:00", tz="UTC"))
        self.assertEqual(sorted(self.frame.turbine_id.unique()), [1, 2, 3, 4, 5, 6])
        source = inject(ticks("2016-04-20", "2016-06-20", seed=1 + 2016)).set_index("timestamp")
        r = self.row(1, "2016-05-03 05:00")
        hour = source.loc[pd.Timestamp("2016-05-03 04:00", tz="UTC"):pd.Timestamp("2016-05-03 04:50", tz="UTC")]
        self.assertAlmostEqual(r.power_kw, hour.power_kw.mean(), places=6)
        self.assertAlmostEqual(r.power_last10_kw, hour.power_kw.iloc[-1], places=6)
        self.assertFalse(self.meta["labels_modified"] or self.meta["clipping"] or self.meta["interpolation"])

    def test_injected_operating_events_are_flagged(self):
        for unit in (1, 4):
            self.assertTrue(self.row(unit, "2016-05-10 11:00").flag_stopped)
            self.assertTrue(self.row(unit, "2016-05-12 15:00").flag_derated)
            self.assertTrue(self.row(unit, "2016-05-15 09:00").flag_underperformance)
            for stamp in ("2016-05-10 11:00", "2016-05-12 15:00", "2016-05-15 09:00"):
                self.assertFalse(self.row(unit, stamp).normal_operation)
            self.assertTrue(self.row(unit, "2016-05-20 12:00").normal_operation)
        complete = self.frame[self.frame.power_kw.notna()]
        self.assertGreater(complete.normal_operation.mean(), .99)                   # only injected events are abnormal
        self.assertFalse(self.frame[self.frame.power_kw.isna()].normal_operation.any())  # no label -> never normal

    def test_available_capacity_rule_only_where_recorded_and_curve_years_recorded(self):
        decisions = self.meta["available_capacity_decisions"]["1"]
        self.assertTrue(decisions["2016"]["derate_rule_used"] if "2016" in decisions else decisions[2016]["derate_rule_used"])
        d2017 = decisions["2017"] if "2017" in decisions else decisions[2017]
        self.assertFalse(d2017["derate_rule_used"])
        late = self.frame[self.frame.label_year == 2017]
        self.assertFalse(late.flag_derated.any())
        self.assertEqual(self.meta["curve_years"], [2016])
        self.assertTrue(all(c["curve_years"] == [2016] for c in self.meta["power_curves"].values()))
        quality = pd.read_csv(self.root / prep.SUBDIR / "metadata" / "quality_by_year.csv")
        self.assertEqual(set(quality.year), {2016, 2017})

    def test_integrity_failure_is_refused(self):
        record = build_source(Path(self.tmp.name) / "other")
        record["files"][0]["checksum"] = "md5:" + "0" * 32
        with self.assertRaises(ValueError):
            prep.prepare(Path(self.tmp.name) / "other", years=[2016, 2017], curve_years=(2016,), record=record)

    def test_multiyear_provider_splits_cohorts_scaler_and_channels(self):
        path = self.root / prep.SUBDIR / "hourly" / prep.OUTPUT_NAME
        series = KelmarshMultiYearSeries(path, splits={"train": (2016,), "test": (2017,)},
                                         extras=("power_last10", "hour_sin", "farm_power_mean"))
        self.assertEqual(series.channels[-1], "power")
        self.assertEqual(series.channels[6:9], ["power_last10", "hour_sin", "farm_power_mean"])
        audit = series.audit["windows"]
        self.assertGreater(audit["train"]["all"], audit["train"]["normal"])
        self.assertGreater(audit["test"]["normal"], 0)
        normal = series.windows("train", "normal")
        rows = normal.window_starts[:, None] + 24 + np.arange(4)
        self.assertTrue(series.normal_hour[rows].all())
        times = normal.target_times()
        self.assertTrue(((pd.DatetimeIndex(times[:, 0]) - pd.Timedelta(hours=1)).year == 2016).all())
        test_all = series.windows("test", "all")
        self.assertTrue(((pd.DatetimeIndex(test_all.target_times()[:, -1]) - pd.Timedelta(hours=1)).year == 2017).all())
        truth, persistence = test_all.truth_and_anchor()
        self.assertEqual(truth.shape[1], 4)
        np.testing.assert_array_equal(persistence[:, 0], series.raw[test_all.window_starts + 23, -1])
        np.testing.assert_array_equal(test_all.last10_anchor()[:, 0], series.power_last10[test_all.window_starts + 23])
        x, y, _, _ = test_all[0]
        self.assertEqual(x.shape, (24, len(series.channels)))
        frame = pd.read_csv(path)
        frame.loc[frame.timestamp_hour_end_utc.str.startswith("2017"), ["power_kw", "wind_speed_mps"]] *= 100
        shifted = KelmarshMultiYearSeries(frame, splits={"train": (2016,), "test": (2017,)},
                                          extras=("power_last10", "hour_sin", "farm_power_mean"), preparation=self.meta)
        np.testing.assert_allclose(shifted.scaler_mean, series.scaler_mean)
        with self.assertRaises(ValueError):
            KelmarshMultiYearSeries(path, splits={"train": (2016,), "test": (2016,)})

    def test_curve_provenance_and_chronology_are_mandatory(self):
        path = self.root / prep.SUBDIR / "hourly" / prep.OUTPUT_NAME
        bad = copy.deepcopy(self.meta)
        bad["curve_years"] = [2016, 2017]
        with self.assertRaisesRegex(ValueError, "curve fit years"):
            KelmarshMultiYearSeries(path, splits={"train": (2016,), "test": (2017,)}, preparation=bad)
        with self.assertRaisesRegex(ValueError, "precede"):
            KelmarshMultiYearSeries(path, splits={"train": (2017,), "validation": (2016,)})
        with self.assertRaisesRegex(ValueError, "metadata"):
            KelmarshMultiYearSeries(self.frame, splits={"train": (2016,)})

    def test_data_hash_and_no_overwrite(self):
        bad = copy.deepcopy(self.meta)
        bad["prepared"]["sha256"] = "0" * 64
        path = self.root / prep.SUBDIR / "hourly" / prep.OUTPUT_NAME
        with self.assertRaisesRegex(ValueError, "hash"):
            KelmarshMultiYearSeries(path, splits={"train": (2016,)}, preparation=bad)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            prep.prepare(self.root, years=[2016, 2017], curve_years=(2016,))
        self.assertEqual(original, path.read_bytes())

    def test_undeclared_future_rows_cannot_change_train_windows(self):
        frame = self.frame.copy()
        args = dict(splits={"train": (2016,)}, preparation=self.meta, extras=("farm_power_mean",))
        first = KelmarshMultiYearSeries(frame[frame.label_year == 2016], **args)
        second = KelmarshMultiYearSeries(frame, **args)
        np.testing.assert_array_equal(first.windows("train").window_starts, second.windows("train").window_starts)
        np.testing.assert_allclose(first.raw, second.raw, equal_nan=True)
        np.testing.assert_array_equal(first.scaler_mean, second.scaler_mean)
        self.assertEqual(second.windows("train").cohort, "all")
        self.assertTrue(second.audit["normal_cohort_uses_future_target_state"])


class ReviewRegressionTests(unittest.TestCase):
    def test_future_capacity_missingness_cannot_change_past_inputs(self):
        frame = ticks("2016-05-01", "2016-05-05", 12)
        early = frame.timestamp < pd.Timestamp("2016-05-02", tz="UTC")
        frame.loc[early, "available_capacity_kw"] = 1000.
        changed = frame.copy()
        changed.loc[~early, "available_capacity_kw"] = np.nan
        a, _ = prep.prepare_turbine("1", frame, RATED, prep.DEFAULT_START, enrich.RULES)
        b, _ = prep.prepare_turbine("1", changed, RATED, prep.DEFAULT_START, enrich.RULES)
        cols = ["available_capacity_frac", "n_derated_ticks", "n_capacity_ticks", "flag_derated"]
        before = a.timestamp_hour_end_utc <= pd.Timestamp("2016-05-02", tz="UTC")
        pd.testing.assert_frame_equal(a.loc[before, cols], b.loc[before, cols])
        self.assertTrue(b.loc[before, "flag_derated"].all())

    def test_unknown_wind_is_not_confirmed_normal(self):
        frame = ticks("2016-05-01", "2016-05-02", 1)
        frame.loc[0, "wind_speed_mps"] = np.nan
        hourly, _ = prep.prepare_turbine("1", frame, RATED, prep.DEFAULT_START, enrich.RULES)
        hourly["flag_underperformance"] = False
        self.assertTrue(hourly.loc[0, "flag_incomplete_state"])
        self.assertFalse(enrich.normal_operation(hourly).iloc[0])
        self.assertTrue(np.isfinite(hourly.loc[0, "power_kw"]))  # target remains intact

    def test_subsecond_offgrid_duplicate_and_naive_ticks_refused(self):
        frame = ticks("2016-05-01", "2016-05-02", 1).set_index("timestamp")
        for index in (frame.index + pd.Timedelta(microseconds=1), frame.index.tz_localize(None),
                      frame.index.where(np.arange(len(frame)) != 1, frame.index[0])):
            bad = frame.copy()
            bad.index = index
            with self.assertRaises(ValueError):
                enrich.hourly_enrichment(bad, RATED)

    def test_no_implicit_heldout_year_preparation(self):
        self.assertEqual(prep.DEFAULT_YEARS, (2016, 2017, 2018, 2019))


try:
    import sklearn  # noqa: F401
    import torch  # noqa: F401
    HAS_TRAINING_STACK = True
except ImportError:
    HAS_TRAINING_STACK = False


@unittest.skipUnless(HAS_TRAINING_STACK, "torch/sklearn unavailable on this host")
class WindowRuleEquivalenceTests(unittest.TestCase):
    def test_copied_window_rule_equals_the_registered_one(self):
        from data_provider.kelmarsh_hourly import valid_window_starts as original
        from data_provider.kelmarsh_multiyear import valid_window_starts as copied
        rng = np.random.default_rng(0)
        values = rng.normal(size=(600, 5))
        values[rng.random(values.shape) < .02] = np.nan
        dates = np.datetime64("2016-05-01T00", "h") + np.arange(600) * np.timedelta64(1, "h")
        dates[300:] += np.timedelta64(5, "h")
        turbines = np.repeat([1, 2], 300)
        for stride in (1, 3):
            a, b = original(values, dates, turbines, 24, 4, stride), copied(values, dates, turbines, 24, 4, stride)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1], b[1])


if __name__ == "__main__":
    unittest.main()
