"""Synthetic P1 boundaries: farm identities, sealed inputs, original labels and causal assembly."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.prepare_kelmarsh_multiyear_v31 import canonical_ticks
from utils import kelmarsh_enrichment as enrich
from utils import kelmarsh_upstream_v31 as upstream
from utils import penmanshiel_v35 as farm


def entry(unit=1, year=2016, begin=None):
    begin = begin or f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return {"archive": f"Penmanshiel_SCADA_{year}_WT01-10_3107.zip", "turbine": unit,
        "begin": begin, "end": end, "bytes": 12,
        "member": f"Turbine_Data_Penmanshiel_{unit:02d}_{begin}_-_{end}_1042.csv"}


def static():
    return pd.DataFrame({"Title": [f"Penmanshiel {u:02d}" for u in farm.TURBINES],
                         "Model": "MM82", "Rated power (kW)": 2050})


def fixture():
    dates = pd.date_range("2019-12-28T01:00Z", "2020-01-03T00:00Z", freq="h")
    frames = []
    for unit in farm.TURBINES:
        power = np.arange(len(dates), dtype=float) + unit - 40  # retain negative labels
        power[9] = np.nan
        frames.append(pd.DataFrame({"timestamp_hour_end_utc": dates, "turbine_id": unit,
            "wind_speed_mps": 8., "ambient_temp_c": 5., "nacelle_position_deg": 230.,
            "pitch_a_deg": 1., "pitch_b_deg": 1., "pitch_c_deg": 1., "rotor_speed_rpm": 12.,
            "power_kw": power, "power_last10_kw": power, "wind_last10_mps": 8., "normal_operation": True}))
    features = pd.DataFrame(1., index=dates.tz_localize(None), columns=upstream.UPSTREAM_FEATURES)
    features.iloc[30, 0] = np.nan
    return pd.concat(frames, ignore_index=True), features, {"schema_version": 2, "curve_years": list(farm.CURVE_YEARS)}


class FarmTests(unittest.TestCase):
    def test_gap_in_ids_is_preserved_and_missing_turbine_rejected(self):
        ratings, audit = farm.validate_static(static())
        self.assertEqual([int(k) for k in ratings], list(farm.TURBINES))
        self.assertNotIn("03", ratings)
        with self.assertRaises(ValueError):
            farm.validate_static(static().iloc[:-1])
        with self.assertRaises(ValueError):
            farm.validate_static(pd.concat([static(), static().iloc[:1]]))

    def test_only_wholly_empty_static_rows_removed(self):
        f = pd.concat([static(), pd.DataFrame([{k: np.nan for k in static().columns}])], ignore_index=True)
        self.assertEqual(farm.validate_static(f)[1]["entirely_empty_rows_removed"], 1)
        f.loc[len(f) - 1, "Model"] = "MM82"
        with self.assertRaises(ValueError):
            farm.validate_static(f)

    def test_protected_archive_rejected_before_open(self):
        for year in (2022, 2023, 2024):
            with patch.object(farm.zipfile, "ZipFile") as opener:
                with self.assertRaisesRegex(ValueError, "Protected"):
                    farm.read_member(entry(year=year), entry(year=year)["archive"])
                opener.assert_not_called()
        forged = entry(year=2021)
        forged["end"] = "2023-01-01"
        with patch.object(farm.zipfile, "ZipFile") as opener:
            with self.assertRaises(ValueError):
                farm.read_member(forged, forged["archive"])
            opener.assert_not_called()

    def test_station_year_guard(self):
        stations = [{"src_id": sid, "years": {str(y): {"files": [f"s{sid}_qcv-1_{y}.csv"]}
                    for y in farm.YEARS}} for sid in farm.STATIONS]
        self.assertEqual(len(farm.station_files(stations)), 66)
        stations[0]["years"]["2021"]["files"] = ["s150_qcv-1_2022.csv"]
        with self.assertRaises(ValueError):
            farm.station_files(stations)

    def test_start_uses_full_month_intervals_not_labels(self):
        rows = [entry(unit=u, begin="2016-06-01") for u in farm.TURBINES]
        rows[-1] = entry(unit=15, begin="2016-07-27")
        self.assertEqual(farm.first_complete_month(rows), "2016-08-01T00:00:00Z")
        with self.assertRaises(ValueError):
            farm.first_complete_month(rows[:-1])

    def test_channel_order_labels_queue_and_train_only_fill(self):
        frame, features, meta = fixture()
        start = "2019-12-28T00:00:00Z"
        series = farm.assemble_series(frame, features, meta, start)
        self.assertEqual(series.channels, upstream.CHANNELS)
        self.assertEqual(sorted(np.unique(series.turbines)), list(farm.TURBINES))
        np.testing.assert_array_equal(series.raw[:, -1], series.base.raw[:, -1])
        np.testing.assert_array_equal(series.starts, series.base.starts)
        self.assertTrue(np.any(series.raw[:, -1] < 0))
        self.assertTrue(np.isnan(series.raw[:, -1]).any())
        changed = features.copy()
        changed.loc[changed.index > pd.Timestamp("2020-01-01"), :] = 9999.
        other = farm.assemble_series(frame, changed, meta, start)
        np.testing.assert_array_equal(other.scaler_mean, series.scaler_mean)
        np.testing.assert_array_equal(other.audit["upstream"]["fill_values_training_mean"],
                                      series.audit["upstream"]["fill_values_training_mean"])
        missing = series.raw[:, series.channels.index("upstream_missing")]
        self.assertEqual(int(missing.sum()), 14)

    def test_strict_label_path_and_whole_row_normalization(self):
        ticks = pd.DataFrame({"timestamp": pd.date_range("2019-01-01", periods=12, freq="10min", tz="UTC")})
        for c in enrich.REQUIRED_TICK_COLUMNS:
            ticks[c] = 1.
        for c in enrich.OPTIONAL_TICK_COLUMNS:
            ticks[c] = np.nan
        ticks["power_kw"] = [-10., 10., 20., 30., 40., 50., 10., 20., np.nan, 40., 50., 60.]
        blank = ticks.iloc[[0]].copy()
        blank.loc[:, blank.columns != "timestamp"] = np.nan
        normalized, audit = canonical_ticks([ticks, blank])
        hourly, _ = farm.frozen.prepare_turbine("15", normalized, 2050., "2019-01-01T00:00Z", enrich.RULES)
        self.assertEqual(audit["cells_filled"], 0)
        self.assertEqual(hourly.power_kw.iloc[0], 140. / 6.)
        self.assertTrue(np.isnan(hourly.power_kw.iloc[1]))
        conflict = ticks.iloc[[0]].copy()
        conflict["power_kw"] = 123.
        with self.assertRaises(ValueError):
            canonical_ticks([ticks, conflict])

    def test_feature_lag_does_not_use_future_observations(self):
        index = pd.date_range("2019-01-01", periods=48, freq="h")
        stations = [{"src_id": s, "distance_km": 65 + i, "bearing_deg": i * 30.}
                    for i, s in enumerate(farm.STATIONS)]
        obs = {sid: pd.DataFrame({"wind_speed": np.arange(48) * .1 + 5,
               "wind_direction": 180., "msl_pressure": np.arange(48) * .1 + 1000}, index=index) for sid in farm.STATIONS}
        labels = index[6:12]
        expected = upstream.hourly_upstream(obs, stations, labels, lag=1)
        for frame in obs.values():
            frame.loc[index[11]:, :] = 999.
        pd.testing.assert_frame_equal(expected, upstream.hourly_upstream(obs, stations, labels, lag=1))


if __name__ == "__main__":
    unittest.main()
