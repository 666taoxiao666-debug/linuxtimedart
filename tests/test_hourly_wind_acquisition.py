import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from scripts.acquire_hourly_wind_datasets import (
    checked_download, circular_mean, hourly_scada, parse_gef_timestamp, prepare_dswe,
)


class HourlyWindAcquisitionTests(unittest.TestCase):
    def test_hour24_rolls_to_following_day(self):
        actual = parse_gef_timestamp(["20121231 24:00", "20130101 1:00"])
        self.assertEqual(str(actual[0]), "2013-01-01 00:00:00")
        self.assertEqual(str(actual[1]), "2013-01-01 01:00:00")

    def test_bad_timestamp_rejected(self):
        for value in ["20120101 24:30", "20120101 25:00", "20120101 1:60", "bad"]:
            with self.assertRaises(ValueError):
                parse_gef_timestamp([value])

    def test_circular_direction_not_arithmetic_average(self):
        self.assertAlmostEqual(circular_mean(pd.Series([359, 1])), 0, places=7)
        self.assertTrue(np.isnan(circular_mean(pd.Series([90, 270]))))

    def test_partial_hour_and_gap_not_filled(self):
        times = list(pd.date_range("2016-01-01", periods=6, freq="10min", tz="UTC"))
        times.extend(pd.date_range("2016-01-01 02:00", periods=5, freq="10min", tz="UTC"))
        frame = pd.DataFrame({"timestamp": times, "power_kw": [-2.0] * 11,
                              "wind_speed_mps": [7.0] * 11, "bearing": [359, 1, 359, 1, 359, 1] + [1] * 5})
        hourly = hourly_scada(frame, ["bearing"])
        self.assertEqual(len(hourly), 3)
        self.assertEqual(hourly.loc[0, "power_kw"], -2.0)  # no label clipping
        self.assertEqual(hourly.loc[0, "timestamp_hour_end_utc"].hour, 1)
        self.assertEqual(hourly.n_power_kw.tolist(), [6, 0, 5])
        self.assertTrue(np.isnan(hourly.loc[1, "power_kw"]))
        self.assertTrue(np.isnan(hourly.loc[2, "power_kw"]))
        self.assertEqual(hourly.loc[2, "power_partial_mean_kw"], -2.0)
        self.assertEqual(hourly.complete_power_hour.tolist(), [True, False, False])

    def test_duplicate_or_offgrid_scada_rejected(self):
        for dates in [["2016-01-01", "2016-01-01"], ["2016-01-01 00:01"]]:
            frame = pd.DataFrame({"timestamp": pd.to_datetime(dates, utc=True), "power_kw": [1.] * len(dates)})
            with self.assertRaises(ValueError):
                hourly_scada(frame)

    def test_verified_raw_reused_bad_raw_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "example.zip"
            data = b"public data"
            path.write_bytes(data)
            with patch("scripts.acquire_hourly_wind_datasets.request") as network:
                checked_download("unused", path, len(data), hashlib.md5(data).hexdigest())
                network.assert_not_called()
                with self.assertRaises(ValueError):
                    checked_download("unused", path, len(data), "0" * 32)
            self.assertEqual(path.read_bytes(), data)

    def test_interrupted_http_body_resumes_without_duplicate_bytes(self):
        class Response:
            status_code = 206
            def __init__(self, content_range, data, fail=False):
                self.headers = {"Content-Range": content_range}
                self.data, self.fail = data, fail
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def iter_content(self, size):
                yield self.data
                if self.fail:
                    raise requests.exceptions.ChunkedEncodingError("interrupted")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "example.csv"
            data = b"abcdef"
            with patch("scripts.acquire_hourly_wind_datasets.request", side_effect=[
                Response("bytes 0-5/6", b"ab", True), Response("bytes 2-5/6", b"cdef")
            ]) as network, patch("scripts.acquire_hourly_wind_datasets.time.sleep"):
                checked_download("public", path, 6, hashlib.md5(data).hexdigest())
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(network.call_args_list[1].kwargs["headers"]["Range"], "bytes=2-5")


if __name__ == "__main__":
    unittest.main()
