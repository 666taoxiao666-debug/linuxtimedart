"""G1 causal preparation; no downloads, label changes, or sealed-year decoding."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("g1_displacement", ROOT / "models/NWPDisplacement.py")
nd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nd)
fit_affine_calibration, apply_calibration = nd.fit_affine_calibration, nd.apply_calibration

LEADS = np.array([3, 6, 9, 12, 15, 18])
CUTOFF = np.datetime64("2024-01-01", "h")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def available_cycle(origin):
    h = np.asarray(origin, dtype="datetime64[h]").astype(np.int64)
    return ((h - 6) // 6 * 6).astype("datetime64[h]")


def period_mask(times, bounds, *, hour_end=False):
    t = np.asarray(times, dtype="datetime64[ns]")
    if hour_end:
        t = t - np.timedelta64(1, "h")
    return (t >= np.datetime64(bounds[0])) & (t < np.datetime64(bounds[1]))


def interpolate_leads(values, requested):
    """Interpolate only admissible leads; masked queries never index outside [3,18]."""
    q = np.asarray(requested, dtype=float)
    v = np.asarray(values, dtype=float)
    if v.shape[0] != 6:
        raise ValueError("Exactly six registered leads required")
    valid = np.isfinite(q) & (q >= 3) & (q <= 18)
    safe = np.where(valid, q, 3)
    left = np.clip(np.searchsorted(LEADS, safe, side="right") - 1, 0, 5)
    right = np.minimum(left + 1, 5)
    # Exact endpoints must not require a different forecast step.
    right = np.where(safe == LEADS[left], left, right)
    a, b = v[left], v[right]
    weight = (safe - LEADS[left]) / np.maximum(1, LEADS[right] - LEADS[left])
    while weight.ndim < a.ndim:
        weight = weight[..., None]
    mask = np.isfinite(a) & np.isfinite(b)
    mask &= valid.reshape(valid.shape + (1,) * (a.ndim - valid.ndim))
    out = np.where(mask, a + weight * (b - a), 0)
    audit = {"min_read_lead": float(LEADS[left[valid]].min()) if valid.any() else None,
             "max_read_lead": float(LEADS[right[valid]].max()) if valid.any() else None,
             "left_valid": bool(np.isfinite(a[valid]).all()),
             "right_valid": bool(np.isfinite(b[valid]).all())}
    return out, mask.astype(np.float32), audit


def bilinear(values, latitude, longitude, points):
    """Interpolate components first, then caller derives speed; no extrapolation."""
    lat, lon = np.asarray(latitude), (np.asarray(longitude) + 180) % 360 - 180
    yi, xi = np.argsort(lat), np.argsort(lon)
    lat, lon = lat[yi], lon[xi]
    v = np.asarray(values)[..., yi, :][..., xi]
    p = np.asarray(points, dtype=float)
    if (not np.isfinite(p).all() or np.any(p[:, 0] < lat[0]) or np.any(p[:, 0] > lat[-1])
            or np.any(p[:, 1] < lon[0]) or np.any(p[:, 1] > lon[-1])):
        raise ValueError("Point outside registered grid")
    y = np.clip(np.searchsorted(lat, p[:, 0]) - 1, 0, len(lat) - 2)
    x = np.clip(np.searchsorted(lon, p[:, 1]) - 1, 0, len(lon) - 2)
    wy = (p[:, 0] - lat[y]) / (lat[y + 1] - lat[y])
    wx = (p[:, 1] - lon[x]) / (lon[x + 1] - lon[x])
    return (v[..., y, x] * (1-wy) * (1-wx) + v[..., y+1, x] * wy * (1-wx)
            + v[..., y, x+1] * (1-wy) * wx + v[..., y+1, x+1] * wy * wx)


class PointForecasts:
    """Read only accepted cycles, with byte identity checked before value access."""
    def __init__(self, data_root, points, audit, acceptance):
        self.root, self.points = Path(data_root), np.asarray(points)
        if acceptance.get("source_audit_sha256") != digest(audit):
            raise ValueError("Acceptance does not bind this audit")
        if not acceptance.get("status", "").startswith("PASS"):
            raise ValueError("GFS acceptance has not passed")
        records = json.loads(Path(audit).read_text(encoding="utf-8"))["records"]
        self.records = {r["cycle"]: r for r in records}
        self.years = {int(r["cycle"][:4]) for r in records}
        self.accepted = set(acceptance["complete_cycles"])
        self.cache, self.identities = {}, {}
        if not all(c in self.records and self.records[c]["complete"] for c in self.accepted):
            raise ValueError("Acceptance names a non-complete cycle")

    def bridge(self, audit, acceptance, out, last_year):
        """Decode in the existing GFS environment; no environment mutation."""
        if importlib.util.find_spec("xarray") is not None:
            return
        points = Path(out).with_suffix(".points.json")
        save_json(points, self.points.tolist())
        command = ["/home/XT/miniconda3/envs/gfs_g0/bin/python", str(Path(__file__).resolve()),
            "--data-root", str(self.root), "--b2-audit", str(audit), "--b2-acceptance", str(acceptance),
            "--extract-points", str(points), "--last-year", str(last_year), "--output", str(out)]
        subprocess.run(command, check=True)
        with np.load(out, allow_pickle=False) as arrays:
            for key in arrays.files:
                self.cache[key] = arrays[key]
        self.identities[str(out)] = digest(out)
        for key in self.cache:
            path = self.root/"datasets/external/gfs_g0/dev"/key[:4]/f"gfs_{key}_cropped.nc"
            if digest(path) != self.records[key]["sha256"]:
                raise ValueError("GFS input changed during point-cache bridge")
            self.identities[str(path)] = self.records[key]["sha256"]

    def get(self, cycle):
        c = np.datetime64(cycle, "h")
        key = str(c).replace("-", "").replace("T", "")
        if c + np.timedelta64(18, "h") >= CUTOFF:
            raise ValueError("Sealed forecast valid time refused before open")
        if key not in self.accepted:
            return np.full((6, len(self.points)), np.nan)
        if key not in self.cache:
            import xarray as xr
            path = self.root / "datasets/external/gfs_g0/dev" / key[:4] / f"gfs_{key}_cropped.nc"
            record = self.records[key]
            if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("GFS file changed since acceptance: " + key)
            with xr.open_dataset(path, cache=False, decode_timedelta=True) as ds:
                valid = ds.valid_time.values
                steps = ds.step.values / np.timedelta64(1, "h")
                if (np.isnat(valid).any() or np.any(valid >= CUTOFF) or not np.array_equal(steps, LEADS)
                        or not np.array_equal(valid, c + LEADS.astype("timedelta64[h]"))):
                    raise ValueError("GFS coordinate identity changed")
                uv = [bilinear(ds[n].transpose("step", "latitude", "longitude").values,
                               ds.latitude.values, ds.longitude.values, self.points) for n in ("u10", "v10")]
                speed = np.hypot(*uv)
                if not np.isfinite(speed).all():
                    raise ValueError("Accepted GFS cycle has non-finite wind")
            self.cache[key] = speed
            self.identities[str(path)] = record["sha256"]
        return self.cache[key]


def prepare_hourly(hours, forecasts, observations, farm_wind, directions, fit_bounds):
    """All winds in physical m/s; calibration uses unique fit hours, never windows."""
    hours = np.asarray(hours, dtype="datetime64[h]")
    cycles = available_cycle(hours)
    age = ((hours - cycles) / np.timedelta64(1, "h")).astype(int)
    obs = np.column_stack([np.asarray(farm_wind), np.asarray(observations)])
    if len(obs) != len(hours) or directions.shape != observations.shape:
        raise ValueError("Observation alignment mismatch")
    hist = np.full_like(obs, np.nan, dtype=float)
    current = np.zeros((len(hours), 13, obs.shape[1]), dtype=np.float32)
    masks = np.zeros_like(current)
    audit = []
    for i, (cycle, a) in enumerate(zip(cycles, age)):
        if cycle + np.timedelta64(18, "h") >= CUTOFF:
            audit.append(dict(origin=str(hours[i]), cycle=str(cycle), age=int(a), cycle_ok=False,
                              outside_development=True, min_read_lead=None, max_read_lead=None,
                              left_valid=False, right_valid=False, available_forecast_fraction=0.))
            continue
        values = forecasts.get(cycle)
        h, hm, _ = interpolate_leads(values, np.array([a]))
        hist[i] = np.where(hm[0], h[0], np.nan)
        current[i], masks[i], item = interpolate_leads(values, a + np.arange(-5, 8))
        item.update(origin=str(hours[i]), cycle=str(cycle), age=int(a),
                    cycle_ok=bool(np.isfinite(values).all()),
                    available_forecast_fraction=float(masks[i].mean()))
        audit.append(item)
    fit = period_mask(hours, fit_bounds, hour_end=True)
    cal, adjusted, adjusted_mask = [], np.zeros_like(obs), np.zeros_like(obs)
    for j in range(obs.shape[1]):
        rule = fit_affine_calibration(obs[fit, j], hist[fit, j])
        adjusted[:, j], adjusted_mask[:, j] = apply_calibration(obs[:, j], np.isfinite(obs[:, j]), rule)
        cal.append(rule)
    fit_g = hist[fit, 0]
    if not np.isfinite(fit_g).any():
        raise ValueError("No fit-period NWP values")
    mean, scale = float(np.nanmean(fit_g)), max(float(np.nanstd(fit_g)), 1e-3)
    return dict(hours=hours, current=current, masks=masks, calibrated=adjusted,
                calibrated_mask=adjusted_mask, directions=np.asarray(directions),
                historical=hist, age=age, calibration=cal, nwp_mean=mean, nwp_scale=scale,
                cycle_ok=np.array([r["cycle_ok"] for r in audit], dtype=float),
                audit=audit, fit_bounds=fit_bounds)


def station_batch(hourly, indices):
    i = np.asarray(indices)
    w = i[:, None] + np.arange(-3, 0)
    if np.any(w < 0):
        raise ValueError("Station history outside preparation")
    direction = hourly["directions"][i-1]
    valid = np.isfinite(direction) & (direction >= 0) & (direction <= 360)
    rad = np.radians(np.where(valid, direction, 0))
    return dict(oS=hourly["calibrated"][w, 1:].transpose(0, 2, 1),
                oS_mask=hourly["calibrated_mask"][w, 1:].transpose(0, 2, 1),
                gS=hourly["current"][i, :8, 1:].transpose(0, 2, 1),
                gS_mask=hourly["masks"][i, :8, 1:].transpose(0, 2, 1),
                gF=hourly["current"][i, :, 0], gF_mask=hourly["masks"][i, :, 0],
                dS=np.stack([np.sin(rad), np.cos(rad)], -1), dS_mask=valid.astype(float),
                oF=hourly["calibrated"][w, 0], oF_mask=hourly["calibrated_mask"][w, 0],
                age=hourly["age"][i], cycle_ok=hourly["cycle_ok"][i])


def validate_lead_audit(rows):
    reads = [(r['min_read_lead'], r['max_read_lead']) for r in rows if r['min_read_lead'] is not None]
    if not reads or any(lo < 3 or hi > 18 or lo > hi for lo, hi in reads):
        raise ValueError('Actual GFS read outside [3, 18] h or no audited reads')
    if any(r['age'] < 6 or r['age'] > 11 for r in rows):
        raise ValueError('Forecast publication lag outside [6, 11] h')
    return dict(status='PASS', min_read_lead=min(x[0] for x in reads),
                max_read_lead=max(x[1] for x in reads), origins=len(rows))


def nwp_input(x21, hourly, hourly_indices):
    idx = np.asarray(hourly_indices)
    issue = idx[:, -1]
    m, s = hourly["nwp_mean"], hourly["nwp_scale"]
    hist = hourly["historical"][idx, 0]
    h = np.where(np.isfinite(hist), (hist-m)/s, 0)[..., None]
    mask = hourly["masks"][issue, :, 0]
    forecast = np.where(mask, (hourly["current"][issue, :, 0]-m)/s, 0)
    const = np.column_stack([forecast, mask, hourly["age"][issue], 1-hourly["cycle_ok"][issue]])
    const = np.broadcast_to(const[:, None], (*x21.shape[:2], 28))
    return np.concatenate([x21[..., :-1], h, const, x21[..., -1:]], -1).astype(np.float32)


def checked(path, expected, identities):
    path = Path(path)
    value = digest(path)
    if value != expected:
        raise ValueError("Input hash mismatch: " + str(path))
    identities[str(path)] = value
    return path


def development_csv(path, last_year):
    """Filter textual rows before numeric parsing; no sealed labels enter pandas."""
    import csv
    import io
    import pandas as pd
    buffer = io.StringIO()
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["label_year"]) <= last_year:
                writer.writerow(row)
    buffer.seek(0)
    return pd.read_csv(buffer)


def load_development(data_root, farm, bounds, identities, *, metadata_only=False):
    """Reuse registered parsers and H features; restrict all value access to development."""
    import pandas as pd
    from scripts import train_penmanshiel_p2_v35 as p2
    u = p2.u
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    recipe = read(p2.PROTOCOL)
    last_year = int(bounds["test"][1][:4])-1
    if last_year >= 2024:
        raise ValueError("Sealed SCADA refused")
    data_root = Path(data_root)
    if farm == "Kelmarsh":
        ref = recipe["data"]["v31_registered_preparation"]
        csvpath = checked(data_root/ref["csv"], ref["csv_sha256"], identities)
        meta = read(checked(csvpath.parent.parent/"metadata/preparation.json", ref["preparation_json_sha256"], identities))
        spec = recipe["data"]["upstream"]
        manifest = data_root/spec["download_manifest"]
        if u.text_sha256_lf(manifest) != spec["download_manifest_sha256_lf"]:
            raise ValueError("Registered MIDAS manifest mismatch")
        identities[str(manifest)] = digest(manifest)
        hashes, stations = u.manifest_hashes(read(manifest)), spec["stations"]
        ids = [int(s["src_id"]) for s in stations]
        files = []
        for item in spec["development_roots"]:
            years = [y for y in item["years"] if y <= last_year]
            if years:
                files += u.station_files(data_root/item["root"], ids, years, hashes, True)[0]
        for entry in files:
            checked(entry["path"], entry["sha256"], identities)
        center = [52.4013, -0.9428]
    elif farm == "Penmanshiel":
        old = read(ROOT/"configs/n1ph_protocol.json")
        prepared = data_root/old["p3_run"]/"prepared"
        expected = read(ROOT/"output/penmanshiel_hourly/p3_v35/r2/prepared/prepared_complete.json")["artifacts"]
        for name in ("penmanshiel.csv", "preparation.json", "input_hashes.json"):
            checked(prepared/name, expected[name], identities)
        csvpath, meta = prepared/"penmanshiel.csv", read(prepared/"preparation.json")
        part_path = ROOT/p2.farm.D2_SUBDIR/"part_A.json"
        part = read(checked(part_path, p2.farm.D2_SHA["part_A.json"], identities))
        center = part["static"]["center"]
        stations = [s for s in part["midas"]["stations"] if int(s["src_id"]) in p2.farm.STATIONS]
        ids = [int(s["src_id"]) for s in stations]
        reg, files = read(prepared/"input_hashes.json"), []
        from scripts.screen_kelmarsh_upstream_observations import read_badc_csv, column
        catalog_reg = read(checked(ROOT/p2.farm.D2_SUBDIR/"input_hashes.json", p2.farm.D2_SHA["input_hashes.json"], identities))
        catalogs = [(p, e) for p, e in catalog_reg.items() if "dv-202607" in Path(p).name and p.endswith("station-metadata.csv")]
        if not catalogs or len({e["sha256"] for _, e in catalogs}) != 1:
            raise ValueError("Frozen station coordinates missing or conflicting")
        cp, ce = catalogs[0]
        catalog = read_badc_csv(checked(data_root/"datasets"/cp.split("/datasets/", 1)[1], ce["sha256"], identities))
        sid_col = column(catalog, "src_id")
        lat_col, lon_col = column(catalog, "station_latitude", "latitude", "lat"), column(catalog, "station_longitude", "longitude", "lon")
        for station in stations:
            rows = catalog[catalog[sid_col].astype(int) == int(station["src_id"])]
            if len(rows[[lat_col, lon_col]].drop_duplicates()) != 1:
                raise ValueError("Non-unique frozen station coordinates")
            station.update(lat=float(rows.iloc[0][lat_col]), lon=float(rows.iloc[0][lon_col]))
        for sid in ids:
            for year in range(2016, last_year+1):
                found = [(path, entry) for path, entry in reg.items() if Path(path).parent.name == str(sid)
                         and Path(path).name.endswith(f"_qcv-1_{year}.csv")]
                if len(found) != 1:
                    raise ValueError("Registered MIDAS station/year missing or duplicate")
                path, entry = found[0]
                f = checked(data_root/"datasets"/path.split("/datasets/", 1)[1], entry["sha256"], identities)
                files.append(dict(path=f, year=year))
    else:
        raise ValueError("Unknown farm")
    if metadata_only:
        if farm != 'Penmanshiel' or len(stations) != 11 or len(set(ids)) != 11:
            raise ValueError('Expected eleven unique registered Penmanshiel stations')
        for station in stations:
            if not np.isfinite([station[k] for k in ('lat', 'lon', 'distance_km', 'bearing_deg')]).all():
                raise ValueError('Incomplete station geometry')
        return dict(status='PASS_METADATA_ONLY', farm=farm, station_count=len(stations),
                    station_ids=ids, stations=stations, midas_files=len(files),
                    input_hashes=identities, gfs_read=False, training=False)
    frame = development_csv(csvpath, last_year)
    obs = u.load_station_observations(files, ids)
    hours = pd.date_range(bounds["train"][0], bounds["test"][1], freq="h")
    upstream = u.hourly_upstream(obs, stations, hours, 1)
    # The provider builds the original queue. Date-exact scalers are replaced below.
    provisional = {"train": list(range(int(bounds["train"][0][:4]), last_year-1)), "val": [last_year-1], "test": [last_year]}
    series = u.KelmarshUpstreamSeries(frame, upstream, splits=provisional, input_len=24, horizon=4, stride=3,
        extras=("power_last10", "wind_last10"), start=bounds["train"][0],
        end=bounds["test"][1]+"T01:00:00Z", preparation=meta)
    return series, frame, obs, stations, hours, center, upstream


class ForecastUnion:
    def __init__(self, stores):
        self.stores, self.years = stores, set()
        for store in stores:
            if self.years & store.years:
                raise ValueError("Overlapping acceptance years")
            self.years |= store.years

    def get(self, cycle):
        year = int(str(cycle)[:4])
        for store in self.stores:
            if year in store.years:
                return store.get(cycle)
        raise ValueError("Required GFS year is not accepted")


def fit_h_scaler(series, upstream, fit_bounds):
    import pandas as pd
    fit = period_mask(series.dates, fit_bounds, hour_end=True)
    features = upstream.reindex(pd.DatetimeIndex(series.dates)).to_numpy(float)
    fill = np.nanmean(features[fit], axis=0)
    if not np.isfinite(fill).all():
        raise ValueError("No fit data for an H feature")
    raw = np.column_stack([series.base.raw[:, :-1], np.where(np.isfinite(features), features, fill),
                           (~np.isfinite(features).all(axis=1)).astype(float), series.base.raw[:, -1]])
    mean, scale = np.nanmean(raw[fit], axis=0), np.nanstd(raw[fit], axis=0)
    scale = np.where(scale == 0, 1., scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Invalid fit-only H scaler")
    series.raw, series.scaler_mean, series.scaler_scale = raw, mean, scale
    series.values = ((raw-mean)/scale).astype(np.float32)
    return dict(mean=mean.tolist(), scale=scale.tolist(), fill=fill.tolist(), fit_bounds=fit_bounds)


def build_packs(series, hours, hourly, bounds):
    import pandas as pd
    row_hour = pd.DatetimeIndex(hours).get_indexer(pd.DatetimeIndex(series.dates))
    starts = series.starts
    idx = starts[:, None] + np.arange(24)
    targets = starts[:, None] + np.arange(24, 28)
    times = series.dates[targets]
    common = (row_hour[idx] >= 3).all(axis=1)
    issue_h = row_hour[idx[:, -1]]
    # The cycle's entire 18h container must be development before opening it.
    common &= available_cycle(np.asarray(hours)[issue_h]) + np.timedelta64(18, "h") < CUTOFF
    out = {}
    for split, limits in bounds.items():
        keep = common & period_mask(times, limits, hour_end=True).all(axis=1)
        if not keep.any():
            raise ValueError("Empty common cohort: " + split)
        h_idx = row_hour[idx[keep]]
        x = series.values[idx[keep]]
        batch = station_batch(hourly, h_idx[:, -1])
        out[split] = dict(x21=x, x50=nwp_input(x, hourly, h_idx),
            truth=series.raw[targets[keep], -1].astype(np.float32),
            anchor=series.raw[idx[keep, -1], -1].astype(np.float32),
            issue=np.asarray(hours)[h_idx[:, -1]].astype("datetime64[ns]").astype(np.int64),
            times=times[keep].astype("datetime64[ns]").astype(np.int64),
            turbines=series.turbines[starts[keep]],
            power_mean=np.array(series.scaler_mean[-1]), power_scale=np.array(series.scaler_scale[-1]),
            **{k: np.asarray(v, dtype=np.float32) for k, v in batch.items()})
        if not np.isfinite(out[split]["truth"]).all():
            raise ValueError("Invalid labels; labels must never be repaired")
    return out


def prepare_farm(data_root, farm, bounds, accepted_pairs, out):
    import pandas as pd
    identities = {}
    series, frame, obs, stations, hours, center, upstream = load_development(data_root, farm, bounds, identities)
    scaler = fit_h_scaler(series, upstream, bounds["train"])
    points = [center] + [[s["lat"], s["lon"]] for s in stations]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    stores = [PointForecasts(data_root, points, audit, json.loads(Path(acc).read_text(encoding="utf-8")))
              for audit, acc in accepted_pairs]
    for j, (store, (audit, acc)) in enumerate(zip(stores, accepted_pairs)):
        store.bridge(audit, acc, out/f"points_{j}.npz", int(bounds["test"][1][:4])-1)
    forecasts = ForecastUnion(stores)
    wind = frame.assign(_date=pd.to_datetime(frame.timestamp_hour_end_utc).dt.tz_localize(None)).groupby("_date").wind_speed_mps.mean().reindex(hours).to_numpy()
    station_wind = np.column_stack([obs[int(s["src_id"])].wind_speed.reindex(hours) for s in stations])
    directions = np.column_stack([obs[int(s["src_id"])].wind_direction.reindex(hours) for s in stations])
    station_wind = np.where(station_wind >= 0, station_wind, np.nan)
    directions = np.where(station_wind > 0, directions, np.nan)
    hourly = prepare_hourly(hours, forecasts, station_wind, wind, directions, bounds["train"])
    lead_summary = validate_lead_audit(hourly['audit'])
    packs = build_packs(series, hours, hourly, bounds)
    geo = np.array([[s["distance_km"]/100, np.sin(np.radians(s["bearing_deg"])), np.cos(np.radians(s["bearing_deg"]))] for s in stations], dtype=np.float32)
    artifacts = {}
    for split, pack in packs.items():
        path = out/(split+".npz")
        np.savez(path, geo=geo, ws_scale=np.array(hourly["nwp_scale"]), **pack)
        artifacts[path.name] = digest(path)
        if split in ("train", "val"):
            unique_issue = np.unique(pack["issue"]).astype("datetime64[ns]")
            index = pd.DatetimeIndex(hours).get_indexer(unique_issue)
            future = index[:, None]+np.arange(1, 5)
            valid = period_mask(np.asarray(hours)[future], bounds[split], hour_end=True).all(axis=1)
            index, future = index[valid], future[valid]
            np.savez(out/(split+"_diagnostics.npz"), **station_batch(hourly, index),
                     truth_wind=hourly["calibrated"][future, 0], truth_mask=hourly["calibrated_mask"][future, 0],
                     issue=np.asarray(hours)[index].astype("datetime64[ns]").astype(np.int64))
            artifacts[split+"_diagnostics.npz"] = digest(out/(split+"_diagnostics.npz"))
    for store in stores:
        identities.update(store.identities)
    for audit, acc in accepted_pairs:
        identities[str(audit)], identities[str(acc)] = digest(audit), digest(acc)
    save_json(out/"lead_audit.json", hourly["audit"])
    save_json(out/"calibration.json", dict(stations=["farm"]+[s["src_id"] for s in stations], rules=hourly["calibration"], fit_bounds=bounds["train"]))
    for name in ('lead_audit.json', 'calibration.json'):
        artifacts[name] = digest(out/name)
    save_json(out/"prepared.json", dict(farm=farm, bounds=bounds, artifacts=artifacts, inputs=identities,
        scaler=scaler, nwp_mean=hourly["nwp_mean"], nwp_scale=hourly["nwp_scale"],
        lead_audit=lead_summary, counts={s: len(p["truth"]) for s, p in packs.items()}, label_policy="unchanged"))
    return out/"prepared.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--b2-audit", type=Path, required=True)
    parser.add_argument("--b2-acceptance", type=Path, required=True)
    parser.add_argument("--b1-audit", type=Path)
    parser.add_argument("--b1-acceptance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--extract-points", type=Path)
    parser.add_argument("--last-year", type=int, default=2020)
    args = parser.parse_args()
    acceptance = json.loads(args.b2_acceptance.read_text(encoding="utf-8"))
    if args.extract_points:
        points = json.loads(args.extract_points.read_text(encoding="utf-8"))
        store = PointForecasts(args.data_root, points, args.b2_audit, acceptance)
        if args.last_year >= 2024:
            raise ValueError("Sealed extraction refused")
        for key in sorted(store.accepted):
            cycle = np.datetime64(f"{key[:4]}-{key[4:6]}-{key[6:8]}T{key[8:10]}", "h")
            if int(key[:4]) <= args.last_year and cycle+np.timedelta64(18, "h") < CUTOFF:
                store.get(cycle)
        if args.output.exists():
            raise FileExistsError(args.output)
        np.savez(args.output, **store.cache)
        return
    if args.preflight:
        store = PointForecasts(args.data_root, [[52.4013, -.9428], [55.90367642857142, -2.2919617857142858]], args.b2_audit, acceptance)
        records = []
        for cycle in ("2016-05-01T00", "2018-01-01T06", "2020-12-30T12"):
            values = store.get(cycle)
            if not np.isfinite(values).all():
                raise ValueError("Fixed B2 preflight cycle is not accepted")
            for age in range(6, 12):
                _, mask, audit = interpolate_leads(values, age+np.arange(-5, 8))
                records.append(dict(cycle=cycle, age=age, mask_fraction=float(mask.mean()), **audit))
        save_json(args.output, dict(status="PASS_B2_POINT_PREFLIGHT", records=records, inputs=store.identities,
                                   script_sha256=digest(__file__), training=False))
        return
    if not args.b1_audit or not args.b1_acceptance:
        parser.error("Formal preparation requires accepted B1; no download or implicit fallback")
    b1 = json.loads(args.b1_acceptance.read_text(encoding="utf-8"))
    if not b1.get("status", "").startswith("PASS"):
        raise ValueError("B1 is not accepted")
    # T is the current approved route. A also requests an extra descriptive B,
    # which requires an explicit fit-count reconciliation before that route can run.
    if not {2021, 2022} <= {int(c[:4]) for c in b1["complete_cycles"]}:
        raise ValueError("Path T unavailable; resolve Path A extra-B fit count before execution")
    protocol = json.loads((ROOT/"configs/g1_protocol.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = {}
    pairs = [(args.b2_audit, args.b2_acceptance), (args.b1_audit, args.b1_acceptance)]
    for farm in protocol["farms"]:
        path = prepare_farm(args.data_root, farm, protocol["path_T"][farm], pairs, args.output/farm)
        prepared[farm] = digest(path)
    save_json(args.output/"complete.json", dict(path="T", farms=prepared,
        protocol_sha256=digest(ROOT/"configs/g1_protocol.json"), b1_acceptance_sha256=digest(args.b1_acceptance)))


if __name__ == "__main__":
    main()
