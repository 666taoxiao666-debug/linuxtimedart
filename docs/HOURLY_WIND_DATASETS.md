# Hourly wind datasets: independent acquisition and classification

Acquisition date: 2026-10-13. No model training, label-based cleaning, old-data
replacement, or modification of existing SDWPF experiments is part of this task.

## Directory map

```text
datasets/
  sdwpf_245days_v1.csv                  # original, remains in place
  sdwpf_fixed.csv                      # original, remains in place
  wind_public/
    catalog.json                      # actual completion, hashes, rows and quality
    original_sdwpf_reference.json      # read-only original path/size/mtime index
    original_sdwpf_unchanged_check.json # original size/mtime unchanged during acquisition
    native_hourly_farm/gefcom2014/
      raw/historical/                 # Task 15 historical power + NWP, 10 zones
      raw/future_nwp/                 # future competition NWP, NOT observed wind
      raw/evaluation_labels/          # Task 15 solution, NOT merged into history
      hourly/gefcom2014_historical_hourly.csv
      metadata/                       # pinned mirror manifest, tree and quality
    provider_hourly_single_turbine/dswe/
      raw/Wind_Time_Series_Dataset.zip # provider's 10-minute and hourly CSVs
      hourly/dswe_hourly.csv
      metadata/                       # source record and quality
    resampled_hourly_scada/kelmarsh_2016/
      raw/                            # official 2016 ZIP, static metadata, signal map
      hourly/kelmarsh_2016_hourly.csv
      metadata/                       # source record and quality
```

Raw downloads and prepared datasets are excluded from Git by the existing
`datasets/` ignore rule. Nothing has been moved out of the old SDWPF paths.
An in-progress `.part` file is not a completed dataset. `catalog.json` contains
only successfully verified and prepared datasets; missing entries are incomplete.

## Sources, role and access limits

### GEFCom2014 wind: native hourly, farm-level

The [competition paper, Section 6.1](https://robjhyndman.com/papers/gefcom2014.pdf)
describes 10 Australian wind farms, capacity-normalized power, and ECMWF wind
component forecasts at 10 m and 100 m. These are NWP forecasts, not turbine
SCADA wind observations. Exact farm capacity and timezone are not established
here. The historical Task 15 files contain 168,000 hourly records in total,
2012-01-01 01:00 to 2013-12-01 00:00, with 112 missing power values preserved.

The original download endpoint timed out from the acquisition server. Files were
therefore acquired from the public [greenlytics mirror](https://github.com/greenlytics/gefcom2014-wind),
at fixed commit `b7e96c24a367700082039123ade8f98ea01b78b6`. Each of 21 files is
checked against the pinned Git tree/blob SHA-1 and given an independent SHA-256.
This verifies mirror snapshot identity, not an unavailable official ZIP checksum.
The [IEEE dataset catalog](https://ieee-pes-data-sharing.org/datasets/detail/0e87366e-2e91-4024-b658-43f6b22faa69)
lists license N/A. Do not assume permission to redistribute the data under an
open-source license. Download history, future NWP and evaluation answers remain
separate. No future answer was used in data fitting or any model.

`TIMESTAMP` is preserved verbatim; a separate parsed timestamp handles `24:00`
by rolling to the next day. Wind magnitude is a deterministic transformation of
the supplied U100/V100 forecasts. No time gaps are interpolated or targets clipped.

### DSWE Wind Time Series: provider hourly aggregation, one turbine

[Zenodo record 5516539](https://zenodo.org/records/5516539), Yu Ding, CC BY 4.0.
The provider explicitly says the hourly series is an average of 10-minute data,
not native hourly measurements. The ZIP contains 39,195 10-minute records and
7,265 published hourly records for a single inland turbine. Source power scale,
capacity and timezone are not established; do not relabel the power as kW.

The provider averages were checked against `(t-1h,t]` source observations:
maximum numerical discrepancy below `1e-8`. The prepared CSV expands only the
hourly time grid: 8,759 rows, 1,494 absent source hours remain NaN, and 5,714 hours
have six source observations. `power_source` keeps all published values including
partial-hour means and negative values; `power_complete_hour_source` additionally
marks incomplete hours NaN. Counts and flags make that distinction explicit.
No imputation or outlier deletion is performed. This dataset can test temporal
forecasting, but cannot establish cross-turbine transfer.

### Kelmarsh: real multi-turbine SCADA, explicitly derived hourly

[Zenodo record 16807551](https://zenodo.org/records/16807551), Cubico Sustainable
Investments Ltd, CC BY 4.0. The source provides six Senvion MM92 turbines, 10-minute
SCADA and event/status files, across 2016–2024. **Only the 2016 archive is selected
in this acquisition**, not all years. Static metadata gives 2050 kW per turbine
and commercial operations date 15/04/2016, so this year includes commissioning,
not a full year of mature operations. Static metadata and the signal map are
retained beside the original ZIP. This source is the closest match to the
existing observable-physics Wiki because it includes power, wind, three blade
pitch positions, nacelle direction, temperature, rotor speed and control fields.
Source columns are retained without assuming SDWPF angle semantics apply here.

Hourly bins are `[t,t+1h)`, labelled/available at `t+1h` in UTC. Each channel needs
six finite on-grid observations to yield its complete-hour average; otherwise its
primary hourly value is NaN. A separate partial-power mean and per-channel counts
are retained. Directions use circular means, not arithmetic means. Negative power,
shutdown/curtailment observations and source missingness are not filtered, filled
or clipped. Actual timestamp range and availability are reported from the files,
not inferred from the export comment's stated annual interval. Event files stay
in the original ZIP; they are not treated as pre-known future interventions.

Computed 2016 quality: 52,416 hourly rows across six turbines, 47,438 complete
power hours and 4,978 incomplete power hours. Blade pitch, nacelle direction and
temperature have about 34,050 complete hours each. **The exported turbine power
setpoint column has zero valid hourly values**; its presence in the header does
not mean it is usable. No setpoint-aware mechanism is supported by this subset.
There are 4,421 negative-power hours and 89 hours slightly above nominal power;
they remain visible and are not automatically declared bad sensor labels.

## Reproducible acquisition and viewing

From the existing repository, with a Python environment containing requests,
numpy and pandas:

```bash
python -m unittest discover -s tests -p test_hourly_wind_acquisition.py -v
python scripts/acquire_hourly_wind_datasets.py
python -m json.tool datasets/wind_public/catalog.json
find datasets/wind_public -maxdepth 5 -type f -name '*.csv' -print
```

Verified existing raw files are reused. Rerunning the acquisition does not launch
training. For a completed download, regenerate deterministic prepared outputs
without networking using `--prepare-only` (optionally `--datasets dswe gefcom kelmarsh`).
Do not run multiple downloaders against the same target concurrently.

## Before any experiment

- New hourly h12 means **12 hours**, whereas the existing 10-minute SDWPF h12
  means **2 hours**. Define comparable physical horizons in a new protocol.
- Kelmarsh has six turbines; the old minimum-eight-turbine transfer criterion
  would reject every event. A different protocol is required, not silent reuse.
- GEFCom is farm-level p.u. with forecast weather, Kelmarsh is turbine-level kW,
  and DSWE has unspecified source power scale. Do not reuse the old 1500-kW scale.
- Normalization, feature/target availability, missing-data masks, temporal splits,
  strong baselines and Wiki gains must be declared separately for each dataset.
- Hour averaging smooths extremes. A better hourly R² does not establish that
  SDWPF was defective or that the Wiki mechanism improved.
- Missing values and operational abnormalities occur in these sources too.
  New data enables a controlled cross-dataset check; it does not predetermine
  whether the previous failure was due to data, implementation or the method.
