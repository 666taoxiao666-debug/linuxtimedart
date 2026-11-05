# N1-PH2 implementation validation

- Server: timedart Python environment; isolated TimeDARTFirst_N1PH2_e682971 checkout.
- `python -m unittest tests.test_train_n1ph2 tests.test_train_n1ph -v`: 16 tests passed, 2.227 seconds.
- Includes synthetic PH-mix fitting, 25-channel source fitting and same-seed warm-start smoke; these are not formal fits.
- Initial test incorrectly treated a January 1 timestamp as a validation label although its hour-ending label belongs to the previous year. Test corrected to use provider label_year. Actual split unchanged.
- Source-only preflight: see source_preflight.json. No target-farm labels or formal model training; three V0 checkpoints and their fitting/validation windows and normalization verified.
- 25-channel model has 9414 parameters; 14-channel model has 5872; PH network has 1572.
- New protocol inherits unchanged N1-PH comparisons, years, thresholds and statistics. Primary results remain posthoc development evidence.
