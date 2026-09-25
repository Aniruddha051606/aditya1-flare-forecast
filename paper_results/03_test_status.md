# Test status

Command: `python scripts/run_tests.py` (each suite in its own process, as in CI), run 2026-09-25 19:11:23 UTC on Python 3.14.7; rendered 2026-09-25 19:50:31 UTC, commit 870eeb1ec6e2c891fd66219dec37a55adfdaf278+uncommitted-changes.

**All passed**: 540 checks passed, 0 failed, across 13 suites; total run time 380 s.

| Suite | Result | Passed | Failed | Warnings in log | Time |
|---|---|---:|---:|---:|---:|
| test_catalog | ok | 42 | 0 | 0 | 1 s |
| test_correctness | ok | 78 | 0 | 0 | 7 s |
| test_extract | ok | 22 | 0 | 0 | 3 s |
| test_forward | ok | 26 | 0 | 0 | 14 s |
| test_goes | ok | 34 | 0 | 0 | 8 s |
| test_hel1os | ok | 27 | 0 | 0 | 6 s |
| test_physics | ok | 19 | 0 | 0 | 0 s |
| test_pipeline | ok | 69 | 0 | 0 | 7 s |
| test_products | ok | 65 | 0 | 0 | 20 s |
| test_robustness | ok | 47 | 0 | 2 | 24 s |
| test_scale | ok | 59 | 0 | 0 | 226 s |
| test_suit (SUIT code, Paper 2: not used by Paper 1) | ok | 24 | 0 | 0 | 13 s |
| test_suit_matrix (SUIT code, Paper 2: not used by Paper 1) | ok | 28 | 0 | 0 | 49 s |
| lint: pyflakes | clean | | | | 2 s |
| lint: ruff | clean | | | | 0 s |

## Failures

None.

## Warnings

- test_robustness / test_truncated_fits_is_reported_not_silent: `WARNING: VerifyWarning: Found a SIMPLE card but its format doesn't respect the FITS Standard [astropy.io.fits.hdu.hdulist]`
- test_robustness / test_truncated_fits_is_reported_not_silent: `WARNING: VerifyWarning: Error validating header for HDU #0 (note: Astropy uses zero-based indexing).`

No suite skips tests; each group runs every check it defines.

Logs: `outputs/tests/suites/`.
