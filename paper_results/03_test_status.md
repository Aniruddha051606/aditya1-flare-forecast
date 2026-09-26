# Test status

Command: `python scripts/run_tests.py` (each suite in its own process, as in CI), run 2026-09-26 02:31:47 UTC on Python 3.14.7; rendered 2026-09-26 02:37:41 UTC, commit 79139e81874fe597bf889bbf8456024685ad6c06+uncommitted-changes.

**All passed**: 549 checks passed, 0 failed, across 13 suites; total run time 352 s.

| Suite | Result | Passed | Failed | Warnings in log | Time |
|---|---|---:|---:|---:|---:|
| test_catalog | ok | 42 | 0 | 0 | 1 s |
| test_correctness | ok | 78 | 0 | 0 | 4 s |
| test_extract | ok | 22 | 0 | 0 | 2 s |
| test_forward | ok | 26 | 0 | 0 | 11 s |
| test_goes | ok | 34 | 0 | 0 | 5 s |
| test_hel1os | ok | 27 | 0 | 0 | 4 s |
| test_physics | ok | 19 | 0 | 0 | 0 s |
| test_pipeline | ok | 69 | 0 | 0 | 5 s |
| test_products | ok | 74 | 0 | 0 | 18 s |
| test_robustness | ok | 47 | 0 | 2 | 18 s |
| test_scale | ok | 59 | 0 | 0 | 219 s |
| test_suit (SUIT code, Paper 2: not used by Paper 1) | ok | 24 | 0 | 0 | 12 s |
| test_suit_matrix (SUIT code, Paper 2: not used by Paper 1) | ok | 28 | 0 | 0 | 50 s |
| lint: pyflakes | clean | | | | 2 s |
| lint: ruff | clean | | | | 0 s |

## Failures

None.

## Warnings

- test_robustness / test_truncated_fits_is_reported_not_silent: `WARNING: VerifyWarning: Found a SIMPLE card but its format doesn't respect the FITS Standard [astropy.io.fits.hdu.hdulist]`
- test_robustness / test_truncated_fits_is_reported_not_silent: `WARNING: VerifyWarning: Error validating header for HDU #0 (note: Astropy uses zero-based indexing).`

No suite skips tests; each group runs every check it defines.

Logs: `outputs/tests/suites/`.
