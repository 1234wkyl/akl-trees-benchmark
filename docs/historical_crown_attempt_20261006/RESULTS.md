# Provisional historical crown-stage results — 6 October 2026

## Reproduction evidence

The unchanged historical crown kernel recovered all 126 frozen crown IDs whose
seeds lie inside the eight training sites, with no extra seed-inside IDs. For
116/126 crowns (92.06%), raster support area, equivalent diameter, mean CHM
and maximum CHM match the frozen attributes within absolute tolerance 1e-6.
Equivalent diameter is derived from area; these are correlated checks.

| Site | Frozen seed-inside crowns | Recovered IDs | All four attributes match |
|---|---:|---:|---:|
| S001 | 11 | 11 | 10 |
| S006 | 20 | 20 | 20 |
| S008 | 25 | 25 | 24 |
| S010 | 20 | 20 | 17 |
| S015 | 15 | 15 | 15 |
| S017 | 14 | 14 | 10 |
| S019 | 20 | 20 | 20 |
| S024 | 1 | 1 | 0 |
| Total | 126 | 126 | 116 |

The [ten-row differences table](CROWN_ATTRIBUTE_DIFFERENCES.csv) preserves
both versions and signed deltas. All ten area deltas are positive, by 1–62 m²;
the cause has not been isolated. The [machine summary](REPRODUCTION_SUMMARY.json)
records actual constants, per-site counts, input fingerprints and limitations.

This evidence is substantial agreement with the frozen attributes, **not
certified reproduction**. The producer revision remains unconfirmed, original
crown polygons are unavailable, and scalar equality cannot certify identical
boundaries. The attempt reuses final retained inventory seeds; it does not
replay detection, canonicalisation or cleanup. No parameter tuning was done.

## Input conditions

The supplied historical commit is
`4f002e21ad2bfa357e00279f3813c0423b4c13fa`; all supplied source hashes matched
its provenance. The original metropolitan CHM grid was retained: EPSG:2193,
1 m pixels, nine selected 1,024 m chunks and 25 m halos. All 40 delivered CHM
TIFF hashes passed. Frozen SQLite was opened immutable and read-only; its
complete before/after SHA-256 remained
`0aa08032104ff21e38f0d3819c5fce2de0fad7cbe71012a3b797cce6a88b6bff`.

Mask inputs comprise nine current OSM building responses and eight of nine
current nonvegetation responses. The missing response is
`nonveg_tiles/osm_nonveg_v2_1760000_5919000_1764000_5923000.json`.
This query cell directly overlaps S019. Acquisition failures, response times,
queries, transport and source snapshots are retained in the evidence archive.
The approximation explicitly enabled `--allow-missing-nonvegetation`; no
successful empty response was invented. Strict completeness remains the
default. Current masks are not verified historical inputs. Historical LINZ
buildings and greenness were unavailable, and greenness was omitted explicitly.

The retained defaults and reproducible commands are documented in the
[attempt guide](../CLASSICAL_REPRODUCTION_ATTEMPT.md).

## Training development diagnostics

The output has 156 un-clipped geometries intersecting the training boundaries,
and 155 positive-area site records after standardisation. Frozen seed-inside
counts above and manual reference counts below are different denominators.

Scoring used all eight training sites and the unfrozen 9.22 derived candidate
with 82 manual single-tree references. The fixed rule uses clipped crown
centroids within 2 m, greedy IoU/distance/ID matching and no minimum detection
IoU cut-off (`minimum_match_iou=0.0`). Matched IoU is reported separately.
Eligible single trees are matched first; unmatched predictions inside group
or uncertainty regions are ignored. No validation/test scoring was performed.

TP = 20; FP = 29; FN = 62; ignored predictions = 106.

| Metric | Estimate | Development 95% bootstrap interval |
|---|---:|---:|
| precision | 0.4082 | 0.2807–0.5500 |
| recall | 0.2439 | 0.1481–0.4444 |
| f1 | 0.3053 | 0.1965–0.4773 |
| median_matched_iou | 0.4876 | 0.3304–0.6605 |

Intervals use 2,000 whole-site resamples, seed 42, without a comparison model.
These are diagnostics under incomplete replacement masks and an unfrozen
reference, not held-out accuracy, acceptance or a model-winner conclusion.
Neither the matching tolerance nor acceptance thresholds were changed.

## Execution evidence

- Crown-stage run: `20261006T015346Z_historical_training_crowns_incomplete_masks_587029`,
  141 seconds, exit 0.
- Standardisation: `20261006T015634Z_historical_incomplete_standardise_train_587237`,
  1 second, exit 0.
- Development scoring: `20261006T015725Z_historical_incomplete_evaluate_train_587293`,
  1 second, exit 0.
- Final safety tests: `20261006T015325Z_historical_crown_explicit_approximation_tests_586959`,
  seven passed.
- S017 no-mask smoke: `20261006T012638Z_historical_crown_S017_nomask_smoke_586484`,
  66 seconds, exit 0; 13/14 IDs and three attribute sets matched under this
  separate diagnostic condition.

These are CPU/network timings, not GPU-hours. Standardisation and scoring
retained a nonfatal GDAL/PROJ database-path diagnostic in their logs. A read-back
confirmed EPSG:2193, 155 valid records, and the evaluator's CRS checks passed.

The local, Git-excluded evidence archive is
`outputs/nectar_receipts/20261006T012214Z_historical_default_attempt/historical_attempt_evidence.tar.gz`.
It contains 175 entries, including 15 completed computation receipts and 174
individually hashed files; the final packaging receipt is stored beside it.
Its downloaded hash and every manifest-listed file hash passed verification.

Archive SHA-256:
`26f1551ef8b40a167dff2517923bb51702ecc367804e74c1d3802f130cc7f87c`.

## Remaining reproduction gates

M3 remains open. Confirm the producing revision and settings, resolve the ten
attribute differences with verified inputs, and obtain original crown geometry
or an approved regeneration path before certifying equivalence. When missing
mask inputs are available, run into a fresh directory and retain this attempt.
Tune only on training sites after establishing the baseline configuration.
Reference freeze and final held-out inference/scoring remain separate gates.
