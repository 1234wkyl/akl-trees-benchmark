# Provisional historical crown-stage reproduction

The completed 6 October attempt and its explicit input limitations are
recorded in [the results receipt](historical_crown_attempt_20261006/RESULTS.md).

## Scope

`scripts/51_attempt_historical_crowns.py` imports the supplied historical
`build_tree_crown_pilot.py` and calls its original `segment_chunk` function.
Every supplied source file is checked against `source_reference/PROVENANCE.json`.
The original producer revision is still unconfirmed. This is an attempt to
recover crown geometry, not a claim that the 30 July baseline was reproduced.

Only training sites are processed. The script rejects validation/test site
requests and never reads reference labels during segmentation. The seed
inventory is read from SQLite with `mode=ro&immutable=1` and `query_only=ON`.
The entire database hash is checked against its existing intake receipt and
rechecked after execution. No database copy or write is needed because the
historical production `main()` and `write_sqlite()` are not invoked.

## Inputs and retained method

- Source: supplied historical commit `4f002e21ad2bfa357e00279f3813c0423b4c13fa`.
- CHM: supplied `reference_sites.vrt` and its 40 source TIFFs. Every referenced
  TIFF is checked against the delivered benchmark manifest. The supplied CHM
  is DSM minus DEM; this attempt does not introduce a pit-free reconstruction.
- Grid: original manifest bbox `[1740000,5895000,1782000,5935000]`, EPSG:2193,
  1 m pixels, 1,024 m chunks and 25 m read halo. Relevant chunks retain their
  original global boundaries; they are not shifted to each 100 m site.
- Crown assignment: nearest retained inventory seed within
  `clip(0.65 * seed_height, 5, 14)` m; the original strict seed-owner chunk rule
  prevents duplicate output between chunks.
- Crown constants: minimum height 3 m, minimum area 4 m², shape aspect ratio
  at most 3.5 and bounding-box fill at least 0.30. The original 1 m exclusion
  buffer, 0.65 m closing, 0.45 m simplification and 4-connected polygonisation
  are retained.
- Masks: current OSM building and nonvegetation responses obtained with the
  original queries, on relevant cells of the original 4 km query lattice.
  These are current replacement inputs, not the original historical snapshot.
  The original LINZ building file has not been supplied.
- Greenness: omitted explicitly unless a raster is provided via `--greenness`.
  No greenness is fabricated from the small RGB chips. Whether the frozen
  producer used this optional raster remains unknown.

HTTP retry bounds and output/cache locations are adapted in mask
acquisition. An explicit `--osm-request-subdivisions 2` option divides an
uncached geographic query bbox into four non-overlapping requests when a
whole-cell request repeatedly fails. The union covers the same bbox; filters
and full-geometry output remain unchanged. Responses are merged by OSM
type/ID, conflicting duplicate geometry is rejected, and response timestamps
are retained. Existing whole-cell caches are reused.
Requests retain their endpoint, full query, timestamp, response
status and elapsed time. Partial/error Overpass responses are rejected.
Strict offline crown execution requires every expected cache file and never
silently falls back to no masks. `--mask-source none` is an explicit diagnostic
option. A separate `--allow-missing-nonvegetation` option permits an explicitly
incomplete attempt after acquisition failures: every missing response and
original query cell are recorded in `used_mask_manifest.json`, building masks
remain mandatory, and `current_mask_coverage_complete` is false. No empty
response file is fabricated. This output cannot be described as the complete
default baseline.

## Standalone VM commands

Run from the benchmark checkout in the established geospatial environment.
These are CPU operations; the installed V100 does not accelerate this code.
Use a new attempt directory and separate run IDs for every command.

```bash
cd /home/ubuntu/akl-trees-benchmark
export PATH=/home/ubuntu/miniforge3/envs/akl-benchmark-prep/bin:$PATH
AKL_ATTEMPT="outputs/nectar/$(date -u +%Y%m%dT%H%M%SZ)_historical_default_attempt"
mkdir -p "$AKL_ATTEMPT"

bash scripts/hpc/run_vm.sh historical_crown_adapter_tests -- \
  python -m unittest discover -s tests -p test_historical_crown_attempt.py -v
```

Fetch and cache masks. If this fails, preserve the run receipt and partial
cache. Retry with a fresh `--output` directory and the same `--mask-cache`;
completed cache files are reused.

For repeated whole-cell failures, add `--osm-request-subdivisions 2` to the
mask command and use another fresh output directory. This changes download
partitioning only and does not modify crown segmentation.
An explicit `--osm-endpoints URL [URL ...]` option can use a replacement
read-only global-data mirror when the historical endpoints are overloaded.
The actual endpoint list and response timestamps are recorded. Public mirror
coverage was checked against the
[OpenStreetMap instance list](https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances)
on 6 October 2026; endpoint availability is not guaranteed.
`--osm-http-method POST` changes transport only. An optional
`--osm-query-timeout 25` lowers the server execution budget; error/partial
responses remain rejected. Both are recorded and do not alter selectors.

```bash
bash scripts/hpc/run_vm.sh historical_current_osm_masks -- \
  python scripts/51_attempt_historical_crowns.py --phase masks \
  --source /home/ubuntu/Xi_Zhang_Benchmark/source_reference \
  --chm /home/ubuntu/Xi_Zhang_Benchmark/chm/reference_sites.vrt \
  --chm-manifest /home/ubuntu/Xi_Zhang_Benchmark/provenance/original_chm_manifest.json \
  --mask-cache "$AKL_ATTEMPT/current_osm_cache" \
  --output "$AKL_ATTEMPT/mask_acquisition"
```

Execute the offline crown stage after mask acquisition succeeds:

```bash
bash scripts/hpc/run_vm.sh historical_default_training_crowns -- \
  python scripts/51_attempt_historical_crowns.py --phase run \
  --source /home/ubuntu/Xi_Zhang_Benchmark/source_reference \
  --chm /home/ubuntu/Xi_Zhang_Benchmark/chm/reference_sites.vrt \
  --chm-manifest /home/ubuntu/Xi_Zhang_Benchmark/provenance/original_chm_manifest.json \
  --database /home/ubuntu/baselines/xi-capstone-20260730/akl_trees.sqlite \
  --intake outputs/nectar/20260920T142229Z_gpu_v100_4319/baseline_intake/intake_report.json \
  --mask-cache "$AKL_ATTEMPT/current_osm_cache" \
  --output "$AKL_ATTEMPT/training_crowns"
```

If acquisition remains externally blocked and an approximate diagnostic is
needed, add `--allow-missing-nonvegetation` to the crown command and use a
clearly separate output such as `training_crowns_incomplete_masks`. Inspect
`used_mask_manifest.json` before interpreting any results. Standardisation
and development scoring use that explicitly named directory. When the missing
responses become available, run strict mode into a new directory; never
overwrite the incomplete attempt.

Standardise candidate polygons and evaluate against the existing derived
9.22 reference, explicitly in development mode:

```bash
bash scripts/hpc/run_vm.sh historical_default_standardise_train -- \
  python scripts/16_standardize_predictions.py \
  --input "$AKL_ATTEMPT/training_crowns/historical_default_crowns.gpkg" \
  --layer crowns --id-column tree_id --split train \
  --output "$AKL_ATTEMPT/standardised_train"

bash scripts/hpc/run_vm.sh historical_default_evaluate_train -- \
  python scripts/12_evaluate.py --stage development \
  --predictions "$AKL_ATTEMPT/standardised_train/predictions.gpkg" \
  --reference outputs/reference_candidates/20260924_prefreeze_249/prefreeze_evaluation_reference.gpkg \
  --output "$AKL_ATTEMPT/training_evaluation"
```

## Outputs and interpretation

- `plan.json`: actual adapter/source hashes, defaults, original chunks and
  query cells. `adapter_source.py` preserves the executable adapter version.
- `mask_cache_manifest.json` and `network_attempts.json`: current OSM evidence.
- `chm_source_fingerprints.json`: verified source TIFF hashes.
- `used_mask_manifest.json`: actual masks, omitted responses if explicitly
  permitted, and the original affected query cells.
- `historical_default_crowns.gpkg`: un-clipped candidate crown geometries in
  EPSG:2193, retained when they intersect a training site.
- `tree_crowns_historical_attempt.geojson`: the same candidate geometry in
  EPSG:4326 with stable tree IDs. It is a new derived output, not the missing
  original `tree_crowns_pilot.geojson`.
- `frozen_seed_subset.json`: frozen input seeds, including chunk/halo competitors.
- `chunk_receipts.json`: seed/mask counts, CHM valid fraction and segmentation
  time. Subset CHM nodata in remote chunk context is recorded.
- `frozen_scalar_comparison.csv`: per-ID comparison of frozen area, diameter,
  mean CHM and maximum CHM for seeds inside training sites.
- `summary.json`: aggregate matches and known limitations. Matching scalar
  values cannot establish identical boundaries; original polygons are needed
  for that comparison.
- `SHA256_manifest.json`, wrapper metadata, console log, environment and
  timing: retained for every successful or failed attempt.

This adapter reconstructs the crown stage from the final retained frozen
inventory. It does not replay inferred detection, prior canonicalisation,
cleanup or the original seed ordering in exactly equidistant cases. Producer
confirmation, original masks/greenness where used and frozen crown geometry
remain necessary to certify exact reproduction. M3 remains open.
