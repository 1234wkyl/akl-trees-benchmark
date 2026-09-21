# Benchmark data layout

| Path | Contents |
|---|---|
| `sites/reference_sites.json` and `.geojson` | Fixed 24-site definitions, strata, and spatial splits |
| `sites/required_tiles.txt` | The 39 LAZ tiles required for the selected sites |
| `sites/chips/<SITE>/` | Site RGB, CHM, georeferenced CHM, metadata, and historical detection cache |
| `labels/crown_labels_XZ_2026-08-13.json` | Original author annotation export; preserve unchanged |
| `exports/reference_crowns.gpkg` | Original 492-object derived reference geometry |
| `sample/demo_labels/` | Synthetic examples only; never use as benchmark truth |

The 39 source LAZ files, frozen classical database, upstream model checkout, checkpoint, and complete VM outputs are external to Git. Input integrity is tracked in `benchmark_input_sha256.txt` and the associated run receipts. The 234-individual-tree accepted pre-freeze working layer is separate from this original export under `docs/group_refinement_2026-09-18/workflow_accepted234/`.

The original labels and site split must remain unchanged for the first blind independent-rater comparison. Do not treat selected `not_tree` polygons as an exhaustive map of background. No reference layer is formally frozen yet.
