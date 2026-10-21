# Auckland urban tree benchmark

A reproducible comparison of a classical CHM pipeline and SegmentAnyTree on sparse urban airborne LiDAR, including nested density transfer and two weak-supervision adaptation experiments.

## Research snapshot

Experimental execution is complete. The analysis uses 24 sites in eight strata and an author-confirmed reference containing 458 individual crowns (147 training, 141 validation, 170 test), 142 unresolved tree groups, 41 non-tree regions and four uncertain regions. Original blind annotations and post-review correspondence are preserved separately.

| Fixed test model | Precision | Recall | F1 | Matched median IoU |
| --- | ---: | ---: | ---: | ---: |
| Classical, training-selected 3 m setting | 0.4588 | 0.2294 | 0.3059 | 0.5352 |
| SegmentAnyTree pretrained | 0.1618 | 0.3588 | 0.2230 | 0.4712 |
| SegmentAnyTree weak pilot | 0.2500 | 0.1471 | 0.1852 | 0.3879 |
| SegmentAnyTree Alto-assisted weak revision | 0.2381 | 0.1471 | 0.1818 | 0.4568 |

No model meets all predeclared acceptance criteria. The Alto-assisted revision does not demonstrate an F1 improvement over the original weak pilot. Reported uncertainty uses paired whole-site bootstrap (2,000 resamples, seed 42). The combined test table, including all five density conditions and F1 intervals, is in `results/benchmark_summary.csv`. Each experiment retains its reference, checkpoint and prediction provenance.

## Repository contents

- `scripts/`: evaluation, point preparation, site assembly, density experiments, historical reconstruction, weak adaptation and reporting.
- `scripts/benchmark/`: shared geometry, matching, weak-loss and provenance implementations.
- `scripts/hpc/`: direct standalone Nectar VM and container execution.
- `config/`: fixed research configurations and operational scoring policies.
- `data/sites/`: site definitions, stratification and required tile list.
- `data/labels/`: the final author reference and two original independent exports.
- `data/exports/`: final scoring geometry and repair provenance.
- `tests/`: synthetic and workflow checks for the delivered pipeline.
- `results/`: compact counts, model/density tables, bootstrap results and rater correspondence.
- `provenance/`: input hashes, source/checkpoint identities, experiment ledger and release manifest.
- `docs/`: annotation protocol, major decisions, methods and results.

## Start here

```bash
conda create -n akl-benchmark-prep -c conda-forge \
  python=3.10 'numpy>=1.24,<3' 'scipy>=1.10' \
  'geopandas>=1.0' 'shapely>=2.0' 'rasterio>=1.3' \
  'pyproj>=3.6' 'pyogrio>=0.9' 'laspy>=2.5' \
  lazrs-python pyyaml pillow requests pytest
conda activate akl-benchmark-prep
python -m pytest tests -q
```

Use [RUNBOOK.md](RUNBOOK.md) for the delivered analysis commands and the complete package's execution/handover guides for GPU regeneration. Build new outputs in fresh directories and retain source hashes and run IDs. Completed experiments do not need rerunning to inspect their results.

[Protocol](docs/ANNOTATION_PROTOCOL.md) · [Results](docs/RESULTS.md) · [Major decisions](docs/DECISIONS.md) · [Source identities](docs/PROVENANCE.md)
