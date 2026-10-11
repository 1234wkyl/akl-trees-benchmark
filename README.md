# Auckland urban tree benchmark

A reproducible comparison of a classical CHM pipeline and SegmentAnyTree on sparse urban airborne LiDAR, including nested density transfer and two weak-supervision adaptation experiments.

## Research snapshot: 12 October 2026

Experimental execution is complete. The analysis uses 24 sites in eight strata and an author-confirmed reference containing 458 individual crowns (147 training, 141 validation, 170 test), 142 unresolved tree groups, 41 non-tree regions and four uncertain regions. Original blind annotations and post-review correspondence are preserved separately.

| Fixed test model | Precision | Recall | F1 | Matched median IoU |
| --- | ---: | ---: | ---: | ---: |
| Classical, training-selected 3 m setting | 0.4588 | 0.2294 | 0.3059 | 0.5352 |
| SegmentAnyTree pretrained | 0.1618 | 0.3588 | 0.2230 | 0.4712 |
| SegmentAnyTree weak pilot | 0.2500 | 0.1471 | 0.1852 | 0.3879 |
| SegmentAnyTree Alto-assisted weak revision | 0.2381 | 0.1471 | 0.1818 | 0.4568 |

No model meets all predeclared acceptance criteria. The Alto-assisted revision does not demonstrate an F1 improvement over the original weak pilot. Reported uncertainty uses paired whole-site bootstrap (2,000 resamples, seed 42). Prior held-out exposure and post-review annotation are disclosed; these results are exploratory, not a newly blind confirmatory benchmark.

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
- `docs/`: annotation protocol, major decisions, methods, results and delivery requirements.

Raw LAZ, imagery, raster tiles, the frozen database, upstream model weights, full prediction files, scientific logs and the dissertation are delivered in the separate complete local package. They are intentionally excluded from Git. Correspondence, weekly updates, authoring history and superseded annotation drafts are excluded.

## Start here

```bash
conda env create -f environment.vm.yml
conda activate akl-benchmark-prep
python -m pytest tests -q
```

Use [RUNBOOK.md](RUNBOOK.md) for the delivered analysis commands and the complete package's execution/handover guides for GPU regeneration. Build new outputs in fresh directories and retain source hashes and run IDs. Completed experiments do not need rerunning to inspect their results.

[Protocol](docs/ANNOTATION_PROTOCOL.md) · [Results](docs/RESULTS.md) · [Major decisions](docs/DECISIONS.md) · [Provenance](docs/PROVENANCE.md) · [Delivery requirements](docs/DELIVERY_REQUIREMENTS.md)
