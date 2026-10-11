# Completed research results

The fixed comparisons and paired confidence intervals are delivered as machine-readable CSV/JSON in `results/`. The complete local package also contains prediction GeoPackages, point outputs, error galleries and full run evidence.

| Test model | TP | FP | FN | Ignored | F1 | 95% paired-site F1 interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Classical, training-selected 3 m | 39 | 46 | 131 | 77 | 0.3059 | [0.2203, 0.4103] |
| SAT pretrained | 61 | 316 | 109 | 247 | 0.2230 | [0.1863, 0.2733] |
| SAT weak pilot | 25 | 75 | 145 | 31 | 0.1852 | [0.0942, 0.2500] |
| SAT Alto-assisted weak revision | 25 | 80 | 145 | 35 | 0.1818 | [0.0968, 0.2400] |

The revised-minus-original weak F1 difference is -0.0034, with 95% paired interval [-0.0189, 0.0165]. No improvement is demonstrated. None of the models meets all acceptance criteria. The pretrained-minus-classical F1 interval also includes zero.

The pretrained density sweep is complete at 100%, 75%, 50%, 25% and 12.5% retained returns, with test F1 0.2230, 0.2264, 0.2331, 0.1662 and 0.0543. Return thinning is not a new survey and is not pulse-density sampling. The fixed classical CHM does not constitute a classical density sweep.

The revised weak queue audited all 19 tiles and assembled all 16 validation/test sites. Its wrappers used 64,009 seconds (17.7803 hours), approximately 2.7459 GPU allocation hours/km² over the recorded source-tile coverage proxy. This measures allocated elapsed time, not active CUDA time or provider billing.

Historical reconstruction uses the immutable 30 July database. Frozen-seed generation recovers 126 crown candidates and matches four stored attributes for 125; independent detection recovers 108 retained inferred detections and matches four attributes for 115 crown candidates. These checks do not prove identical historical polygon vertices or unique producer parameters.

Reference revisions and earlier held-out exposure are disclosed. All labels, checkpoints and fixed predictions remain separate by experiment. Completed execution does not imply that performance thresholds were met, that point targets are independently verified, or that final document review/submission is complete.
