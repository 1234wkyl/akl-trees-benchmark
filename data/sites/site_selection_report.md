# Reference site selection

Seed 42. Site size 100 m. 24 sites selected from 50,306 eligible cells.

| Stratum | Sites | Available cells |
|---|--:|--:|
| open_street_trees | 3 | 1,004 |
| park_open_grown | 3 | 16,279 |
| dense_broadleaf | 3 | 2,972 |
| conifer_dominant | 3 | 19,526 |
| palm_present | 3 | 226 |
| small_subcanopy | 3 | 261 |
| canopy_edge | 3 | 7,908 |
| hard_negative_built | 3 | 2,130 |

| Split | Sites |
|---|--:|
| test | 8 |
| train | 8 |
| val | 8 |

Minimum separation between sites: 400 m.
Minimum buffer between different splits: 1000 m.
Splits assigned by 4000 m spatial block, so neighbouring sites
cannot land in different splits and leak information.
