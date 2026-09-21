# Independent annotation protocol

This file supplies the annotation rules used by `scripts/07_prepare_blind_package.py`. That script builds a site-specific package containing only RGB/CHM chips and a labeller. The current blind pair is S016 and S020; its original export must be archived before comparison with the original author labels.

## Annotation rules

- `tree`: one defensibly identifiable visible crown; follow irregular observable edges. Internal shadow, minor colour differences, and crown contact alone do not prove multiple trees. A continuous crown with several apparent stems remains one object when separate individuals cannot be established.
- `tree_group`: multiple plausible crowns without a repeatable internal division. Do not invent single-tree boundaries.
- `uncertain`: unresolved identity or boundary after RGB and CHM inspection, including shadow, saturation, and conflicting evidence. CHM height alone does not prove a tree.
- `not_tree`: optionally outline clear non-tree confusers such as roofs, roads, vehicles, water, and bare ground that could be mistaken for canopy. These are supplementary examples, not an exhaustive negative audit.
- Retain only the observable portion of a boundary crown within the 100 m site. Include overhanging canopy above roads and roofs.

The independent JSON must be saved before either annotator sees the other's labels. Agreement uses one-to-one polygon IoU ≥ 0.50, with overall and individual-tree counts shown; matched IoU and class kappa are undefined when their denominators or category variation are insufficient. Disagreements are listed before adjudication.
