PYTHON ?= python
.DEFAULT_GOAL := help
.PHONY: help test check reference composition label receive-tw
help:
	@echo "make test       synthetic and workflow verification"
	@echo "make check      analysis environment and input access"
	@echo "make reference  derive author-confirmed scoring geometry in fresh outputs"
	@echo "make composition  class/split/stratum counts"
	@echo "make label      serve the annotation interface (stage site chips first)"
	@echo "make receive-tw TW_JSON=/path/file.json  reproduce original blind comparison"
test:
	$(PYTHON) -m pytest tests -q
check:
	$(PYTHON) scripts/00_check_environment.py
reference:
	$(PYTHON) scripts/45_prepare_prefreeze_scoring_candidate.py --labels-json data/labels/crown_labels_XZ_confirmed_2026-10-10.json --output outputs/reproduction/reference
composition:
	$(PYTHON) scripts/44_summarise_reference_composition.py --current data/labels/crown_labels_XZ_confirmed_2026-10-10.json --output outputs/reproduction/composition
label:
	$(PYTHON) scripts/03_serve_labeller.py
receive-tw:
	@test -n "$(TW_JSON)" || (echo 'Set TW_JSON to the returned JSON path'; exit 2)
	$(PYTHON) scripts/31_receive_independent_labels.py --tim-json "$(TW_JSON)"
