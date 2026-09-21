# Auckland tree detection benchmark, starter kit.
# Run `make` on its own to see what is available.

PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help check sites chips chips-points label export agreement receive-tw smoke test-benchmark clean-labels

help:
	@echo ""
	@echo "  Auckland tree detection benchmark"
	@echo ""
	@echo "  make check         verify environment and data access   (start here)"
	@echo "  make sites         choose stratified reference sites"
	@echo "  make chips         cut imagery and canopy height per site"
	@echo "  make chips-points  same, and clip the point cloud (needs the drive)"
	@echo "  make label         open the crown labeller"
	@echo "  make export        build the reference set from your labels"
	@echo "  make agreement     inter-rater agreement (A=XZ B=TW)"
	@echo "  make receive-tw TW_JSON=/path/returned.json  archive, compare and prepare adjudication"
	@echo "  make smoke         GPU smoke test (run this on the Nectar VM)"
	@echo "  make test-benchmark known-answer synthetic scoring and workflow checks"
	@echo ""
	@echo "  Normal first day:  make check && make sites && make chips && make label"
	@echo ""

check:
	$(PYTHON) scripts/00_check_environment.py

sites:
	$(PYTHON) scripts/01_select_reference_sites.py

chips:
	$(PYTHON) scripts/02_extract_site_chips.py

chips-points:
	$(PYTHON) scripts/02_extract_site_chips.py --with-points

label:
	$(PYTHON) scripts/03_serve_labeller.py

export:
	$(PYTHON) scripts/04_export_reference_set.py

A ?= XZ
B ?= TW
agreement:
	$(PYTHON) scripts/05_interrater_agreement.py --a $(A) --b $(B)

receive-tw:
	@test -n "$(TW_JSON)" || (echo 'Set TW_JSON to the original returned JSON path'; exit 2)
	$(PYTHON) scripts/31_receive_independent_labels.py --tim-json "$(TW_JSON)"

smoke:
	$(PYTHON) scripts/hpc/smoke_test_gpu.py

test-benchmark:
	$(PYTHON) -m unittest discover -s tests -v

# Deliberately does not touch data/labels. Your labels are hand work and this
# will not be the command that deletes them.
clean-labels:
	@echo "Not deleting anything. Your labels are in data/labels/ and are hand work."
	@echo "If you really mean it, move the files yourself."
