PYTHON ?= python3

.PHONY: test verify aggregate

test:
	$(PYTHON) -m unittest discover -s tests -v

verify:
	$(PYTHON) scripts/verify_artifact.py

aggregate:
	$(PYTHON) scripts/aggregate.py results/rtx5090 results/h100 --output-dir results/reproduced
