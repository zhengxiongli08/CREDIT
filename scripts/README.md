# Scripts

The scripts are separated by responsibility:

- `run_local.py` and `run_modal.py` collect new measurements. They consume CUDA
  binaries produced by the root Makefile; compiler commands do not live here.
- `analyze_results.py` and `rescore_cost_model.py` transform saved result
  bundles into derived CSV tables and reports.
- `verify_artifact.py` checks the tracked release data.
- `plot/` contains one entry point per reproducible paper figure.

Run scripts from the repository root so their documented default paths resolve
under `results/`.
