# Release candidate validation

Version: 0.3.0. Local validation environment: Linux, Python 3.13.5, Pillow 12.3.0, PyTorch 2.10.0+cpu, pytest 9.0.2, scikit-learn 1.8.0. No network or private datasets were needed for these checks.

## Recorded results

- 81 automated tests passed, including the actual Tkinter desktop smoke test under Xvfb, and all 12 analysis workspaces exercised. No tests failed or were skipped in this environment.
- Previous v0.2.0 validation: 100 deterministic synthetic binary datasets (including tied probabilities) were compared to scikit-learn. All 300 metric comparisons (ROC-AUC, average precision and Brier score) matched within relative and absolute tolerance 1e-12.
- In v0.3.0, new tests cover navigation filtering, focus mode, retained inputs, findings filters, threshold previews and applied results, probability chart export, and the About window. Compact results and collapsed-input mode were inspected at 1000 × 720.
- The Python wheel built successfully using local setuptools, and an isolated-target installation outside the source directory passed the version and bundled binary-score demo checks.
- Real screenshots were captured from the running app at 1440 x 900 and 1000 x 720 using synthetic inputs.
- Automated tests cover legacy behavior, group leakage, redaction, YOLO label errors, bounded image matching, score validation, tied scores, comparisons, manifests, HTML escaping, report charts and bundled demos.

## Not validated

Windows rendering and setup scripts, macOS rendering, Python 3.10/3.14/3.15 execution, GPU/CUDA performance, GitHub Actions remote runs, large production datasets, clinical use, installer signing, and PyPI release have not been validated. The tools are research utilities, not a clinical or commercial certification. The 5,000-image cap for similarity is intentional; exact duplicate and group audits solve different problems. The comparison test against scikit-learn is a local independent cross-check, not proof for all possible inputs.

Run the local suite with `python -m pytest -q` or `xvfb-run -a python -m pytest -q` on Linux. All GUI screenshots in this package show synthetic data only.
