# Changelog

## 0.7.1 - Profiler refinement candidate

- Added independent PyTorch operator FLOP estimates and an explicitly opt-in CUDA allocator peak measurement.
- Improved Model Inspector batch input, per-layer MAC chart and readable shape findings.
- Added regression tests and actual desktop architecture gallery capture; CUDA hardware verification remains outstanding.

## 0.7.0 - Architecture profiler candidate

- Added executed leaf-module traces, tensor metadata, parameter inventories and partial Linear/Conv MAC/FLOP estimates.
- Added warmup and repeated CPU/CUDA forward timing, logical output bytes and CUDA allocator observations.
- Extended the desktop Model inspector, CLI, report findings and examples without removing previous workspaces.
- Added analytical and cleanup regression tests; real CUDA validation remains outstanding.

## 0.6.0 - Training analytics candidate

- Added configurable loss-trend signals, gradient threshold observations and per-batch class counts.
- Added optional JSONL event archives and explicitly selected bounded image previews for desktop and notebook snapshots.
- Added learning-rate capture and repeated final-validation callback handling; added a real CPU YOLO smoke job.

## 0.5.0 - Live training monitoring candidate

- Added atomic local training snapshots, progress, throughput, ETA, bounded metric history and output metadata.
- Added independent jade-themed desktop live viewer and opt-in Ultralytics callback adapter.
- Added separate validation metric history, selectable batch metric curves, optional CPU/RAM and PyTorch CUDA allocator sampling, and notebook snapshot rendering.
- Added synthetic demo and tests; retained existing Studio, CLI and Python APIs.

## 0.4.0 - Automation and training health release candidate

- Added composable Python quality gates for dataset, group, image, YOLO and classification checks.
- Added explicit PyTorch loss and gradient health guards, optional model parameter budget and custom gradient norm limits.
- Added synthetic training example and automated tests; retained all desktop and CLI features.

## 0.3.0 - Jade Edition release candidate

- Redesigned Studio with a cream canvas, deep jade navigation and restrained gold accents.
- Integrated a transparent XVARNA logo, application icon and branded About panel crediting Alireza Khatibi.
- Added searchable, scrollable navigation, collapsible sidebar, remembered workspace inputs and keyboard shortcuts.
- Added searchable findings, row copying, plot hover inspection and interactive binary threshold previews.
- HTML export now reflects the selected probability chart.
- Refreshed actual application screenshots and expanded regression tests. All 12 existing tools and CLI commands are retained.

## 0.2.0 - Studio Professional release candidate

- Redesigned the desktop dashboard, navigation, configuration panels, report tabs and charts.
- Added six new workspaces: image integrity and perceptual matching, group leakage, YOLO label audit, experiment comparison, binary score evaluation and reproducibility manifests.
- Added ROC, precision-recall and calibration visualization; ranked run comparisons and specific findings previews.
- Added self-contained HTML reports, embedded SVG charts, and a bounded in-memory session history.
- Added synthetic demo inputs for nine workspaces and Windows setup/launch scripts.
- Preserved the existing CLI and Python API while extending test and CI coverage.
- No cloud backend, telemetry, checkpoint deserialization or automatic dataset modification.

## 0.1.0

- Initial six research utilities with CLI, Python API and Tkinter interface.
- JSON export, basic charts, synthetic examples, tests and MIT license.
