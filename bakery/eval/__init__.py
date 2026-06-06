"""Evaluation: a metric registry + composable metrics that land in metrics.json.

Kept import-light: metric modules (which touch the training stack) live under
`eval/metrics/` and are imported on demand by the runner, not at package import.
"""
