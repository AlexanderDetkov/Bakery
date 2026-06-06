"""Trajectory data + the safety kernel.

The gate (``base.py``) is the only producer of the frozen ``TrajectoryDataset`` that
training/eval accept. Keep this package import light: builders and generation backends
(which load models) are imported by the experiment/runner that uses them, not here.
"""
