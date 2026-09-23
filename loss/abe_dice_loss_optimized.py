"""Backward-compatible alias for the optimized default loss.

New training commands should use ``--loss main_loss``.
"""

from .main_loss import MainLoss, MultiScopeLoss


class AbeDiceLossOptimized(MainLoss):
    """Compatibility wrapper for historical configs and checkpoints."""

    pass
