"""K-FOCUS: fixed-confidence Koopman residual tail search."""

from .stats import tail_index, tail_index_bounds, anytime_dkw_radius

__all__ = ["tail_index", "tail_index_bounds", "anytime_dkw_radius"]
