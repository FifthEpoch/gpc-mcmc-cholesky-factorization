"""
Utility functions for working with low-rank PSD approximations
and kernel matrices in the my_cholesky project.
"""


def approximation_error(A, lra, relative: bool = False):
    """
    Compute Frobenius-trace-based approximation error between A and its low-rank
    approximation lra, using just traces.

    error = trace(A) - trace(lra)
    If relative is True, returns error / trace(A).
    """
    error = A.trace() - lra.trace()
    if relative:
        error /= A.trace()
    return error

