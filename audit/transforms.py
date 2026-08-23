"""Image transformations grouped by how much they may legitimately change a score.

Re-exported from :mod:`centres.transforms`, which is the single source of truth —
the product's ``analyse --robust`` mode (#23) and this harness score over the same
transform definitions. See that module for the rationale behind the ISOMETRY /
BENIGN / PRACTICAL grouping.
"""

from centres.transforms import (  # noqa: F401
    BENIGN,
    ISOMETRY,
    PRACTICAL,
    crop,
    gamma,
    identity,
    invert,
    jpeg,
    mirror,
    pad,
    perspective,
    rot90,
    vignette,
)
