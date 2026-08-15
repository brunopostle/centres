"""Validation harness for the centres pipeline.

This package does not compute wholeness. It tests whether the measures in
``centres.properties`` behave like measurements at all: whether they are
stable under transformations that cannot change an artwork's structure,
whether they distinguish structure from its absence, and whether they
report the image rather than the pipeline's own tuning constants.

Run with::

    python -m audit
"""
