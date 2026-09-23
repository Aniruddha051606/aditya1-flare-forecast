"""Flare forecasting from Aditya-L1 SUIT full-disk ultraviolet images (see README.md).

Separate from the X-ray model on purpose: it reads SUIT images, but shares the
GOES truth, the train/test split and the scoring of solarflare, so the two are
scored on the same days and can later be combined.
"""
