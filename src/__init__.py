"""Compatibility package so `python -m src.*` works from repository root.

This package extends its module search path to include the implementation
package at ``pcb_yolo/src``.
"""

from pathlib import Path

_impl_src = Path(__file__).resolve().parent.parent / "pcb_yolo" / "src"
if _impl_src.exists():
    __path__.append(str(_impl_src))
