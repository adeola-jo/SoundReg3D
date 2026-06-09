"""Microphone array geometry.

Two sources of geometry:

    circular_array   parametric uniform circular array; what the
                     synthetic simulator uses.

    load_mic_xml     parser for the real array's geometry XML (M1).
                     The exact schema of the on-hand file is not known
                     yet, so the parser is deliberately tolerant: it
                     collects every element that exposes x/y/z either as
                     attributes or as child elements. When the real file
                     lands, tighten this to its actual schema.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import numpy as np


def circular_array(n_mics: int, radius_m: float, z_m: float = 0.0) -> np.ndarray:
    """Uniform circular array in the BEV plane.

    Args:
        n_mics: Number of microphones, evenly spaced over the circle.
        radius_m: Array radius (m).
        z_m: Mounting height of the ring (m).

    Returns:
        (M, 3) float64 positions, ego frame, meters. Mic 0 sits on the
        +x (forward) axis; the rest follow counter-clockwise.
    """
    angles = 2.0 * np.pi * np.arange(n_mics) / n_mics
    pos = np.stack(
        [radius_m * np.cos(angles), radius_m * np.sin(angles), np.full(n_mics, z_m)],
        axis=1,
    )
    return pos


def load_mic_xml(path) -> np.ndarray:
    """Parse microphone positions from an XML geometry file.

    Accepts both attribute style and child-element style:

        <mic x="0.1" y="-0.2" z="0.5"/>
        <mic><x>0.1</x><y>-0.2</y><z>0.5</z></mic>

    Returns:
        (M, 3) float64 positions in file order.

    Raises:
        ValueError: If nothing position-like was found — with a pointer
            to this function, because the fix is adapting the parser to
            the real schema, not debugging the call site.
    """
    root = ET.parse(Path(path)).getroot()
    positions: List[List[float]] = []
    for elem in root.iter():
        coords = _xyz_from_element(elem)
        if coords is not None:
            positions.append(coords)
    if not positions:
        raise ValueError(
            f"No microphone positions found in {path}. "
            "Adapt soundreg/data/geometry.py:load_mic_xml to the file's schema."
        )
    return np.asarray(positions, dtype=np.float64)


def _xyz_from_element(elem) -> Optional[List[float]]:
    """[x, y, z] floats from one element, or None if it has no full
    coordinate triple in either supported style."""
    keys = ("x", "y", "z")
    attrs = {k.lower(): v for k, v in elem.attrib.items()}
    if all(k in attrs for k in keys):
        try:
            return [float(attrs[k]) for k in keys]
        except ValueError:
            return None
    children = {child.tag.lower(): child.text for child in elem}
    if all(k in children for k in keys):
        try:
            return [float(children[k]) for k in keys]
        except (TypeError, ValueError):
            return None
    return None
