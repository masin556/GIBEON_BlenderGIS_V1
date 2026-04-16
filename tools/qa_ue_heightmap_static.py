#!/usr/bin/env python3
"""Static QA checks for UE heightmap export math/helpers.

Runs pure-Python checks by extracting helper functions directly from
operators/io_export_ue_heightmap.py, so tests stay aligned with source logic
without requiring Blender runtime.
"""

from __future__ import annotations

import ast
import logging
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "operators" / "io_export_ue_heightmap.py"

WANTED_CONSTS = {
    "_UE_Z_RANGE_CM_AT_100",
    "_DEFAULT_UE_SCALE_Z",
    "_LAYOUT_EPSILON",
    "_U16_LEVELS",
}
WANTED_FUNCS = {
    "_calc_ue_layout",
    "_validate_layout",
    "_calc_ue_scale_z",
    "_calc_vertical_precision_cm",
    "_max_z_range_for_precision",
    "_normalize_heightmap_u16",
    "_fill_missing_heights",
    "_tile_slices",
    "_choose_tile_grid_for_precision",
    "_snap_value",
    "_snap_location_cm",
}


def _load_helpers():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))

    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                t.id for t in node.targets if isinstance(t, ast.Name)
            }
            if names & WANTED_CONSTS:
                selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCS:
            selected_nodes.append(node)

    module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace = {
        "np": np,
        "math": math,
        "log": logging.getLogger("qa_ue_heightmap_static"),
    }
    exec(compile(module, filename=str(SOURCE_PATH), mode="exec"), namespace)
    missing = [name for name in WANTED_FUNCS if name not in namespace]
    if missing:
        raise RuntimeError(f"Failed to load helper(s): {missing}")
    return namespace


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def test_layout_checks(ns):
    calc_layout = ns["_calc_ue_layout"]
    validate_layout = ns["_validate_layout"]

    layout = calc_layout(1000.0, 500.0, 2.0, section_size=63, sec_per_comp=1)
    errors = validate_layout(layout, 2.0)

    _assert(not errors, f"layout validation errors: {errors}")
    _assert(layout["res_x"] == layout["quads_x"] + 1, "res_x formula mismatch")
    _assert(layout["res_y"] == layout["quads_y"] + 1, "res_y formula mismatch")
    _assert(layout["actual_mpp_x"] <= 2.0 + 1e-9, "actual_mpp_x exceeds target")
    _assert(layout["actual_mpp_y"] <= 2.0 + 1e-9, "actual_mpp_y exceeds target")


def test_z_scale(ns):
    calc_z = ns["_calc_ue_scale_z"]

    _assert(abs(calc_z(512.0) - 100.0) < 1e-9, "512m should map to 100%")
    _assert(abs(calc_z(1024.0) - 200.0) < 1e-9, "1024m should map to 200%")
    _assert(abs(calc_z(0.0) - 100.0) < 1e-9, "0m should use fallback 100%")


def test_vertical_precision_math(ns):
    calc_step = ns["_calc_vertical_precision_cm"]
    max_range = ns["_max_z_range_for_precision"]

    _assert(abs(calc_step(655.35) - 1.0) < 1e-3, "655.35m should be ~1cm step")
    _assert(abs(calc_step(0.0) - 0.0) < 1e-9, "0m range should be 0cm step")
    _assert(abs(max_range(1.0) - 655.35) < 1e-2, "1cm target max range should be ~655.35m")
    _assert(calc_step(800.0) > 1.0, "800m range should exceed 1cm step")


def test_raw_little_endian():
    hmap = np.array([[0, 1], [256, 65535]], dtype=np.uint16)
    raw_bytes = hmap.astype("<u2", copy=False).tobytes(order="C")
    _assert(len(raw_bytes) == hmap.size * 2, "raw byte length mismatch")
    round_trip = np.frombuffer(raw_bytes, dtype="<u2").reshape(hmap.shape)
    _assert(np.array_equal(round_trip, hmap), "raw little-endian round-trip mismatch")


def test_fill_and_normalize(ns):
    fill_missing = ns["_fill_missing_heights"]
    normalize = ns["_normalize_heightmap_u16"]

    hmap = np.array(
        [
            [np.nan, 2.0, np.nan],
            [3.0, np.nan, 4.0],
        ],
        dtype=np.float64,
    )
    missing_count, _iterations = fill_missing(hmap, min_z=1.0)
    _assert(missing_count > 0, "expected missing values before fill")
    _assert(not np.isnan(hmap).any(), "fill should remove NaN values")

    u16 = normalize(hmap, min_z=1.0, max_z=4.0)
    _assert(u16.dtype == np.uint16, "normalize dtype should be uint16")
    _assert(int(u16.min()) >= 0, "normalize min out of range")
    _assert(int(u16.max()) <= 65535, "normalize max out of range")

    flat = normalize(np.full((2, 2), 5.0), min_z=5.0, max_z=5.0)
    _assert(np.all(flat == 32768), "flat terrain should map to midpoint 32768")

    all_missing = np.full((2, 2), np.nan, dtype=np.float64)
    fill_missing(all_missing, min_z=7.0)
    _assert(np.all(all_missing == 7.0), "all-missing map should fill with min_z")


def test_tiling_and_snap(ns):
    tile_slices = ns["_tile_slices"]
    choose_tiles = ns["_choose_tile_grid_for_precision"]
    snap_value = ns["_snap_value"]
    snap_loc = ns["_snap_location_cm"]

    slices = tile_slices(257, 2)
    _assert(slices[0][0] == 0, "first tile should start at 0")
    _assert(slices[-1][1] == 257, "last tile should end at length")
    _assert(slices[0][1] > slices[1][0], "tile boundaries should overlap by one sample")

    x = np.linspace(0.0, 1000.0, 257, dtype=np.float64)
    hmap = np.tile(x[None, :], (257, 1))
    tx, ty, worst = choose_tiles(hmap, target_vertical_precision_cm=1.0, max_tiles=4)
    _assert(tx * ty > 1, "precision tiling should split map when needed")
    _assert(worst <= 1.0 + 1e-9, "worst per-tile precision should satisfy target")

    _assert(abs(snap_value(123.456, 1.0) - 123.0) < 1e-9, "snap value (1cm step) mismatch")
    sx, sy, sz = snap_loc((123.456, -78.901, 0.005), 0.01)
    _assert(abs(sx - 123.46) < 1e-9, "snap location x mismatch")
    _assert(abs(sy + 78.9) < 1e-9, "snap location y mismatch")
    _assert(abs(sz) < 1e-9, "snap location z mismatch")


def main():
    ns = _load_helpers()
    test_layout_checks(ns)
    test_z_scale(ns)
    test_vertical_precision_math(ns)
    test_raw_little_endian()
    test_fill_and_normalize(ns)
    test_tiling_and_snap(ns)
    print("qa_ue_heightmap_static: PASS")


if __name__ == "__main__":
    main()
