#!/usr/bin/env python3
"""Static QA checks for network URL candidate helper logic."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAPSERVICE_PATH = ROOT / "core" / "basemaps" / "mapservice.py"
DEM_PATH = ROOT / "operators" / "io_get_dem.py"


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _load_function(path: Path, func_name: str):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            fn = node
            break
    if fn is None:
        raise RuntimeError(f"Function not found: {func_name} in {path}")
    module = ast.Module(body=[fn], type_ignores=[])
    ns = {}
    exec(compile(module, filename=str(path), mode="exec"), ns)
    return ns[func_name]


def test_tile_url_candidates():
    fn = _load_function(MAPSERVICE_PATH, "_build_tile_url_candidates")
    cands = fn("http://example.com/tile.png")
    _assert(cands == ["http://example.com/tile.png", "https://example.com/tile.png"], "http->https fallback mismatch")

    cands2 = fn("https://example.com/tile.png")
    _assert(cands2 == ["https://example.com/tile.png"], "https URL should not duplicate")


def test_dem_url_candidates():
    fn = _load_function(DEM_PATH, "_build_dem_url_candidates")
    base_gl3 = "https://x?demtype=SRTMGL3&west=1"
    base_gl1 = "https://x?demtype=SRTMGL1&west=1"

    c1 = fn(base_gl3, prefer_high_res_srtm=True, allow_srtm_fallback=True)
    _assert(c1[0].find("SRTMGL1") >= 0 and c1[1].find("SRTMGL3") >= 0, "GL3 should prioritize GL1")

    c2 = fn(base_gl1, prefer_high_res_srtm=True, allow_srtm_fallback=True)
    _assert(len(c2) == 2 and c2[0].find("SRTMGL1") >= 0 and c2[1].find("SRTMGL3") >= 0, "GL1 should append GL3 fallback")

    c3 = fn(base_gl3, prefer_high_res_srtm=False, allow_srtm_fallback=True)
    _assert(c3 == [base_gl3], "No high-res preference should keep base GL3")

    other = "https://x?demtype=AW3D30&west=1"
    c4 = fn(other, prefer_high_res_srtm=True, allow_srtm_fallback=True)
    _assert(c4 == [other], "Non-SRTM URL should remain unchanged")


def main():
    test_tile_url_candidates()
    test_dem_url_candidates()
    print("qa_network_helpers_static: PASS")


if __name__ == "__main__":
    main()

