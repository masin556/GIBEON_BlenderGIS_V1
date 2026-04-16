#!/usr/bin/env python3
"""QA runner for ASC DEM -> UE heightmap export with 1cm precision targeting.

This script can:
- read a real .asc DEM file,
- optionally auto-tile it so each tile can satisfy a target vertical precision,
- export each tile using BlenderGIS UE exporter,
- validate per-tile JSON outputs.

Usage (inside Blender):
  blender --background --factory-startup --python tools/qa_ue_heightmap_from_asc.py -- \
    --asc path/to/dem.asc --outdir tools/qa_out_dem --target-vertical-cm 1.0 --auto-tile
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import bmesh
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from BlenderGIS.operators import io_export_ue_heightmap


@dataclass
class AscMeta:
    ncols: int
    nrows: int
    xll: float
    yll: float
    cellsize: float
    nodata: float


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser()
    p.add_argument("--asc", type=str, default="", help="Path to input ASC DEM")
    p.add_argument("--outdir", type=str, default=str(ROOT / "tools" / "qa_out_dem"))
    p.add_argument("--target-mpp", type=float, default=1.5)
    p.add_argument("--target-vertical-cm", type=float, default=1.0)
    p.add_argument("--sampling-grid", choices=["1", "2", "3", "4"], default="3")
    p.add_argument("--sampling-reduce", choices=["MEDIAN", "MEAN", "MAX"], default="MEDIAN")
    p.add_argument("--auto-tile", action="store_true")
    p.add_argument("--max-tiles", type=int, default=64)
    p.add_argument("--demo-size", type=int, default=193)
    p.add_argument("--demo-z-range", type=float, default=1400.0)
    return p.parse_args(argv)


def write_demo_asc(path: Path, size: int, z_range_m: float):
    size = max(33, int(size))
    local_amp = max(10.0, z_range_m * 0.12)
    global_span = max(20.0, z_range_m - (2.0 * local_amp))
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)
    # Global slope + local relief. This mimics large-area DEMs where tiling
    # can significantly reduce per-tile vertical range.
    dem = (
        (global_span * 0.5) * xx
        + local_amp * np.sin(xx * 3.1)
        + local_amp * np.cos(yy * 2.7)
        + 0.25 * local_amp * xx * yy
    ).astype(np.float64)

    header = (
        f"ncols {size}\n"
        f"nrows {size}\n"
        "xllcorner 0\n"
        "yllcorner 0\n"
        "cellsize 1\n"
        "NODATA_value -9999\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for row in dem:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def read_asc(path: Path) -> tuple[AscMeta, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        lines = [f.readline().strip() for _ in range(6)]
        values = {}
        for line in lines:
            m = re.match(r"^([^\s]+)\s+([^\s]+)$", line)
            if not m:
                raise ValueError(f"Invalid ASC header line: {line}")
            values[m.group(1).lower()] = m.group(2)

        ncols = int(values["ncols"])
        nrows = int(values["nrows"])
        cellsize = float(values["cellsize"])
        nodata = float(values.get("nodata_value", "-9999"))
        xll = float(values.get("xllcorner", values.get("xllcenter", "0")))
        yll = float(values.get("yllcorner", values.get("yllcenter", "0")))

        data = []
        for _ in range(nrows):
            row = f.readline()
            if not row:
                raise ValueError("ASC ended before reading all rows")
            vals = row.split()
            if len(vals) != ncols:
                raise ValueError(f"ASC row has {len(vals)} cols, expected {ncols}")
            data.append([float(v) for v in vals])

    arr = np.asarray(data, dtype=np.float64)
    arr[arr == nodata] = np.nan
    return AscMeta(ncols, nrows, xll, yll, cellsize, nodata), arr


def fill_nans(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if not np.isnan(out).any():
        return out
    valid = ~np.isnan(out)
    if not valid.any():
        raise ValueError("DEM contains only nodata")

    min_v = float(np.nanmin(out))
    missing = np.isnan(out)
    max_iter = out.shape[0] + out.shape[1]
    for _ in range(max_iter):
        if not missing.any():
            break
        up = np.roll(out, 1, axis=0); up[0, :] = np.nan
        dn = np.roll(out, -1, axis=0); dn[-1, :] = np.nan
        lf = np.roll(out, 1, axis=1); lf[:, 0] = np.nan
        rt = np.roll(out, -1, axis=1); rt[:, -1] = np.nan
        neighbors = np.stack([up, dn, lf, rt], axis=0)
        counts = np.sum(~np.isnan(neighbors), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where(counts > 0, np.nansum(neighbors, axis=0) / counts, np.nan)
        fill_now = missing & ~np.isnan(vals)
        if not fill_now.any():
            break
        out[fill_now] = vals[fill_now]
        missing = np.isnan(out)
    out[np.isnan(out)] = min_v
    return out


def vertical_step_cm(data: np.ndarray) -> float:
    zmin = float(np.nanmin(data))
    zmax = float(np.nanmax(data))
    zrange = zmax - zmin
    if zrange <= 0:
        return 0.0
    return (zrange * 100.0) / 65535.0


def tile_slices(length: int, tiles: int) -> list[tuple[int, int]]:
    # Overlap one row/col at boundaries to avoid seams.
    max_idx = length - 1
    slices = []
    for t in range(tiles):
        s = round((t * max_idx) / tiles)
        e = round(((t + 1) * max_idx) / tiles) + 1
        e = min(e, length)
        slices.append((s, e))
    return slices


def choose_tiles(data: np.ndarray, target_cm: float, max_tiles: int) -> tuple[int, int, float]:
    best = None
    best_relaxed = None
    rows, cols = data.shape
    for ty in range(1, max_tiles + 1):
        for tx in range(1, max_tiles + 1):
            if tx * ty > max_tiles:
                continue
            rs = tile_slices(rows, ty)
            cs = tile_slices(cols, tx)
            worst = 0.0
            ok = True
            for ry in rs:
                for cx in cs:
                    tile = data[ry[0]:ry[1], cx[0]:cx[1]]
                    step = vertical_step_cm(tile)
                    worst = max(worst, step)
                    if step > target_cm:
                        ok = False
                        break
                if not ok:
                    break
            relaxed = (worst, tx * ty, tx, ty)
            if best_relaxed is None or relaxed < best_relaxed:
                best_relaxed = relaxed
            if ok:
                cand = (tx * ty, worst, tx, ty)
                if best is None or cand < best:
                    best = cand
    if best is None:
        if best_relaxed is not None:
            w, total, tx, ty = best_relaxed
            raise RuntimeError(
                f"Could not satisfy {target_cm:.3f}cm precision within max_tiles={max_tiles}. "
                f"Best attempt: {tx}x{ty} ({total} tiles), worst tile step={w:.3f}cm"
            )
        raise RuntimeError(
            f"Could not satisfy {target_cm:.3f}cm precision within max_tiles={max_tiles}"
        )
    return best[2], best[3], best[1]


def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def build_mesh_from_tile(tile: np.ndarray, cellsize: float, name: str):
    rows, cols = tile.shape
    verts = []
    faces = []

    # ASC rows are top->bottom; Blender Y forward, so invert row index for natural orientation.
    for r in range(rows):
        y = (rows - 1 - r) * cellsize
        for c in range(cols):
            x = c * cellsize
            verts.append((x, y, float(tile[r, c])))

    for r in range(rows - 1):
        for c in range(cols - 1):
            i = r * cols + c
            faces.append((i, i + cols, i + cols + 1, i + 1))

    me = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    me.from_pydata(verts, [], faces)
    me.update()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def export_tile(obj, out_base: Path, args) -> tuple[dict, dict]:
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    result = bpy.ops.exportgis.ue_heightmap(
        'EXEC_DEFAULT',
        filepath=str(out_base) + ".png",
        target_mpp=args.target_mpp,
        target_vertical_precision_cm=args.target_vertical_cm,
        enforce_vertical_precision=True,
        sampling_grid=args.sampling_grid,
        sampling_reduce=args.sampling_reduce,
        smooth_detail=False,
        section_size='63',
        sec_per_comp='1',
        export_raw=True,
        export_fbx=False,
        save_json=True,
    )
    if result != {'FINISHED'}:
        raise RuntimeError(f"Export failed for tile {out_base.name}: {result}")

    meta_path = Path(str(out_base) + "_ue_scale.json")
    if not meta_path.exists():
        raise RuntimeError(f"Missing metadata for tile {out_base.name}")
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    ue = meta["ue_settings"]
    if not ue.get("vertical_precision_ok", False):
        raise RuntimeError(f"Tile precision check failed: {out_base.name}")
    if ue.get("vertical_step_cm", 999.0) > args.target_vertical_cm + 1e-6:
        raise RuntimeError(
            f"Tile vertical step exceeds target: {out_base.name} -> {ue.get('vertical_step_cm')} cm"
        )
    return meta, ue


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    asc_path = Path(args.asc) if args.asc else outdir / "demo_input.asc"
    if not asc_path.exists():
        write_demo_asc(asc_path, args.demo_size, args.demo_z_range)

    meta, data = read_asc(asc_path)
    data = fill_nans(data)

    global_step = vertical_step_cm(data)
    print(f"input_asc: {asc_path}")
    print(f"input_shape: {data.shape[1]}x{data.shape[0]}")
    print(f"global_vertical_step_cm: {global_step:.6f}")

    if global_step <= args.target_vertical_cm:
        tx, ty, worst = 1, 1, global_step
    elif args.auto_tile:
        tx, ty, worst = choose_tiles(data, args.target_vertical_cm, args.max_tiles)
    else:
        raise RuntimeError(
            f"Global precision {global_step:.4f} cm exceeds target {args.target_vertical_cm:.4f} cm. "
            "Use --auto-tile or provide lower Z-range DEM."
        )

    print(f"tile_grid: {tx}x{ty} (worst_tile_step_cm={worst:.6f})")

    io_export_ue_heightmap.register()
    try:
        reset_scene()
        row_slices = tile_slices(data.shape[0], ty)
        col_slices = tile_slices(data.shape[1], tx)

        summaries = []
        for yi, ry in enumerate(row_slices):
            for xi, cx in enumerate(col_slices):
                tile = data[ry[0]:ry[1], cx[0]:cx[1]]
                tile_name = f"tile_{yi:02d}_{xi:02d}"
                obj = build_mesh_from_tile(tile, meta.cellsize, tile_name)
                out_base = outdir / f"{asc_path.stem}_{tile_name}"
                tile_meta, ue = export_tile(obj, out_base, args)
                summaries.append((tile_name, ue["vertical_step_cm"], tile_meta["resolution_x"], tile_meta["resolution_y"]))

                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                bpy.ops.object.delete(use_global=False)

        print("qa_ue_heightmap_from_asc: PASS")
        for name, step_cm, rx, ry in summaries:
            print(f"  {name}: {rx}x{ry}, vertical_step_cm={step_cm:.6f}")
    finally:
        io_export_ue_heightmap.unregister()


if __name__ == "__main__":
    main()
