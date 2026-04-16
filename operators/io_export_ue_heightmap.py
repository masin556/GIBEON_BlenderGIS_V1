# -*- coding:utf-8 -*-

#  ***** GPL LICENSE BLOCK *****
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  All rights reserved.
#  ***** GPL LICENSE BLOCK *****

"""
Export terrain mesh → 16-bit heightmap for Unreal Engine 5 Landscape.

Outputs
-------
* 16-bit PNG  (non-square, matching real terrain aspect ratio)
* UE Import Guide (.txt) — exact values for ALL UE5 Landscape panel fields
* JSON metadata (optional)
* .r16 RAW (optional)
"""

import os, json, math, logging
from datetime import datetime

import bpy, bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from bpy_extras.io_utils import ExportHelper, ImportHelper
from bpy.props import StringProperty, EnumProperty, BoolProperty, FloatProperty, IntProperty
from bpy.types import Operator

from ..geoscene import GeoScene

log = logging.getLogger(__name__)

# =====================================================================
# UE Landscape layout calculator
# =====================================================================
_SECTION_SIZES = [7, 15, 31, 63, 127, 255]
_UE_Z_RANGE_CM_AT_100 = 512.0
_DEFAULT_UE_SCALE_Z = 100.0
_LAYOUT_EPSILON = 1e-9
_U16_LEVELS = 65535.0
_MAX_RAYCAST_SAMPLES = 120_000_000


def _calc_vertical_precision_cm(rw_z_range_m):
    if not math.isfinite(rw_z_range_m):
        raise ValueError("rw_z_range_m must be finite")
    if rw_z_range_m <= 0:
        return 0.0
    return (rw_z_range_m * 100.0) / _U16_LEVELS


def _max_z_range_for_precision(target_vertical_precision_cm):
    if target_vertical_precision_cm <= 0:
        raise ValueError("target_vertical_precision_cm must be > 0")
    return (target_vertical_precision_cm * _U16_LEVELS) / 100.0

def _calc_ue_layout(rw_size_x_m, rw_size_y_m, target_m_per_px,
                    section_size=63, sec_per_comp=1):
    """
    Calculate a valid UE5 Landscape layout for a terrain of given
    real-world size (metres).

    Returns dict with all UE panel values, or None if impossible.

    UE formula per axis:
        Quads = SectionSize × SectionsPerComponent × NumComponents
        Resolution = Quads + 1
        WorldSize = Quads × ScaleXY  (cm)
    """
    if target_m_per_px <= 0:
        raise ValueError("target_m_per_px must be > 0")
    if rw_size_x_m <= 0 or rw_size_y_m <= 0:
        raise ValueError("real-world terrain sizes must be > 0")

    comp_quads = section_size * sec_per_comp  # quads per component

    # Components per axis = ceil(terrain_quads / comp_quads)
    target_quads_x = rw_size_x_m / target_m_per_px
    target_quads_y = rw_size_y_m / target_m_per_px

    comp_x = max(1, math.ceil(target_quads_x / comp_quads))
    comp_y = max(1, math.ceil(target_quads_y / comp_quads))

    quads_x = comp_quads * comp_x
    quads_y = comp_quads * comp_y
    res_x = quads_x + 1
    res_y = quads_y + 1

    # Scale = cm per quad  (so total size = quads × scale cm)
    scale_x = (rw_size_x_m * 100.0) / quads_x
    scale_y = (rw_size_y_m * 100.0) / quads_y

    actual_mpp_x = rw_size_x_m / quads_x
    actual_mpp_y = rw_size_y_m / quads_y

    return {
        'section_size': section_size,
        'sec_per_comp': sec_per_comp,
        'comp_x': comp_x,
        'comp_y': comp_y,
        'total_components': comp_x * comp_y,
        'quads_x': quads_x,
        'quads_y': quads_y,
        'res_x': res_x,
        'res_y': res_y,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'actual_mpp_x': actual_mpp_x,
        'actual_mpp_y': actual_mpp_y,
    }


def _validate_layout(layout, target_m_per_px):
    errors = []
    int_fields = (
        "section_size", "sec_per_comp", "comp_x", "comp_y",
        "total_components", "quads_x", "quads_y", "res_x", "res_y",
    )
    float_fields = ("scale_x", "scale_y", "actual_mpp_x", "actual_mpp_y")

    for key in int_fields:
        value = layout.get(key)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"Invalid layout[{key}]: {value}")

    for key in float_fields:
        value = layout.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errors.append(f"Invalid layout[{key}]: {value}")

    if layout.get("res_x") != layout.get("quads_x", 0) + 1:
        errors.append("Resolution X must be quads_x + 1")
    if layout.get("res_y") != layout.get("quads_y", 0) + 1:
        errors.append("Resolution Y must be quads_y + 1")

    if layout.get("actual_mpp_x", float("inf")) - target_m_per_px > _LAYOUT_EPSILON:
        errors.append(
            f"actual_mpp_x ({layout['actual_mpp_x']:.6f}) exceeds target_mpp ({target_m_per_px:.6f})"
        )
    if layout.get("actual_mpp_y", float("inf")) - target_m_per_px > _LAYOUT_EPSILON:
        errors.append(
            f"actual_mpp_y ({layout['actual_mpp_y']:.6f}) exceeds target_mpp ({target_m_per_px:.6f})"
        )

    return errors


def _calc_ue_scale_z(rw_z_range_m):
    if not math.isfinite(rw_z_range_m):
        raise ValueError("rw_z_range_m must be finite")
    if rw_z_range_m <= 0:
        return _DEFAULT_UE_SCALE_Z
    return (rw_z_range_m * 100.0) / _UE_Z_RANGE_CM_AT_100


def _normalize_heightmap_u16(hmap, min_z, max_z):
    z_bl = max_z - min_z
    if z_bl < 1e-9:
        return np.full(hmap.shape, 32768, dtype=np.uint16)

    safe_hmap = np.nan_to_num(hmap, nan=min_z, posinf=max_z, neginf=min_z)
    normalized = (safe_hmap - min_z) / z_bl
    # Use unbiased rounding instead of truncation to prevent micro-stepping/banding
    return np.clip(np.rint(normalized * 65535.0), 0, 65535).astype(np.uint16)


def _fill_missing_heights(hmap, min_z):
    mask = np.isnan(hmap)
    missing_count = int(mask.sum())
    if missing_count == 0:
        return 0, 0

    valid = ~mask
    if not valid.any():
        hmap.fill(min_z)
        return missing_count, 0

    # Try exact nearest-neighbor fill first.
    try:
        from scipy.ndimage import distance_transform_edt
        indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
        hmap[mask] = hmap[tuple(indices[:, mask])]
        remaining = int(np.isnan(hmap).sum())
        if remaining > 0:
            hmap[np.isnan(hmap)] = min_z
        return missing_count, 1
    except ImportError:
        pass

    # NumPy fallback: bounded iterative neighbor dilation.
    max_iterations = max(1, hmap.shape[0] + hmap.shape[1])
    missing_mask = mask.copy()
    iterations = 0
    while missing_mask.any() and iterations < max_iterations:
        up = np.roll(hmap, shift=1, axis=0); up[0, :] = np.nan
        down = np.roll(hmap, shift=-1, axis=0); down[-1, :] = np.nan
        left = np.roll(hmap, shift=1, axis=1); left[:, 0] = np.nan
        right = np.roll(hmap, shift=-1, axis=1); right[:, -1] = np.nan

        neighbors = np.stack([up, down, left, right], axis=0)
        valid_count = np.sum(~np.isnan(neighbors), axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            fill_values = np.where(
                valid_count > 0,
                np.nansum(neighbors, axis=0) / valid_count,
                np.nan,
            )

        filled_now = missing_mask & ~np.isnan(fill_values)
        if not filled_now.any():
            break

        hmap[filled_now] = fill_values[filled_now]
        missing_mask[filled_now] = False
        iterations += 1

    remaining = int(missing_mask.sum())
    if remaining > 0:
        hmap[missing_mask] = min_z

    return missing_count, iterations


# =====================================================================
# Height sampling — true vertical raycast
# =====================================================================
def _combine_subpixel_samples(sample_values, combine_mode):
    values = np.asarray(sample_values, dtype=np.float64)
    if combine_mode == 'MEAN':
        return float(np.mean(values))
    if combine_mode == 'MAX':
        return float(np.max(values))
    # MEDIAN is robust against outlier hits on steep triangles.
    return float(np.median(values))


def _tile_slices(length, tiles):
    tiles = max(1, int(tiles))
    if length <= 1:
        return [(0, length)]

    max_idx = length - 1
    slices = []
    for t in range(tiles):
        start = round((t * max_idx) / tiles)
        end = round(((t + 1) * max_idx) / tiles) + 1
        end = min(end, length)
        if end <= start:
            end = min(length, start + 1)
        slices.append((start, end))
    return slices


def _snap_value(value, step):
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step must be a positive finite value")
    return round(value / step) * step


def _snap_location_cm(location_cm, snap_step_cm):
    return (
        _snap_value(float(location_cm[0]), snap_step_cm),
        _snap_value(float(location_cm[1]), snap_step_cm),
        _snap_value(float(location_cm[2]), snap_step_cm),
    )


def _choose_tile_grid_for_precision(hmap, target_vertical_precision_cm, max_tiles):
    rows, cols = hmap.shape
    if rows <= 0 or cols <= 0:
        raise ValueError("heightmap is empty")

    best = None
    best_relaxed = None
    for ty in range(1, max_tiles + 1):
        for tx in range(1, max_tiles + 1):
            if tx * ty > max_tiles:
                continue
            if tx >= cols or ty >= rows:
                continue
            row_slices = _tile_slices(rows, ty)
            col_slices = _tile_slices(cols, tx)
            worst = 0.0
            ok = True
            for rs in row_slices:
                for cs in col_slices:
                    tile = hmap[rs[0]:rs[1], cs[0]:cs[1]]
                    step_cm = _calc_vertical_precision_cm(float(np.nanmax(tile) - np.nanmin(tile)))
                    worst = max(worst, step_cm)
                    if step_cm > target_vertical_precision_cm:
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
            worst, total, tx, ty = best_relaxed
            raise RuntimeError(
                f"Could not satisfy {target_vertical_precision_cm:.3f}cm precision with "
                f"max_tiles={max_tiles}. Best attempt: {tx}x{ty} ({total} tiles), "
                f"worst tile step={worst:.3f}cm"
            )
        raise RuntimeError(
            f"Could not satisfy {target_vertical_precision_cm:.3f}cm precision with max_tiles={max_tiles}"
        )

    _, worst, tx, ty = best
    return tx, ty, worst


def _sample_heights(bvh, min_x, max_x, min_y, max_y, min_z, max_z,
                    res_x, res_y, sample_grid=1, combine_mode='MEDIAN', context=None):
    """
    Sample heights on a res_x × res_y grid using true vertical raycasting.
    Find_nearest causes horizontal warping (terracing) on slopes.
    """
    dx = (max_x - min_x) / (res_x - 1) if res_x > 1 else 0
    dy = (max_y - min_y) / (res_y - 1) if res_y > 1 else 0
    sample_grid = max(1, int(sample_grid))
    combine_mode = combine_mode.upper().strip()
    if combine_mode not in {'MEAN', 'MEDIAN', 'MAX'}:
        combine_mode = 'MEDIAN'

    if sample_grid == 1:
        offsets = [(0.0, 0.0)]
    else:
        offsets = []
        for sy in range(sample_grid):
            oy = ((sy + 0.5) / sample_grid) - 0.5
            for sx in range(sample_grid):
                ox = ((sx + 0.5) / sample_grid) - 0.5
                offsets.append((ox, oy))
    
    # Raycast must be perfectly vertical, starting above the highest point
    ray_z = max_z + abs(max_z - min_z) + 10.0
    down = Vector((0.0, 0.0, -1.0))
    ray_dist = (ray_z - min_z) + 20.0

    hmap = np.full((res_y, res_x), np.nan, dtype=np.float64)
    miss_count = 0

    wm = context.window_manager if context else None
    if wm:
        wm.progress_begin(0, res_y)

    for row in range(res_y):
        y = max_y - row * dy     # row 0 = top = max_y
        for col in range(res_x):
            x = min_x + col * dx
            sample_hits = []

            for ox, oy in offsets:
                sx = x + (ox * dx if dx > 0 else 0.0)
                sy = y + (oy * dy if dy > 0 else 0.0)
                sx = min(max(sx, min_x), max_x)
                sy = min(max(sy, min_y), max_y)

                origin = Vector((sx, sy, ray_z))

                # True vertical projection
                loc, normal, index, dist = bvh.ray_cast(origin, down, ray_dist)
                if loc is not None:
                    sample_hits.append(loc.z)

            if sample_hits:
                hmap[row, col] = _combine_subpixel_samples(sample_hits, combine_mode)
            else:
                miss_count += 1
                
        if wm and (row % max(1, res_y // 100) == 0):
            wm.progress_update(row)
            
    if wm:
        wm.progress_end()
                
    # Handle NoData / Ray misses safely.
    if miss_count > 0:
        missing_count, iterations = _fill_missing_heights(hmap, min_z)
        log.info(
            "Heightmap raycast missed %d points. Filled %d points in %d pass(es).",
            miss_count, missing_count, iterations
        )

    return hmap


# =====================================================================
# UE Import Guide writer
# =====================================================================
def _write_ue_guide(path, *, layout, ue_settings,
                    rw_size_x, rw_size_y, rw_z_range,
                    min_z_rw, max_z_rw, geo_scale,
                    geo_info, png_name, raw_name):

    L = []
    a = L.append

    a("=" * 72)
    a("   UNREAL ENGINE 5 — LANDSCAPE IMPORT GUIDE")
    a("=" * 72)
    a(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"   Heightmap: {png_name}")
    if raw_name:
        a(f"   RAW file:  {raw_name}")
    a("")

    # --- STEP 1 ---
    a("─" * 72)
    a("   STEP 1:  Landscape Mode → New → Import from File")
    a("─" * 72)
    a("")
    a(f"   Heightmap File:    {png_name}")
    a(f"   Heightmap Size:    {ue_settings['overall_resolution_x']} x {ue_settings['overall_resolution_y']}")
    a("")

    # --- STEP 2: ALL 7 VALUES ---
    a("─" * 72)
    a("   STEP 2:  Enter these EXACT values")
    a("─" * 72)
    a("")

    w1, w2 = 32, 35
    def row(field, val):
        a(f"   ║  {field:<{w1}s}║  {val:<{w2}s}║")

    a(f"   ╔══{'═'*w1}╦══{'═'*w2}╗")
    a(f"   ║  {'Field':<{w1}s}║  {'Value':<{w2}s}║")
    a(f"   ╠══{'═'*w1}╬══{'═'*w2}╣")

    row("① World Partition Grid Size", str(ue_settings['world_partition_grid_size']))
    row("② World Partition Region Size", str(ue_settings['world_partition_region_size']))
    a(f"   ╠══{'═'*w1}╬══{'═'*w2}╣")
    row("③ Section Size", f"{ue_settings['section_size']}x{ue_settings['section_size']} Quads")
    row("④ Sections Per Component", f"{ue_settings['sections_per_component']}x{ue_settings['sections_per_component']} Section")
    row("⑤ Number of Components", f"{ue_settings['number_of_components_x']}  x  {ue_settings['number_of_components_y']}")
    a(f"   ╠══{'═'*w1}╬══{'═'*w2}╣")
    row("⑥ Overall Resolution", f"{ue_settings['overall_resolution_x']}  x  {ue_settings['overall_resolution_y']}")
    a(f"   ╠══{'═'*w1}╬══{'═'*w2}╣")
    row("⑦ Location  X", f"{ue_settings['location_x']:.1f}")
    row("   Location  Y", f"{ue_settings['location_y']:.1f}")
    row("   Location  Z", f"{ue_settings['location_z']:.1f}")
    a(f"   ╠══{'═'*w1}╬══{'═'*w2}╣")
    row("⑧ Scale  X", f"{ue_settings['scale_x']:.4f}")
    row("   Scale  Y", f"{ue_settings['scale_y']:.4f}")
    row("   Scale  Z", f"{ue_settings['scale_z']:.4f}")

    a(f"   ╚══{'═'*w1}╩══{'═'*w2}╝")
    a("")

    # --- TERRAIN INFO ---
    a("─" * 72)
    a("   TERRAIN DATA  (from Blender GIS)")
    a("─" * 72)
    a("")
    a(f"   Real-world size X:     {rw_size_x:>12.2f} m  ({rw_size_x/1000:.3f} km)")
    a(f"   Real-world size Y:     {rw_size_y:>12.2f} m  ({rw_size_y/1000:.3f} km)")
    a(f"   Min elevation:         {min_z_rw:>12.2f} m")
    a(f"   Max elevation:         {max_z_rw:>12.2f} m")
    a(f"   Height range (Z):      {rw_z_range:>12.2f} m")
    a(f"   GeoScene scale:        {geo_scale:>12.6f}")
    a(f"   Metres/pixel X:        {layout['actual_mpp_x']:>12.4f} m")
    a(f"   Metres/pixel Y:        {layout['actual_mpp_y']:>12.4f} m")
    a(f"   Vertical step:         {ue_settings['vertical_step_cm']:>12.4f} cm")
    a(f"   Vertical target:       {ue_settings['target_vertical_precision_cm']:>12.4f} cm")
    a(f"   Vertical status:       {'PASS' if ue_settings['vertical_precision_ok'] else 'FAIL'}")
    a(
        f"   Subpixel sampling:     "
        f"{ue_settings['sampling_grid']}x{ue_settings['sampling_grid']} ({ue_settings['sampling_reduce']})"
    )
    a("")

    a("   ┌─ UE World Size (computed) ─────────────────────────────┐")
    a(f"   │  X: {layout['quads_x']} quads × {ue_settings['scale_x']:.2f} cm"
      f" = {layout['quads_x']*ue_settings['scale_x']/100:.2f} m"
      f"{'':>10s}│")
    a(f"   │  Y: {layout['quads_y']} quads × {ue_settings['scale_y']:.2f} cm"
      f" = {layout['quads_y']*ue_settings['scale_y']/100:.2f} m"
      f"{'':>10s}│")
    a(f"   │  Z: {ue_settings['scale_z']:.2f}% of 512cm"
      f" = {ue_settings['scale_z']/100*512:.2f} cm"
      f" = {ue_settings['scale_z']/100*5.12:.2f} m"
      f"{'':>14s}│")
    a("   └────────────────────────────────────────────────────────┘")
    a("")

    if geo_info:
        a("─" * 72)
        a("   GEO-REFERENCE")
        a("─" * 72)
        a("")
        for k, v in geo_info.items():
            a(f"   {k}: {v}")
        a("")

    # --- HOW TO ---
    a("─" * 72)
    a("   HOW TO IMPORT")
    a("─" * 72)
    a("""
   1. Open UE5 → Landscape Mode → click "New"
   2. Click "Import from File" tab
   3. Heightmap File: Select the PNG (or .raw) file
   4. Enter ALL values from the table above EXACTLY
   5. Click "Import"

   🚨 CRITICAL TROUBLESHOOTING:
   • BLACK VOIDS / CUT-OFF TERRAIN: 
     By default, UE5 World Partition unloads terrain. 
     Open the 'World Partition' window, select cells, Right-Click → "Load".
   • SLICED / DUPLICATED TERRAIN: 
     Use the `.raw` file instead of `.png`! UE5 has an odd-pixel-width PNG bug.
   • SCALE Z ISSUES (FLAT MOUNTAINS): 
     DO NOT divide Scale Z by 100. If Scale Z is 5123.0, enter 5123.0 exactly!
     If you enter 51.23, your mountain will be squashed flat to 2 metres tall,
     destroying all natural detail.
   • SCALE X/Y: 
     Unreal may lock X and Y. Our script calculates uniform Scale X and Y.
""")
    a("")
    a("-" * 72)
    a("   UE5 NANITE / OPTIMIZATION NOTES")
    a("-" * 72)
    a("   Landscape import values above stay authoritative for terrain shape accuracy.")
    a("   Recommended runtime setup in UE5:")
    a("   1) Import Landscape from heightmap with exact scales/location.")
    a("   2) Keep Landscape for collision and gameplay.")
    a("   3) Optionally use static-mesh tile overlays with Nanite for distant detail.")
    a("   4) Combine with World Partition + HLOD for large open worlds.")
    a("=" * 72)
    a("   Generated by BlenderGIS — UE Heightmap Export")
    a("=" * 72)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    log.info("Saved UE Guide: %s", path)


def _build_ue_settings(layout, ue_scale_z, ue_loc, vertical_step_cm,
                       target_vertical_precision_cm, vertical_precision_ok,
                       sampling_grid, sampling_reduce):
    return {
        "world_partition_grid_size": 2,
        "world_partition_region_size": 16,
        "section_size": layout["section_size"],
        "sections_per_component": layout["sec_per_comp"],
        "number_of_components_x": layout["comp_x"],
        "number_of_components_y": layout["comp_y"],
        "total_components": layout["total_components"],
        "overall_resolution_x": layout["res_x"],
        "overall_resolution_y": layout["res_y"],
        "scale_x": float(layout["scale_x"]),
        "scale_y": float(layout["scale_y"]),
        "scale_z": float(ue_scale_z),
        "location_x": float(ue_loc[0]),
        "location_y": float(ue_loc[1]),
        "location_z": float(ue_loc[2]),
        "vertical_step_cm": float(vertical_step_cm),
        "target_vertical_precision_cm": float(target_vertical_precision_cm),
        "vertical_precision_ok": bool(vertical_precision_ok),
        "sampling_grid": int(sampling_grid),
        "sampling_reduce": str(sampling_reduce),
    }


def _write_export_bundle(base_path, *, hmap_u16, layout, ue_settings,
                         rw_size_x, rw_size_y, rw_z_range,
                         min_z_rw, max_z_rw, geo_scale, geo_info,
                         save_json=True, export_raw=True, extra_meta=None):
    filepath = base_path + '.png'
    png_name = os.path.basename(filepath)
    raw_name = os.path.basename(base_path + '.raw') if export_raw else None
    json_name = os.path.basename(base_path + '_ue_scale.json') if save_json else None

    try:
        from ..core.lib import imageio
        imageio.imwrite(filepath, hmap_u16)
        log.info("PNG saved (imageio): %s", filepath)
    except Exception:
        log.warning("imageio failed, fallback to Blender", exc_info=True)
        img = bpy.data.images.new("ue_hmap_tmp", width=hmap_u16.shape[1], height=hmap_u16.shape[0],
                                  alpha=False, float_buffer=True, is_data=True)
        pixels = np.zeros(hmap_u16.shape[0] * hmap_u16.shape[1] * 4, dtype=np.float32)
        flipped = np.flipud(hmap_u16).astype(np.float32) / 65535.0
        flat = flipped.ravel()
        pixels[0::4] = flat
        pixels[1::4] = flat
        pixels[2::4] = flat
        pixels[3::4] = 1.0
        img.pixels.foreach_set(pixels)
        img.file_format = 'PNG'
        img.filepath_raw = filepath
        img.colorspace_settings.name = 'Non-Color'
        img.save_render(filepath)
        bpy.data.images.remove(img)

    if export_raw:
        hmap_u16.astype('<u2', copy=False).tofile(base_path + '.raw')

    guide_path = base_path + '_UE_IMPORT_GUIDE.txt'
    _write_ue_guide(
        guide_path,
        layout=layout,
        ue_settings=ue_settings,
        rw_size_x=rw_size_x,
        rw_size_y=rw_size_y,
        rw_z_range=rw_z_range,
        min_z_rw=min_z_rw,
        max_z_rw=max_z_rw,
        geo_scale=geo_scale,
        geo_info=geo_info,
        png_name=png_name,
        raw_name=raw_name,
    )

    if save_json:
        meta = {
            "heightmap_file": png_name,
            "resolution_x": layout["res_x"],
            "resolution_y": layout["res_y"],
            "z_scale_formula_version": "v2_meters_times_100_div_512",
            "z_scale_formula": "scale_z_percent = z_range_m * 100 / 512",
            "ue_settings": {
                "world_partition_grid_size": ue_settings["world_partition_grid_size"],
                "world_partition_region_size": ue_settings["world_partition_region_size"],
                "section_size": f"{ue_settings['section_size']}x{ue_settings['section_size']}",
                "sections_per_component": f"{ue_settings['sections_per_component']}x{ue_settings['sections_per_component']}",
                "number_of_components_x": ue_settings["number_of_components_x"],
                "number_of_components_y": ue_settings["number_of_components_y"],
                "total_components": ue_settings["total_components"],
                "overall_resolution_x": ue_settings["overall_resolution_x"],
                "overall_resolution_y": ue_settings["overall_resolution_y"],
                "scale_x": round(ue_settings["scale_x"], 4),
                "scale_y": round(ue_settings["scale_y"], 4),
                "scale_z": round(ue_settings["scale_z"], 4),
                "location_x": round(ue_settings["location_x"], 2),
                "location_y": round(ue_settings["location_y"], 2),
                "location_z": round(ue_settings["location_z"], 2),
                "vertical_step_cm": round(ue_settings["vertical_step_cm"], 6),
                "target_vertical_precision_cm": round(ue_settings["target_vertical_precision_cm"], 6),
                "vertical_precision_ok": ue_settings["vertical_precision_ok"],
                "sampling_grid": ue_settings["sampling_grid"],
                "sampling_reduce": ue_settings["sampling_reduce"],
            },
            "terrain": {
                "size_x_m": round(rw_size_x, 4),
                "size_y_m": round(rw_size_y, 4),
                "min_z_m": round(min_z_rw, 4),
                "max_z_m": round(max_z_rw, 4),
                "z_range_m": round(rw_z_range, 4),
                "geo_scale": round(geo_scale, 6),
                "metres_per_pixel_x": round(layout['actual_mpp_x'], 6),
                "metres_per_pixel_y": round(layout['actual_mpp_y'], 6),
            },
            "ue_runtime_recommendations": {
                "landscape_accuracy_priority": True,
                "landscape_collision_authoritative": True,
                "optional_nanite_static_mesh_overlay": True,
                "world_partition_recommended": True,
                "hlod_recommended": True,
            },
        }
        if geo_info:
            meta["georef"] = geo_info
        if extra_meta:
            for k, v in extra_meta.items():
                meta[k] = v
        with open(base_path + '_ue_scale.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return {
        "png_name": png_name,
        "raw_name": raw_name,
        "guide_name": os.path.basename(guide_path),
        "json_name": json_name,
    }


# =====================================================================
# Operator
# =====================================================================
def _ue_cm_to_blender_m(ue_location_cm):
    # UE uses X-forward/Y-right, exporter stores (UE_X, UE_Y, UE_Z) in cm.
    # Blender world uses X-right/Y-forward in metres.
    return (
        float(ue_location_cm[1]) / 100.0,
        float(ue_location_cm[0]) / 100.0,
        float(ue_location_cm[2]) / 100.0,
    )


def _build_tile_bounds_mesh(name, size_x, size_y):
    hx = float(size_x) * 0.5
    hy = float(size_y) * 0.5
    me = bpy.data.meshes.new(name + "_mesh")
    verts = [
        (-hx, -hy, 0.0),
        (hx, -hy, 0.0),
        (hx, hy, 0.0),
        (-hx, hy, 0.0),
    ]
    faces = [(0, 1, 2, 3)]
    me.from_pydata(verts, [], faces)
    me.update()
    return me


class IMPORTGIS_OT_ue_tile_manifest(Operator, ImportHelper):
    """Import UE tile manifest and place snapped tile bounds in Blender scene"""
    bl_idname = "importgis.ue_tile_manifest"
    bl_label = "Import UE Tile Manifest"
    bl_description = "Load UE tile manifest JSON and place snapped tile bounds"
    bl_options = {"UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    use_snapped_location: BoolProperty(
        name="Use Snapped UE Location", default=True,
        description="Use snapped UE tile location when present in manifest",
    )
    create_collection: BoolProperty(
        name="Create Collection", default=True,
        description="Create a dedicated collection for imported tile bounds",
    )
    create_bounds_mesh: BoolProperty(
        name="Create Tile Bounds", default=True,
        description="Create one plane per tile showing tile footprint",
    )
    z_offset_m: FloatProperty(
        name="Z Offset (m)",
        description="Offset added to imported tile object Z location",
        default=0.0, min=-100000.0, max=100000.0, precision=3,
    )

    def execute(self, context):
        path = self.filepath
        if not path or not os.path.exists(path):
            self.report({'ERROR'}, "Tile manifest file not found")
            return {'CANCELLED'}

        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to read manifest JSON: {exc}")
            return {'CANCELLED'}

        tiles = manifest.get("tiles")
        if not isinstance(tiles, list) or not tiles:
            self.report({'ERROR'}, "Manifest has no tile entries")
            return {'CANCELLED'}

        schema = str(manifest.get("schema_version", ""))
        if schema and not schema.startswith("ue_tile_manifest_"):
            self.report({'ERROR'}, f"Unsupported manifest schema: {schema}")
            return {'CANCELLED'}

        col = None
        if self.create_collection:
            stem = os.path.splitext(os.path.basename(path))[0]
            cname = f"UE_Tiles_{stem}"
            col = bpy.data.collections.get(cname)
            if col is None:
                col = bpy.data.collections.new(cname)
                context.scene.collection.children.link(col)

        imported = 0
        for tile in tiles:
            tile_id = str(tile.get("tile_id", f"tile_{imported:03d}"))
            size_x = float(tile.get("size_x_m", 0.0))
            size_y = float(tile.get("size_y_m", 0.0))
            if self.use_snapped_location:
                ue_loc = tile.get("ue_location_cm")
            else:
                ue_loc = tile.get("ue_location_raw_cm") or tile.get("ue_location_cm")
            if not isinstance(ue_loc, dict):
                self.report({'WARNING'}, f"Skipping {tile_id}: missing ue_location_cm")
                continue
            if not math.isfinite(size_x) or not math.isfinite(size_y) or size_x <= 0 or size_y <= 0:
                self.report({'WARNING'}, f"Skipping {tile_id}: invalid tile size")
                continue

            loc_cm = (
                float(ue_loc.get("x", 0.0)),
                float(ue_loc.get("y", 0.0)),
                float(ue_loc.get("z", 0.0)),
            )
            loc_m = _ue_cm_to_blender_m(loc_cm)
            loc_m = (loc_m[0], loc_m[1], loc_m[2] + float(self.z_offset_m))

            obj_name = f"UE_TILE_{tile_id}"
            if self.create_bounds_mesh:
                me = _build_tile_bounds_mesh(obj_name, size_x, size_y)
                obj = bpy.data.objects.new(obj_name, me)
            else:
                obj = bpy.data.objects.new(obj_name, None)
                obj.empty_display_type = 'PLAIN_AXES'
                obj.empty_display_size = max(size_x, size_y) * 0.25

            obj.location = loc_m
            obj["ue_tile_id"] = tile_id
            obj["ue_grid_x"] = int(tile.get("grid_index_x", 0))
            obj["ue_grid_y"] = int(tile.get("grid_index_y", 0))
            obj["ue_size_x_m"] = float(size_x)
            obj["ue_size_y_m"] = float(size_y)

            if col is not None:
                col.objects.link(obj)
            else:
                context.scene.collection.objects.link(obj)
            imported += 1

        if imported == 0:
            self.report({'ERROR'}, "No valid tile entries were imported")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Imported {imported} tiles from manifest")
        return {'FINISHED'}


_SEC_ITEMS = [
    ('7',   '7×7 Quads',   ''),
    ('15',  '15×15 Quads',  ''),
    ('31',  '31×31 Quads',  ''),
    ('63',  '63×63 Quads',  'Recommended default'),
    ('127', '127×127 Quads', ''),
    ('255', '255×255 Quads', ''),
]

class EXPORTGIS_OT_ue_heightmap(Operator, ExportHelper):
    """Export selected mesh as a 16-bit heightmap for Unreal Engine 5"""
    bl_idname = "exportgis.ue_heightmap"
    bl_description = (
        "Export the active mesh as a 16-bit heightmap for UE5 Landscape "
        "with precise import settings"
    )
    bl_label = "Export UE Heightmap"
    bl_options = {"UNDO"}

    # ExportHelper
    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png", options={'HIDDEN'})

    target_mpp: FloatProperty(
        name="Target m/pixel",
        description=(
            "Target resolution in metres per pixel. Lower = more detail. "
            "Actual value is adjusted to fit UE layout constraints"
        ),
        default=2.0, min=0.1, max=100.0, precision=2,
    )
    target_vertical_precision_cm: FloatProperty(
        name="Target Vertical Precision (cm)",
        description=(
            "Desired vertical quantization step in centimeters. "
            "1.0 cm requires terrain Z range <= 655.35 m with 16-bit output"
        ),
        default=1.0, min=0.01, max=100.0, precision=3,
    )
    enforce_vertical_precision: BoolProperty(
        name="Require Target Vertical Precision", default=True,
        description=(
            "Cancel export when 16-bit quantization cannot satisfy the target precision"
        ),
    )
    auto_tile_export: BoolProperty(
        name="Auto Tile When Needed", default=True,
        description=(
            "Split the sampled DEM into multiple exports when a single heightmap "
            "cannot satisfy the target vertical precision"
        ),
    )
    max_tiles: IntProperty(
        name="Max Tiles",
        description="Maximum total number of tiles used for auto-tiling",
        default=64, min=1, max=64,
    )
    tile_count_x: IntProperty(
        name="Tile Count X",
        description=(
            "Force fixed tile split along X. Values > 1 enable fixed grid tiling "
            "even when precision target is already satisfied"
        ),
        default=1, min=1, max=128,
    )
    tile_count_y: IntProperty(
        name="Tile Count Y",
        description=(
            "Force fixed tile split along Y. Values > 1 enable fixed grid tiling "
            "even when precision target is already satisfied"
        ),
        default=1, min=1, max=128,
    )
    snap_tile_locations: BoolProperty(
        name="Snap Tile Locations", default=True,
        description=(
            "Snap exported UE landscape location values to a fixed centimeter grid "
            "for robust tile alignment"
        ),
    )
    snap_step_cm: FloatProperty(
        name="Snap Step (cm)",
        description="Snap increment used for exported UE tile locations",
        default=1.0, min=0.01, max=100000.0, precision=4,
    )
    save_tile_manifest: BoolProperty(
        name="Save Tile Manifest", default=True,
        description=(
            "When exporting tiles, save one manifest JSON that includes tile index, "
            "bounds, UE location and snapping offsets"
        ),
    )
    sampling_grid: EnumProperty(
        name="Subpixel Sampling",
        description="Number of vertical raycasts per output pixel axis",
        items=[
            ('1', '1x1 (Fast)', ''),
            ('2', '2x2', ''),
            ('3', '3x3 (Detailed)', ''),
            ('4', '4x4 (High Detail)', ''),
        ],
        default='3',
    )
    sampling_reduce: EnumProperty(
        name="Sampling Combine",
        description="How multiple subpixel ray hits are combined into one height value",
        items=[
            ('MEDIAN', 'Median (Robust)', ''),
            ('MEAN', 'Mean (Smooth)', ''),
            ('MAX', 'Max (Ridge Preserve)', ''),
        ],
        default='MEDIAN',
    )
    smooth_detail: BoolProperty(
        name="Smooth Mesh Detail before Export", default=True,
        description="Temporarily applies a Subdivision Surface modifier during export so rays hit smooth curved geometry instead of blocky 30m SRTM triangles. Removes flat-plane artifacts.",
    )
    include_selected_meshes: BoolProperty(
        name="Include Selected Meshes", default=True,
        description=(
            "Sample all selected mesh objects as one terrain surface. "
            "Useful when GIS tiles are split across multiple mesh objects"
        ),
    )
    section_size: EnumProperty(
        name="Section Size",
        description="UE Landscape section size in quads",
        items=_SEC_ITEMS,
        default='63',
    )
    sec_per_comp: EnumProperty(
        name="Sections / Component",
        description="Number of sections per landscape component",
        items=[
            ('1', '1×1 Section', ''),
            ('2', '2×2 Sections', ''),
        ],
        default='1',
    )
    export_raw: BoolProperty(
        name="Export .raw (Gaea / UE5 Safe)", default=True,
        description="Save raw 16-bit LE file. Bypasses UE5's PNG dimension bugs",
    )
    export_fbx: BoolProperty(
        name="Also export .fbx Mesh (Reference)", default=False,
        description=(
            "Exports static mesh reference (or tile overlays) for Unreal Engine. "
            "Can be used as optional Nanite detail overlay above Landscape"
        ),
    )
    save_json: BoolProperty(
        name="Save .json metadata", default=True,
        description="Save machine-readable JSON",
    )

    # Poll
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (context.mode == 'OBJECT'
                and obj is not None and obj.type == 'MESH')

    # Draw
    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'target_mpp')
        layout.prop(self, 'target_vertical_precision_cm')
        layout.prop(self, 'enforce_vertical_precision')
        layout.prop(self, 'auto_tile_export')
        if self.auto_tile_export:
            layout.prop(self, 'max_tiles')
        layout.prop(self, 'tile_count_x')
        layout.prop(self, 'tile_count_y')
        layout.prop(self, 'snap_tile_locations')
        if self.snap_tile_locations:
            layout.prop(self, 'snap_step_cm')
        if self.tile_count_x > 1 or self.tile_count_y > 1 or self.auto_tile_export:
            layout.prop(self, 'save_tile_manifest')
        layout.prop(self, 'sampling_grid')
        layout.prop(self, 'sampling_reduce')
        layout.prop(self, 'smooth_detail')
        layout.prop(self, 'include_selected_meshes')
        layout.separator()
        layout.label(text="UE Landscape Setup:")
        layout.prop(self, 'section_size')
        layout.prop(self, 'sec_per_comp')
        layout.separator()
        layout.prop(self, 'export_raw')
        layout.prop(self, 'export_fbx')
        layout.prop(self, 'save_json')

    # Execute
    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh selected")
            return {'CANCELLED'}

        geoscn = GeoScene(context.scene)
        geo_scale = geoscn.scale if geoscn.hasScale else 1.0
        if not math.isfinite(geo_scale) or geo_scale <= 0:
            self.report({'ERROR'}, f"Invalid GeoScene scale: {geo_scale}")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        temp_modifiers = []
        eval_objects = []
        bm = None

        min_x = max_x = min_y = max_y = min_z = max_z = 0.0
        rw_size_x = rw_size_y = rw_z_range = min_z_rw = max_z_rw = 0.0
        layout = None
        res_x = res_y = 0
        hmap = None
        vertical_step_cm = 0.0
        target_vertical_precision_cm = float(self.target_vertical_precision_cm)
        if target_vertical_precision_cm <= 0 or not math.isfinite(target_vertical_precision_cm):
            self.report({'ERROR'}, "Target vertical precision must be a positive finite value")
            return {'CANCELLED'}
        snap_step_cm = float(self.snap_step_cm)
        if self.snap_tile_locations and (snap_step_cm <= 0 or not math.isfinite(snap_step_cm)):
            self.report({'ERROR'}, "Snap Step (cm) must be a positive finite value")
            return {'CANCELLED'}
        max_z_range_for_target = _max_z_range_for_precision(target_vertical_precision_cm)
        vertical_precision_ok = True
        sample_grid = int(self.sampling_grid)
        sample_reduce = self.sampling_reduce

        try:
            source_objects = [obj]
            if self.include_selected_meshes:
                selected_meshes = [o for o in context.selected_objects if o.type == 'MESH']
                if selected_meshes:
                    source_objects = selected_meshes

            if self.smooth_detail:
                # Add temporary subdivision so vertical raycasts hit smoothed terrain
                for src_obj in source_objects:
                    mod = src_obj.modifiers.new(name="BGIS_TMP_SUBSURF", type='SUBSURF')
                    mod.subdivision_type = 'CATMULL_CLARK'
                    mod.levels = 3
                    mod.render_levels = 3
                    temp_modifiers.append((src_obj, mod.name))
                depsgraph.update()

            bm = bmesh.new()
            for src_obj in source_objects:
                src_eval = src_obj.evaluated_get(depsgraph)
                eval_objects.append(src_eval)
                mesh_eval = src_eval.to_mesh()
                if mesh_eval is None:
                    continue
                tmp_mesh = mesh_eval.copy()
                try:
                    tmp_mesh.transform(src_obj.matrix_world)
                    bm.from_mesh(tmp_mesh)
                finally:
                    bpy.data.meshes.remove(tmp_mesh)

            if len(bm.verts) == 0:
                self.report({'ERROR'}, "Mesh has no vertices")
                return {'CANCELLED'}

            xs = [v.co.x for v in bm.verts]
            ys = [v.co.y for v in bm.verts]
            zs = [v.co.z for v in bm.verts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            min_z, max_z = min(zs), max(zs)

            rw_size_x = (max_x - min_x) * geo_scale
            rw_size_y = (max_y - min_y) * geo_scale
            rw_z_range = (max_z - min_z) * geo_scale
            min_z_rw = min_z * geo_scale
            max_z_rw = max_z * geo_scale

            if (not math.isfinite(rw_size_x) or rw_size_x <= 0 or
                not math.isfinite(rw_size_y) or rw_size_y <= 0):
                self.report({'ERROR'}, "Terrain size is invalid for UE export")
                return {'CANCELLED'}
            if not math.isfinite(rw_z_range) or rw_z_range < 0:
                self.report({'ERROR'}, "Terrain Z range is invalid for UE export")
                return {'CANCELLED'}
            vertical_step_cm = _calc_vertical_precision_cm(rw_z_range)
            vertical_precision_ok = vertical_step_cm <= (target_vertical_precision_cm + _LAYOUT_EPSILON)
            if not vertical_precision_ok and not self.auto_tile_export:
                precision_msg = (
                    f"Vertical precision {vertical_step_cm:.4f} cm exceeds target "
                    f"{target_vertical_precision_cm:.4f} cm. "
                    f"Current Z range={rw_z_range:.3f} m, target requires <= {max_z_range_for_target:.3f} m."
                )
                if self.enforce_vertical_precision:
                    self.report({'ERROR'}, precision_msg)
                    return {'CANCELLED'}
                self.report({'WARNING'}, precision_msg)

            sec_sz = int(self.section_size)
            spc = int(self.sec_per_comp)
            layout = _calc_ue_layout(rw_size_x, rw_size_y, self.target_mpp, sec_sz, spc)
            layout_errors = _validate_layout(layout, self.target_mpp)
            if layout_errors:
                for err in layout_errors:
                    log.error("UE layout validation failed: %s", err)
                self.report({'ERROR'}, "UE layout validation failed. Check logs for details.")
                return {'CANCELLED'}

            res_x = layout['res_x']
            res_y = layout['res_y']
            raycast_samples = res_x * res_y * sample_grid * sample_grid
            if raycast_samples > _MAX_RAYCAST_SAMPLES:
                self.report(
                    {'ERROR'},
                    (
                        f"Sampling workload too large ({raycast_samples:,} raycasts). "
                        f"Increase Target m/pixel or lower Subpixel Sampling."
                    ),
                )
                return {'CANCELLED'}
            self.report(
                {'INFO'},
                f"Sampling {res_x}x{res_y} heightmap | terrain {rw_size_x:.0f}x{rw_size_y:.0f} m | "
                f"subpixel {sample_grid}x{sample_grid} ({sample_reduce}) | sources {len(source_objects)}"
            )

            bvh = BVHTree.FromBMesh(bm)
            hmap = _sample_heights(
                bvh, min_x, max_x, min_y, max_y, min_z, max_z,
                res_x, res_y, sample_grid=sample_grid, combine_mode=sample_reduce, context=context
            )
        except Exception as exc:
            log.exception("UE heightmap export failed")
            self.report({'ERROR'}, f"UE heightmap export failed: {exc}")
            return {'CANCELLED'}
        finally:
            if bm is not None:
                bm.free()
            for eval_obj in eval_objects:
                try:
                    eval_obj.to_mesh_clear()
                except Exception:
                    log.warning("Failed to clear evaluated mesh", exc_info=True)
            for src_obj, mod_name in temp_modifiers:
                try:
                    mod = src_obj.modifiers.get(mod_name)
                    if mod is not None:
                        src_obj.modifiers.remove(mod)
                except Exception:
                    log.warning("Failed to remove temporary modifier", exc_info=True)

        hmap_u16 = _normalize_heightmap_u16(hmap, min_z, max_z)

        # Scale Z %: 100% equals 512 cm (5.12 m) vertical range in UE landscape import.
        ue_scale_z = _calc_ue_scale_z(rw_z_range)

        # Maintain world offset. Blender (X=Right, Y=Forward) -> UE (X=Forward, Y=Right).
        cen_x_cm = ((min_x + max_x) / 2.0) * geo_scale * 100.0
        cen_y_cm = ((min_y + max_y) / 2.0) * geo_scale * 100.0
        cen_z_cm = ((min_z_rw + max_z_rw) / 2.0) * 100.0
        ue_loc_raw = (cen_y_cm, cen_x_cm, cen_z_cm)
        ue_loc = _snap_location_cm(ue_loc_raw, snap_step_cm) if self.snap_tile_locations else ue_loc_raw
        ue_settings = _build_ue_settings(
            layout, ue_scale_z, ue_loc, vertical_step_cm,
            target_vertical_precision_cm, vertical_precision_ok,
            sample_grid, sample_reduce
        )

        # --- Geo info -----------------------------------------------------
        geo_info = {}
        if geoscn.isGeoref:
            try:
                sw = geoscn.view3dToProj(min_x, min_y)
                ne = geoscn.view3dToProj(max_x, max_y)
                geo_info["CRS"] = geoscn.crs
                geo_info["SW corner (CRS)"] = f"({sw[0]:.2f}, {sw[1]:.2f})"
                geo_info["NE corner (CRS)"] = f"({ne[0]:.2f}, {ne[1]:.2f})"
                if geoscn.hasOriginGeo:
                    ll = geoscn.getOriginGeo()
                    geo_info["Scene origin (lon, lat)"] = f"({ll[0]:.6f}, {ll[1]:.6f})"
            except Exception as e:
                log.warning("Geo corners failed: %s", e)

        # --- File paths ---------------------------------------------------
        filepath = self.filepath
        if not filepath.lower().endswith('.png'):
            filepath += '.png'
        base = os.path.splitext(filepath)[0]
        generated_files = []
        tile_report = None
        tile_entries = []
        dx = (max_x - min_x) / (res_x - 1) if res_x > 1 else 0.0
        dy = (max_y - min_y) / (res_y - 1) if res_y > 1 else 0.0
        manual_tile_tx = max(1, int(self.tile_count_x))
        manual_tile_ty = max(1, int(self.tile_count_y))
        manual_tile_total = manual_tile_tx * manual_tile_ty
        if manual_tile_total > 4096:
            self.report(
                {'ERROR'},
                f"Tile grid {manual_tile_tx}x{manual_tile_ty} is too large ({manual_tile_total} tiles, max 4096)"
            )
            return {'CANCELLED'}
        tile_mode = "single"
        tile_tx = 1
        tile_ty = 1
        tile_worst = vertical_step_cm

        if manual_tile_tx > 1 or manual_tile_ty > 1:
            tile_mode = "fixed_grid"
            tile_tx = manual_tile_tx
            tile_ty = manual_tile_ty
        elif (not vertical_precision_ok) and self.auto_tile_export:
            tile_mode = "auto_precision"
            try:
                tile_tx, tile_ty, tile_worst = _choose_tile_grid_for_precision(
                    hmap, target_vertical_precision_cm, self.max_tiles
                )
            except RuntimeError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}

        if tile_mode == "single":
            single_extra_meta = {
                "source_mesh_count": int(len(source_objects)),
                "snap": {
                    "enabled": bool(self.snap_tile_locations),
                    "step_cm": float(snap_step_cm),
                    "raw_location_cm": {
                        "x": round(ue_loc_raw[0], 6),
                        "y": round(ue_loc_raw[1], 6),
                        "z": round(ue_loc_raw[2], 6),
                    },
                    "snapped_location_cm": {
                        "x": round(ue_loc[0], 6),
                        "y": round(ue_loc[1], 6),
                        "z": round(ue_loc[2], 6),
                    },
                }
            }
            bundle = _write_export_bundle(
                base,
                hmap_u16=hmap_u16,
                layout=layout,
                ue_settings=ue_settings,
                rw_size_x=rw_size_x,
                rw_size_y=rw_size_y,
                rw_z_range=rw_z_range,
                min_z_rw=min_z_rw,
                max_z_rw=max_z_rw,
                geo_scale=geo_scale,
                geo_info=geo_info,
                save_json=self.save_json,
                export_raw=self.export_raw,
                extra_meta=single_extra_meta,
            )
            generated_files.extend(filter(None, [
                bundle["png_name"],
                bundle["raw_name"],
                bundle["guide_name"],
                bundle["json_name"],
            ]))
        else:
            if tile_tx >= res_x or tile_ty >= res_y:
                self.report(
                    {'ERROR'},
                    f"Tile grid {tile_tx}x{tile_ty} is too dense for resolution {res_x}x{res_y}"
                )
                return {'CANCELLED'}
            row_slices = _tile_slices(res_y, tile_ty)
            col_slices = _tile_slices(res_x, tile_tx)
            tile_worst_actual = 0.0

            for yi, rs in enumerate(row_slices):
                for xi, cs in enumerate(col_slices):
                    tile = hmap[rs[0]:rs[1], cs[0]:cs[1]]
                    if tile.shape[0] < 2 or tile.shape[1] < 2:
                        raise RuntimeError(
                            f"Auto-tiling produced an invalid tile size {tile.shape[1]}x{tile.shape[0]}"
                        )

                    tile_min = float(np.nanmin(tile))
                    tile_max = float(np.nanmax(tile))
                    tile_rw_z_range = (tile_max - tile_min) * geo_scale
                    tile_min_z_rw = tile_min * geo_scale
                    tile_max_z_rw = tile_max * geo_scale
                    tile_step_cm = _calc_vertical_precision_cm(tile_rw_z_range)
                    tile_worst_actual = max(tile_worst_actual, tile_step_cm)
                    if tile_step_cm > target_vertical_precision_cm + _LAYOUT_EPSILON and self.enforce_vertical_precision:
                        raise RuntimeError(
                            f"Tile {yi:02d}_{xi:02d} still exceeds target precision: {tile_step_cm:.4f} cm"
                        )

                    tile_size_x = (cs[1] - cs[0] - 1) * dx * geo_scale
                    tile_size_y = (rs[1] - rs[0] - 1) * dy * geo_scale
                    tile_layout = _calc_ue_layout(tile_size_x, tile_size_y, self.target_mpp, sec_sz, spc)
                    tile_layout_errors = _validate_layout(tile_layout, self.target_mpp)
                    if tile_layout_errors:
                        raise RuntimeError(
                            f"Tile {yi:02d}_{xi:02d} UE layout validation failed: {tile_layout_errors[0]}"
                        )
                    tile_ue_scale_z = _calc_ue_scale_z(tile_rw_z_range)

                    tile_min_x = min_x + cs[0] * dx
                    tile_max_x = min_x + (cs[1] - 1) * dx
                    tile_max_y = max_y - rs[0] * dy
                    tile_min_y = max_y - (rs[1] - 1) * dy
                    tile_cen_x_cm = ((tile_min_x + tile_max_x) / 2.0) * geo_scale * 100.0
                    tile_cen_y_cm = ((tile_min_y + tile_max_y) / 2.0) * geo_scale * 100.0
                    tile_cen_z_cm = ((tile_min_z_rw + tile_max_z_rw) / 2.0) * 100.0
                    tile_ue_loc_raw = (tile_cen_y_cm, tile_cen_x_cm, tile_cen_z_cm)
                    tile_ue_loc = (
                        _snap_location_cm(tile_ue_loc_raw, snap_step_cm)
                        if self.snap_tile_locations else tile_ue_loc_raw
                    )
                    tile_snap_delta = (
                        tile_ue_loc[0] - tile_ue_loc_raw[0],
                        tile_ue_loc[1] - tile_ue_loc_raw[1],
                        tile_ue_loc[2] - tile_ue_loc_raw[2],
                    )
                    tile_ue_settings = _build_ue_settings(
                        tile_layout,
                        tile_ue_scale_z,
                        tile_ue_loc,
                        tile_step_cm,
                        target_vertical_precision_cm,
                        tile_step_cm <= (target_vertical_precision_cm + _LAYOUT_EPSILON),
                        sample_grid,
                        sample_reduce,
                    )

                    tile_geo_info = {}
                    if geoscn.isGeoref:
                        try:
                            sw = geoscn.view3dToProj(tile_min_x, tile_min_y)
                            ne = geoscn.view3dToProj(tile_max_x, tile_max_y)
                            tile_geo_info["CRS"] = geoscn.crs
                            tile_geo_info["SW corner (CRS)"] = f"({sw[0]:.2f}, {sw[1]:.2f})"
                            tile_geo_info["NE corner (CRS)"] = f"({ne[0]:.2f}, {ne[1]:.2f})"
                            if geoscn.hasOriginGeo:
                                ll = geoscn.getOriginGeo()
                                tile_geo_info["Scene origin (lon, lat)"] = f"({ll[0]:.6f}, {ll[1]:.6f})"
                        except Exception as e:
                            log.warning("Tile geo corners failed: %s", e)

                    tile_base = f"{base}_tile_{yi:02d}_{xi:02d}"
                    tile_extra_meta = {
                        "tile": {
                            "mode": tile_mode,
                            "grid_index_x": int(xi),
                            "grid_index_y": int(yi),
                            "grid_count_x": int(tile_tx),
                            "grid_count_y": int(tile_ty),
                            "neighbor_indices": {
                                "left": [yi, xi - 1] if xi > 0 else None,
                                "right": [yi, xi + 1] if xi < tile_tx - 1 else None,
                                "up": [yi - 1, xi] if yi > 0 else None,
                                "down": [yi + 1, xi] if yi < tile_ty - 1 else None,
                            },
                            "scene_bounds": {
                                "min_x": round(tile_min_x, 6),
                                "max_x": round(tile_max_x, 6),
                                "min_y": round(tile_min_y, 6),
                                "max_y": round(tile_max_y, 6),
                            },
                        },
                        "snap": {
                            "enabled": bool(self.snap_tile_locations),
                            "step_cm": float(snap_step_cm),
                            "raw_location_cm": {
                                "x": round(tile_ue_loc_raw[0], 6),
                                "y": round(tile_ue_loc_raw[1], 6),
                                "z": round(tile_ue_loc_raw[2], 6),
                            },
                            "snapped_location_cm": {
                                "x": round(tile_ue_loc[0], 6),
                                "y": round(tile_ue_loc[1], 6),
                                "z": round(tile_ue_loc[2], 6),
                            },
                            "delta_cm": {
                                "x": round(tile_snap_delta[0], 6),
                                "y": round(tile_snap_delta[1], 6),
                                "z": round(tile_snap_delta[2], 6),
                            },
                        },
                    }
                    tile_bundle = _write_export_bundle(
                        tile_base,
                        hmap_u16=_normalize_heightmap_u16(tile, tile_min, tile_max),
                        layout=tile_layout,
                        ue_settings=tile_ue_settings,
                        rw_size_x=tile_size_x,
                        rw_size_y=tile_size_y,
                        rw_z_range=tile_rw_z_range,
                        min_z_rw=tile_min_z_rw,
                        max_z_rw=tile_max_z_rw,
                        geo_scale=geo_scale,
                        geo_info=tile_geo_info,
                        save_json=self.save_json,
                        export_raw=self.export_raw,
                        extra_meta=tile_extra_meta,
                    )
                    generated_files.extend(filter(None, [
                        tile_bundle["png_name"],
                        tile_bundle["raw_name"],
                        tile_bundle["guide_name"],
                        tile_bundle["json_name"],
                    ]))
                    tile_entries.append({
                        "tile_id": f"{yi:02d}_{xi:02d}",
                        "grid_index_x": int(xi),
                        "grid_index_y": int(yi),
                        "heightmap_file": tile_bundle["png_name"],
                        "raw_file": tile_bundle["raw_name"],
                        "guide_file": tile_bundle["guide_name"],
                        "json_file": tile_bundle["json_name"],
                        "resolution_x": int(tile_layout["res_x"]),
                        "resolution_y": int(tile_layout["res_y"]),
                        "size_x_m": round(tile_size_x, 6),
                        "size_y_m": round(tile_size_y, 6),
                        "min_z_m": round(tile_min_z_rw, 6),
                        "max_z_m": round(tile_max_z_rw, 6),
                        "z_range_m": round(tile_rw_z_range, 6),
                        "vertical_step_cm": round(tile_step_cm, 6),
                        "ue_location_cm": {
                            "x": round(tile_ue_loc[0], 6),
                            "y": round(tile_ue_loc[1], 6),
                            "z": round(tile_ue_loc[2], 6),
                        },
                        "ue_location_raw_cm": {
                            "x": round(tile_ue_loc_raw[0], 6),
                            "y": round(tile_ue_loc_raw[1], 6),
                            "z": round(tile_ue_loc_raw[2], 6),
                        },
                        "ue_scale": {
                            "x": round(tile_ue_settings["scale_x"], 6),
                            "y": round(tile_ue_settings["scale_y"], 6),
                            "z": round(tile_ue_settings["scale_z"], 6),
                        },
                    })

            tile_worst = tile_worst_actual
            if tile_mode == "auto_precision":
                tile_report = f"Auto-tiled {tile_tx}x{tile_ty} | worst step {tile_worst:.3f} cm"
                self.report(
                    {'INFO'},
                    f"Auto-tiled export used {tile_tx}x{tile_ty} tiles to satisfy {target_vertical_precision_cm:.2f} cm target"
                )
            else:
                tile_report = f"Fixed tiles {tile_tx}x{tile_ty} | worst step {tile_worst:.3f} cm"
                self.report(
                    {'INFO'},
                    f"Fixed tiled export {tile_tx}x{tile_ty} completed | worst step {tile_worst:.3f} cm"
                )

            if self.save_tile_manifest and tile_entries:
                manifest = {
                    "schema_version": "ue_tile_manifest_v1",
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "source_base": os.path.basename(base),
                    "source_mesh_count": int(len(source_objects)),
                    "tile_mode": tile_mode,
                    "grid_count_x": int(tile_tx),
                    "grid_count_y": int(tile_ty),
                    "target_vertical_precision_cm": float(target_vertical_precision_cm),
                    "worst_vertical_step_cm": float(tile_worst),
                    "snap": {
                        "enabled": bool(self.snap_tile_locations),
                        "step_cm": float(snap_step_cm),
                    },
                    "global": {
                        "size_x_m": round(rw_size_x, 6),
                        "size_y_m": round(rw_size_y, 6),
                        "min_z_m": round(min_z_rw, 6),
                        "max_z_m": round(max_z_rw, 6),
                        "z_range_m": round(rw_z_range, 6),
                    },
                    "tiles": tile_entries,
                }
                manifest_path = base + "_tile_manifest.json"
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                generated_files.append(os.path.basename(manifest_path))

        # --- Save FBX (Static Mesh) ---------------------------------------
        fbx_name = None
        if getattr(self, 'export_fbx', False):
            fbx_path = base + '.fbx'
            try:
                bpy.ops.export_scene.fbx(
                    filepath=fbx_path,
                    use_selection=True,
                    global_scale=geo_scale,
                )
                log.info("FBX saved: %s", fbx_path)
                fbx_name = os.path.basename(fbx_path)
                generated_files.append(fbx_name)
            except Exception:
                log.warning("FBX export failed", exc_info=True)
                self.report({'WARNING'}, "FBX Export Failed, check console")

        # --- Done ---------------------------------------------------------
        if tile_report is not None:
            display_files = generated_files[:8]
            extra = len(generated_files) - len(display_files)
            if extra > 0:
                display_files.append(f"...(+{extra} more)")
            self.report(
                {'INFO'},
                f"UE export complete | {tile_report} | Files: {', '.join(display_files)}"
            )
        else:
            self.report(
                {'INFO'},
                f"UE export complete | {res_x}x{res_y} | "
                f"Scale XY {ue_settings['scale_x']:.2f}/{ue_settings['scale_y']:.2f} cm | "
                f"Scale Z {ue_settings['scale_z']:.2f}% ({ue_settings['vertical_step_cm']:.3f} cm step) | "
                f"Sampling {ue_settings['sampling_grid']}x{ue_settings['sampling_grid']} {ue_settings['sampling_reduce']} | "
                f"Components {ue_settings['number_of_components_x']}x{ue_settings['number_of_components_y']} | "
                f"Files: {', '.join(generated_files)}"
            )
        return {'FINISHED'}

    # --- Blender fallback PNG writer ---
    @staticmethod
    def _save_png_blender(hmap_u16, filepath, w, h):
        img = bpy.data.images.new("ue_hmap_tmp", width=w, height=h,
                                  alpha=False, float_buffer=True, is_data=True)
        pixels = np.zeros(w * h * 4, dtype=np.float32)
        flipped = np.flipud(hmap_u16).astype(np.float32) / 65535.0
        flat = flipped.ravel()
        pixels[0::4] = flat
        pixels[1::4] = flat
        pixels[2::4] = flat
        pixels[3::4] = 1.0
        img.pixels.foreach_set(pixels)
        img.file_format = 'PNG'
        img.filepath_raw = filepath
        img.colorspace_settings.name = 'Non-Color'
        img.save_render(filepath)
        bpy.data.images.remove(img)


# =====================================================================
# Registration
# =====================================================================
def register():
    for cls in (IMPORTGIS_OT_ue_tile_manifest, EXPORTGIS_OT_ue_heightmap):
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            log.warning('Re-registering %s', cls)
            try:
                bpy.utils.unregister_class(cls)
            except Exception:
                pass
            bpy.utils.register_class(cls)

def unregister():
    for cls in (EXPORTGIS_OT_ue_heightmap, IMPORTGIS_OT_ue_tile_manifest):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
