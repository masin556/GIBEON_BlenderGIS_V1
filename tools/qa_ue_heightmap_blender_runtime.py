import json
import math
import shutil
import sys
from pathlib import Path

import bpy
import bmesh

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

import BlenderGIS
from BlenderGIS.operators import io_export_ue_heightmap

OUT_DIR = ROOT / "tools" / "qa_out"
BASE = OUT_DIR / "qa_runtime_heightmap"
BASE_TILED = OUT_DIR / "qa_runtime_heightmap_tiled"
BASE_MULTI = OUT_DIR / "qa_runtime_heightmap_multi"
BASE_FAIL = OUT_DIR / "qa_runtime_heightmap_fail"


def clean_outputs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for base in [BASE, BASE_TILED, BASE_MULTI, BASE_FAIL]:
        for suffix in [
            ".png",
            ".raw",
            ".fbx",
            "_UE_IMPORT_GUIDE.txt",
            "_ue_scale.json",
            "_tile_manifest.json",
        ]:
            p = Path(str(base) + suffix)
            if p.exists():
                p.unlink()
        for p in OUT_DIR.glob(base.name + "_tile_*"):
            if p.is_file():
                p.unlink()


def build_test_mesh():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=180,
        y_subdivisions=220,
        size=100.0,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    obj.name = "qa_dem_mesh"

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for v in bm.verts:
        x = v.co.x
        y = v.co.y
        # Z range ~ 300 m -> should satisfy 1cm target.
        v.co.z = (
            80.0 * math.sin(x * 0.04)
            + 95.0 * math.cos(y * 0.035)
            + 0.005 * x * y
        )
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def run_export(
    obj,
    base_path: Path,
    enforce_precision=True,
    max_tiles=64,
    tile_count_x=1,
    tile_count_y=1,
    snap_step_cm=1.0,
    save_tile_manifest=True,
    include_selected_meshes=True,
    selected_objs=None,
):
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    targets = selected_objs if selected_objs else [obj]
    for o in targets:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        result = bpy.ops.exportgis.ue_heightmap(
            'EXEC_DEFAULT',
            filepath=str(base_path) + ".png",
            target_mpp=1.5,
            target_vertical_precision_cm=1.0,
            enforce_vertical_precision=enforce_precision,
            max_tiles=max_tiles,
            tile_count_x=tile_count_x,
            tile_count_y=tile_count_y,
            snap_tile_locations=True,
            snap_step_cm=snap_step_cm,
            save_tile_manifest=save_tile_manifest,
            include_selected_meshes=include_selected_meshes,
            sampling_grid='3',
            sampling_reduce='MEDIAN',
            smooth_detail=False,
            section_size='63',
            sec_per_comp='1',
            export_raw=True,
            export_fbx=False,
            save_json=True,
        )
        return result, None
    except RuntimeError as exc:
        return {'CANCELLED'}, str(exc)


def validate_outputs():
    png_path = Path(str(BASE) + ".png")
    raw_path = Path(str(BASE) + ".raw")
    guide_path = Path(str(BASE) + "_UE_IMPORT_GUIDE.txt")
    json_path = Path(str(BASE) + "_ue_scale.json")

    for p in [png_path, raw_path, guide_path, json_path]:
        if not p.exists():
            raise RuntimeError(f"Missing output file: {p}")

    meta = json.loads(json_path.read_text(encoding='utf-8'))
    ue = meta["ue_settings"]
    terrain = meta["terrain"]

    if not ue.get("vertical_precision_ok", False):
        raise RuntimeError("vertical_precision_ok is false")
    if ue.get("vertical_step_cm", 999) > 1.0 + 1e-6:
        raise RuntimeError(f"vertical_step_cm too high: {ue.get('vertical_step_cm')}")

    res_x = int(meta["resolution_x"])
    res_y = int(meta["resolution_y"])
    expected_raw_size = res_x * res_y * 2
    actual_raw_size = raw_path.stat().st_size
    if actual_raw_size != expected_raw_size:
        raise RuntimeError(f"RAW size mismatch: got {actual_raw_size}, expected {expected_raw_size}")

    print("qa_ue_heightmap_blender_runtime: PASS")
    print(f"  resolution: {res_x}x{res_y}")
    print(f"  z_range_m: {terrain['z_range_m']}")
    print(f"  vertical_step_cm: {ue['vertical_step_cm']}")


def validate_multi_outputs():
    json_path = Path(str(BASE_MULTI) + "_ue_scale.json")
    if not json_path.exists():
        raise RuntimeError(f"Missing multi-source metadata file: {json_path}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if int(meta.get("source_mesh_count", 0)) < 2:
        raise RuntimeError("Multi-source export did not record source_mesh_count >= 2")


def scale_mesh_height(obj, factor: float):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for v in bm.verts:
        v.co.z *= factor
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def validate_precision_gate_fail():
    json_path = Path(str(BASE_FAIL) + "_ue_scale.json")
    if json_path.exists():
        raise RuntimeError("precision fail case should not create JSON output")


def validate_tiled_outputs(grid_x: int, grid_y: int, snap_step_cm: float):
    manifest_path = Path(str(BASE_TILED) + "_tile_manifest.json")
    if not manifest_path.exists():
        raise RuntimeError(f"Missing tile manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tile_mode") != "fixed_grid":
        raise RuntimeError(f"Unexpected tile_mode: {manifest.get('tile_mode')}")
    if int(manifest.get("grid_count_x", 0)) != grid_x or int(manifest.get("grid_count_y", 0)) != grid_y:
        raise RuntimeError("Manifest grid counts mismatch")

    tiles = manifest.get("tiles", [])
    if len(tiles) != grid_x * grid_y:
        raise RuntimeError(f"Tile count mismatch: got {len(tiles)}, expected {grid_x * grid_y}")

    for tile in tiles:
        png = OUT_DIR / tile["heightmap_file"]
        raw = OUT_DIR / tile["raw_file"]
        guide = OUT_DIR / tile["guide_file"]
        jmeta = OUT_DIR / tile["json_file"]
        for p in [png, raw, guide, jmeta]:
            if not p.exists():
                raise RuntimeError(f"Missing tile output: {p}")

        location = tile["ue_location_cm"]
        for axis in ("x", "y", "z"):
            snapped = float(location[axis])
            nearest = round(snapped / snap_step_cm) * snap_step_cm
            if abs(snapped - nearest) > 1e-6:
                raise RuntimeError(f"Tile {tile['tile_id']} location {axis} not snapped: {snapped}")

        meta = json.loads(jmeta.read_text(encoding="utf-8"))
        if "tile" not in meta or "snap" not in meta:
            raise RuntimeError(f"Tile metadata missing tile/snap keys: {jmeta}")


def validate_manifest_import(grid_x: int, grid_y: int):
    manifest_path = Path(str(BASE_TILED) + "_tile_manifest.json")
    if not manifest_path.exists():
        raise RuntimeError("Tile manifest not found for import test")

    result = bpy.ops.importgis.ue_tile_manifest(
        'EXEC_DEFAULT',
        filepath=str(manifest_path),
        create_collection=True,
        create_bounds_mesh=True,
        use_snapped_location=True,
        z_offset_m=0.0,
    )
    if result != {'FINISHED'}:
        raise RuntimeError(f"Manifest import failed: {result}")

    expected = grid_x * grid_y
    objs = [o for o in bpy.data.objects if o.name.startswith("UE_TILE_")]
    if len(objs) < expected:
        raise RuntimeError(f"Imported tile objects mismatch: got {len(objs)}, expected >= {expected}")


def main():
    clean_outputs()
    io_export_ue_heightmap.register()
    try:
        obj = build_test_mesh()
        result, err = run_export(obj, BASE, enforce_precision=True)
        print(f"operator_result: {result}")
        if err is not None:
            raise RuntimeError(f"Unexpected export error: {err}")
        validate_outputs()

        # Multi-source export: two selected meshes should be sampled together.
        obj2 = obj.copy()
        obj2.data = obj.data.copy()
        obj2.name = "qa_dem_mesh_copy"
        obj2.location.x += 95.0
        bpy.context.scene.collection.objects.link(obj2)
        result_multi, err_multi = run_export(
            obj,
            BASE_MULTI,
            enforce_precision=True,
            include_selected_meshes=True,
            selected_objs=[obj, obj2],
        )
        print(f"operator_result_multi: {result_multi}")
        if err_multi is not None:
            raise RuntimeError(f"Unexpected multi-source export error: {err_multi}")
        validate_multi_outputs()

        # Fixed-grid tiling + snap + manifest path.
        result_tiled, err_tiled = run_export(
            obj,
            BASE_TILED,
            enforce_precision=True,
            max_tiles=64,
            tile_count_x=2,
            tile_count_y=2,
            snap_step_cm=100.0,
            save_tile_manifest=True,
        )
        print(f"operator_result_tiled: {result_tiled}")
        if err_tiled is not None:
            raise RuntimeError(f"Unexpected tiled export error: {err_tiled}")
        validate_tiled_outputs(grid_x=2, grid_y=2, snap_step_cm=100.0)
        validate_manifest_import(grid_x=2, grid_y=2)

        # Precision-gate failure case: force large Z range and expect cancellation.
        scale_mesh_height(obj, 8.0)
        result_fail, fail_err = run_export(obj, BASE_FAIL, enforce_precision=True, max_tiles=4)
        print(f"operator_result_fail_case: {result_fail}")
        if result_fail != {'CANCELLED'}:
            raise RuntimeError(f"Expected CANCELLED for precision gate, got: {result_fail}")
        if fail_err is not None and (
            "Vertical precision" not in fail_err and "Could not satisfy" not in fail_err
        ):
            raise RuntimeError(f"Expected precision-related error message, got: {fail_err}")
        validate_precision_gate_fail()
    finally:
        io_export_ue_heightmap.unregister()


if __name__ == "__main__":
    main()
