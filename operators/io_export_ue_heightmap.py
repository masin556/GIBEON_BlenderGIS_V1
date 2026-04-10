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

from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, EnumProperty, IntProperty, BoolProperty, FloatProperty
from bpy.types import Operator

from ..geoscene import GeoScene

log = logging.getLogger(__name__)

# =====================================================================
# UE Landscape layout calculator
# =====================================================================
_SECTION_SIZES = [7, 15, 31, 63, 127, 255]

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
    comp_quads = section_size * sec_per_comp  # quads per component

    # Components per axis = ceil(terrain_quads / comp_quads)
    target_quads_x = rw_size_x_m / target_m_per_px
    target_quads_y = rw_size_y_m / target_m_per_px

    comp_x = max(1, round(target_quads_x / comp_quads))
    comp_y = max(1, round(target_quads_y / comp_quads))

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


# =====================================================================
# Height sampling — non-square
# =====================================================================
def _sample_heights(bvh, min_x, max_x, min_y, max_y, min_z, max_z,
                    res_x, res_y):
    """
    Sample heights on a res_x × res_y grid stretched across the
    full bounding box.  Returns float64 array shape (res_y, res_x).
    """
    dx = (max_x - min_x) / (res_x - 1) if res_x > 1 else 0
    dy = (max_y - min_y) / (res_y - 1) if res_y > 1 else 0
    mid_z = (min_z + max_z) * 0.5
    search = max(max_x - min_x, max_y - min_y, max_z - min_z) * 2.0 + 10.0

    hmap = np.full((res_y, res_x), float(min_z), dtype=np.float64)

    for row in range(res_y):
        y = max_y - row * dy     # row 0 = top = max_y
        for col in range(res_x):
            x = min_x + col * dx
            loc, _n, _i, _d = bvh.find_nearest(
                Vector((x, y, mid_z)), search)
            if loc is not None:
                hmap[row, col] = loc.z

    return hmap


# =====================================================================
# UE Import Guide writer
# =====================================================================
def _write_ue_guide(path, *, layout, ue_scale_z, ue_loc,
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
    a(f"   Heightmap Size:    {layout['res_x']} x {layout['res_y']}")
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

    row("① World Partition Grid Size", "2")
    row("② World Partition Region Size", "16")
    a(f"   ╠══{'═'*w1}╬══{'═'*w2}╣")
    row("③ Section Size", f"{layout['section_size']}x{layout['section_size']} Quads")
    row("④ Sections Per Component", f"{layout['sec_per_comp']}x{layout['sec_per_comp']} Section")
    row("⑤ Number of Components", f"{layout['comp_x']}  x  {layout['comp_y']}")
    a(f"   ╠══{'═'*w1}╬══{'═'*w2}╣")
    row("⑥ Overall Resolution", f"{layout['res_x']}  x  {layout['res_y']}")
    a(f"   ╠══{'═'*w1}╬══{'═'*w2}╣")
    row("⑦ Location  X", f"{ue_loc[0]:.1f}")
    row("   Location  Y", f"{ue_loc[1]:.1f}")
    row("   Location  Z", f"{ue_loc[2]:.1f}")
    a(f"   ╠══{'═'*w1}╬══{'═'*w2}╣")
    row("⑧ Scale  X", f"{layout['scale_x']:.4f}")
    row("   Scale  Y", f"{layout['scale_y']:.4f}")
    row("   Scale  Z", f"{ue_scale_z:.4f}")

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
    a("")

    a("   ┌─ UE World Size (computed) ─────────────────────────────┐")
    a(f"   │  X: {layout['quads_x']} quads × {layout['scale_x']:.2f} cm"
      f" = {layout['quads_x']*layout['scale_x']/100:.2f} m"
      f"{'':>10s}│")
    a(f"   │  Y: {layout['quads_y']} quads × {layout['scale_y']:.2f} cm"
      f" = {layout['quads_y']*layout['scale_y']/100:.2f} m"
      f"{'':>10s}│")
    a(f"   │  Z: {ue_scale_z:.2f}% of 512cm"
      f" = {ue_scale_z/100*512:.2f} cm"
      f" = {ue_scale_z/100*5.12:.2f} m"
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
   3. Click "..." next to Heightmap File → select the PNG
   4. Enter ALL values from the table above EXACTLY
   5. Click "Import"

   TROUBLESHOOTING:
   • If terrain looks flipped → enable "Flip Y Axis"
   • If heightmap resolution shows "(invalid)" → check that
     the PNG size matches Overall Resolution exactly
   • Scale X/Y = centimetres per quad vertex
   • Scale Z = percentage (100% = 512 cm total height range)
""")
    a("=" * 72)
    a("   Generated by BlenderGIS — UE Heightmap Export")
    a("=" * 72)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    log.info("Saved UE Guide: %s", path)


# =====================================================================
# Operator
# =====================================================================
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

    # --- Properties ---
    target_mpp: FloatProperty(
        name="Target m/pixel",
        description=(
            "Target resolution in metres per pixel. Lower = more detail. "
            "Actual value is adjusted to fit UE layout constraints"
        ),
        default=2.0, min=0.1, max=100.0, precision=2,
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
        name="Export .r16 (RAW)", default=False,
        description="Also save raw uint16 LE file",
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
        layout.separator()
        layout.label(text="UE Layout:")
        layout.prop(self, 'section_size')
        layout.prop(self, 'sec_per_comp')
        layout.separator()
        layout.prop(self, 'export_raw')
        layout.prop(self, 'save_json')

    # Execute
    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh selected")
            return {'CANCELLED'}

        # GeoScene
        geoscn = GeoScene(context.scene)
        geo_scale = geoscn.scale if geoscn.hasScale else 1.0

        # Build bmesh world-space
        depsgraph = context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh_eval = obj_eval.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(mesh_eval)
        bm.transform(obj.matrix_world)

        if len(bm.verts) == 0:
            bm.free(); obj_eval.to_mesh_clear()
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

        # --- Calculate UE layout ------------------------------------------
        sec_sz = int(self.section_size)
        spc = int(self.sec_per_comp)
        layout = _calc_ue_layout(rw_size_x, rw_size_y, self.target_mpp,
                                 sec_sz, spc)

        res_x = layout['res_x']
        res_y = layout['res_y']

        self.report({'INFO'},
            f"Sampling {res_x}×{res_y} heightmap – "
            f"terrain {rw_size_x:.0f}×{rw_size_y:.0f} m …")

        # --- BVH + sample -------------------------------------------------
        bvh = BVHTree.FromBMesh(bm)
        hmap = _sample_heights(bvh, min_x, max_x, min_y, max_y,
                               min_z, max_z, res_x, res_y)
        bm.free()
        obj_eval.to_mesh_clear()

        # --- Normalize uint16 ---------------------------------------------
        z_bl = max_z - min_z
        if z_bl < 1e-9:
            hmap_u16 = np.full((res_y, res_x), 32768, dtype=np.uint16)
        else:
            hn = (hmap - min_z) / z_bl
            hmap_u16 = np.clip(hn * 65535.0, 0, 65535).astype(np.uint16)

        # --- UE Z scale ---------------------------------------------------
        # Scale Z %: 100% → 512 cm total range
        ue_scale_z = (rw_z_range * 100.0 / 512.0) * 100.0 if rw_z_range > 0 else 100.0

        # --- UE Location --------------------------------------------------
        # Location centres the landscape in the UE world.
        # X, Y → 0 (origin), Z → midpoint of height range (cm)
        ue_loc = (0.0, 0.0, ((min_z_rw + max_z_rw) / 2.0) * 100.0)

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
        png_name = os.path.basename(filepath)
        raw_name = os.path.basename(base + '.r16') if self.export_raw else None

        # --- Save PNG (16 bit) --------------------------------------------
        try:
            from ..core.lib import imageio
            imageio.imwrite(filepath, hmap_u16)
            log.info("PNG saved (imageio): %s", filepath)
        except Exception:
            log.warning("imageio failed, fallback to Blender", exc_info=True)
            self._save_png_blender(hmap_u16, filepath, res_x, res_y)

        # --- Save .r16 ----------------------------------------------------
        if self.export_raw:
            hmap_u16.tofile(base + '.r16')

        # --- Save UE Guide (.txt) -----------------------------------------
        guide_path = base + '_UE_IMPORT_GUIDE.txt'
        _write_ue_guide(
            guide_path,
            layout=layout,
            ue_scale_z=ue_scale_z,
            ue_loc=ue_loc,
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

        # --- Save JSON ----------------------------------------------------
        if self.save_json:
            meta = {
                "heightmap_file": png_name,
                "resolution_x": res_x,
                "resolution_y": res_y,
                "ue_settings": {
                    "world_partition_grid_size": 2,
                    "world_partition_region_size": 16,
                    "section_size": f"{sec_sz}x{sec_sz}",
                    "sections_per_component": f"{spc}x{spc}",
                    "number_of_components_x": layout['comp_x'],
                    "number_of_components_y": layout['comp_y'],
                    "total_components": layout['total_components'],
                    "overall_resolution_x": res_x,
                    "overall_resolution_y": res_y,
                    "scale_x": round(layout['scale_x'], 4),
                    "scale_y": round(layout['scale_y'], 4),
                    "scale_z": round(ue_scale_z, 4),
                    "location_x": round(ue_loc[0], 2),
                    "location_y": round(ue_loc[1], 2),
                    "location_z": round(ue_loc[2], 2),
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
            }
            if geo_info:
                meta["georef"] = geo_info
            with open(base + '_ue_scale.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        # --- Done ---------------------------------------------------------
        self.report(
            {'INFO'},
            f"OK – {res_x}×{res_y}  |  "
            f"Scale {layout['scale_x']:.2f}×{layout['scale_y']:.2f}×{ue_scale_z:.2f}%  |  "
            f"Comp {layout['comp_x']}×{layout['comp_y']}  |  "
            f"Guide: {os.path.basename(guide_path)}"
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
    try:
        bpy.utils.register_class(EXPORTGIS_OT_ue_heightmap)
    except ValueError:
        log.warning('Re-registering %s', EXPORTGIS_OT_ue_heightmap)
        unregister()
        bpy.utils.register_class(EXPORTGIS_OT_ue_heightmap)

def unregister():
    bpy.utils.unregister_class(EXPORTGIS_OT_ue_heightmap)
