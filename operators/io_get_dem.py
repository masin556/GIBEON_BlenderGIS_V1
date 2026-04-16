import os
import time

import logging
log = logging.getLogger(__name__)

from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import bpy
import bmesh
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, FloatVectorProperty

from ..geoscene import GeoScene
from .utils import adjust3Dview, getBBOX, isTopView
from ..core.proj import SRS, reprojBbox

from ..core import settings
USER_AGENT = settings.user_agent
from ..core.lib import imghdr

PKG, SUBPKG = __package__.split('.', maxsplit=1)

TIMEOUT = 120
MIN_DEM_BYTES = 4096


def _is_valid_dem_payload(data):
	if data is None or len(data) < MIN_DEM_BYTES:
		return False
	if data.startswith(b'<?xml') or data.startswith(b'<html') or data.startswith(b'{"error"'):
		return False
	img_fmt = imghdr.what(None, data)
	return img_fmt in {'tiff', 'geotiff'}


def _build_dem_url_candidates(base_url, prefer_high_res_srtm=True, allow_srtm_fallback=True):
	candidates = [base_url]
	if prefer_high_res_srtm and 'demtype=SRTMGL3' in base_url:
		candidates.insert(0, base_url.replace('demtype=SRTMGL3', 'demtype=SRTMGL1'))
	if allow_srtm_fallback and 'demtype=SRTMGL1' in base_url:
		candidates.append(base_url.replace('demtype=SRTMGL1', 'demtype=SRTMGL3'))
	# Keep order while removing duplicates.
	return list(dict.fromkeys(candidates))

class IMPORTGIS_OT_dem_query(Operator):
	"""Import elevation data from a web service"""

	bl_idname = "importgis.dem_query"
	bl_description = 'Query for elevation data from a web service'
	bl_label = "Get elevation (SRTM)"
	bl_options = {"UNDO"}

	prefer_high_res_srtm: BoolProperty(
		name="Prefer SRTM 30m", default=True,
		description="Try SRTMGL1 (30m) first when SRTM data source is used",
	)
	allow_srtm_fallback: BoolProperty(
		name="Allow 90m Fallback", default=True,
		description="Fallback to SRTMGL3 (90m) if high-resolution request fails",
	)
	fill_nodata: BoolProperty(
		name="Fill DEM Nodata", default=True,
		description="Interpolate no-data values for more stable terrain displacement",
	)

	def invoke(self, context, event):

		#check georef
		geoscn = GeoScene(context.scene)
		if not geoscn.isGeoref:
				self.report({'ERROR'}, "Scene is not georef")
				return {'CANCELLED'}
		if geoscn.isBroken:
				self.report({'ERROR'}, "Scene georef is broken, please fix it beforehand")
				return {'CANCELLED'}

		#return self.execute(context)
		return context.window_manager.invoke_props_dialog(self)#, width=350)

	def draw(self,context):
		prefs = context.preferences.addons[PKG].preferences
		layout = self.layout
		row = layout.row(align=True)
		row.prop(prefs, "demServer", text='Server')
		if 'opentopography' in prefs.demServer:
			row = layout.row(align=True)
			row.prop(prefs, "opentopography_api_key", text='Api Key')
		layout.prop(self, 'prefer_high_res_srtm')
		layout.prop(self, 'allow_srtm_fallback')
		layout.prop(self, 'fill_nodata')

	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT'

	def execute(self, context):

		prefs = bpy.context.preferences.addons[PKG].preferences
		scn = context.scene
		geoscn = GeoScene(scn)
		crs = SRS(geoscn.crs)

		#Validate selection
		objs = bpy.context.selected_objects
		aObj = context.active_object
		if len(objs) == 1 and aObj.type == 'MESH':
			onMesh = True
			bbox = getBBOX.fromObj(aObj).toGeo(geoscn)
		elif isTopView(context):
			onMesh = False
			bbox = getBBOX.fromTopView(context).toGeo(geoscn)
		else:
			self.report({'ERROR'}, "Please define the query extent in orthographic top view or by selecting a reference object")
			return {'CANCELLED'}

		if bbox.dimensions.x > 1000000 or bbox.dimensions.y > 1000000:
			self.report({'ERROR'}, "Too large extent")
			return {'CANCELLED'}

		bbox = reprojBbox(geoscn.crs, 4326, bbox)

		if 'SRTM' in prefs.demServer:
			if bbox.ymin > 60:
				self.report({'ERROR'}, "SRTM is not available beyond 60 degrees north")
				return {'CANCELLED'}
			if bbox.ymax < -56:
				self.report({'ERROR'}, "SRTM is not available below 56 degrees south")
				return {'CANCELLED'}

		if 'opentopography' in prefs.demServer:
			if not prefs.opentopography_api_key:
				self.report({'ERROR'}, "Please register to opentopography.org and request for an API key")
				return {'CANCELLED'}

		#Set cursor representation to 'loading' icon
		w = context.window
		w.cursor_set('WAIT')
		try:
			#url template
			e = 0.002 #opentopo service does not always respect the entire bbox, so request for a little more
			xmin, xmax = bbox.xmin - e, bbox.xmax + e
			ymin, ymax = bbox.ymin - e, bbox.ymax + e

			base_url = prefs.demServer.format(W=xmin, E=xmax, S=ymin, N=ymax, API_KEY=prefs.opentopography_api_key)
			url_candidates = _build_dem_url_candidates(
				base_url,
				prefer_high_res_srtm=self.prefer_high_res_srtm,
				allow_srtm_fallback=self.allow_srtm_fallback,
			)

			# Download the file from url and save it locally
			# opentopo return a geotiff object in wgs84
			if bpy.data.is_saved:
				filePath = os.path.join(os.path.dirname(bpy.data.filepath), 'srtm.tif')
			else:
				filePath = os.path.join(bpy.app.tempdir, 'srtm.tif')

			last_error = None
			used_url = None
			for url in url_candidates:
				log.debug("DEM candidate URL: %s", url)
				rq = Request(url, headers={'User-Agent': USER_AGENT})
				try:
					with urlopen(rq, timeout=TIMEOUT) as response:
						data = response.read()
					if not _is_valid_dem_payload(data):
						last_error = RuntimeError("Response is not a valid DEM GeoTIFF payload")
						continue
					with open(filePath, 'wb') as outFile:
						outFile.write(data)
					used_url = url
					break
				except (URLError, HTTPError, TimeoutError, RuntimeError) as err:
					last_error = err
					log.warning(
						"DEM request failed for url:%s code:%s error:%s",
						url, getattr(err, 'code', None), err
					)

			if used_url is None:
				log.error("All DEM candidates failed. last_error=%s", last_error)
				self.report({'ERROR'}, "Cannot download a valid DEM from configured providers")
				return {'CANCELLED'}

			if 'demtype=' in used_url:
				self.report({'INFO'}, f"DEM source selected: {used_url.split('demtype=')[1].split('&')[0]}")

			if not onMesh:
				bpy.ops.importgis.georaster(
				'EXEC_DEFAULT',
				filepath = filePath,
				reprojection = True,
				rastCRS = 'EPSG:4326',
				importMode = 'DEM',
				subdivision = 'subsurf',
				demInterpolation = True,
				fillNodata = self.fill_nodata)
			else:
				bpy.ops.importgis.georaster(
				'EXEC_DEFAULT',
				filepath = filePath,
				reprojection = True,
				rastCRS = 'EPSG:4326',
				importMode = 'DEM',
				subdivision = 'subsurf',
				demInterpolation = True,
				demOnMesh = True,
				objectsLst = [str(i) for i, obj in enumerate(scn.collection.all_objects) if obj.name == bpy.context.active_object.name][0],
				clip = False,
				fillNodata = self.fill_nodata)
		finally:
			w.cursor_set('DEFAULT')

		bbox = getBBOX.fromScn(scn)
		adjust3Dview(context, bbox, zoomToSelect=False)

		return {'FINISHED'}


def register():
	try:
		bpy.utils.register_class(IMPORTGIS_OT_dem_query)
	except ValueError as e:
		log.warning('{} is already registered, now unregister and retry... '.format(IMPORTGIS_OT_srtm_query))
		unregister()
		bpy.utils.register_class(IMPORTGIS_OT_dem_query)

def unregister():
	bpy.utils.unregister_class(IMPORTGIS_OT_dem_query)
