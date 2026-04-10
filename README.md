<img width="1137" height="320" alt="image" src="https://github.com/user-attachments/assets/74a25575-3e41-43a0-a6de-700496d69d4f" />Blender GIS
==========
Blender minimum version required : v2.83
==
😄 Original creator : https://github.com/domlysz/BlenderGIS.git [domlysz]

Note : Since 2022, the OpenTopography web service requires an API key. Please register to opentopography.org and request a key. This service is still free.


[Wiki](https://github.com/domlysz/BlenderGIS/wiki/Home) - [FAQ](https://github.com/domlysz/BlenderGIS/wiki/FAQ) - [Quick start guide](https://github.com/domlysz/BlenderGIS/wiki/Quick-start) - [Flowchart](https://raw.githubusercontent.com/wiki/domlysz/blenderGIS/flowchart.jpg)
--------------------

## Functionalities overview

**GIS datafile import :** Import in Blender most commons GIS data format : Shapefile vector, raster image, geotiff DEM, OpenStreetMap xml.

There are a lot of possibilities to create a 3D terrain from geographic data with BlenderGIS, check the [Flowchart](https://raw.githubusercontent.com/wiki/domlysz/blenderGIS/flowchart.jpg) to have an overview.

Exemple : import vector contour lines, create faces by triangulation and put a topographic raster texture.

![](https://raw.githubusercontent.com/wiki/domlysz/blenderGIS/Blender28x/gif/bgis_demo_delaunay.gif)

**Grab geodata directly from the web :** display dynamics web maps inside Blender 3d view, requests for OpenStreetMap data (buildings, roads ...), get true elevation data from the NASA SRTM mission.

![](https://raw.githubusercontent.com/wiki/domlysz/blenderGIS/Blender28x/gif/bgis_demo_webdata.gif)

**And more :** Manage georeferencing informations of a scene, compute a terrain mesh by Delaunay triangulation, drop objects on a terrain mesh, make terrain analysis using shader nodes, setup new cameras from geotagged photos, setup a camera to render with Blender a new georeferenced raster.

The Creator of the Budget : ChungSik Shin

# Update
----
you can use heightmap export
- This will provide accurate GIS information. 
<img width="1193" height="393" alt="image" src="https://github.com/user-attachments/assets/af17596f-ed65-442a-8f51-0039cd73cda6" />

## Update Issue : When exporting, the tilemap does not render properly, causing the grid to appear broken in Unreal Engine [Under correction]

# when u use GIS
---
API Key Need

<img width="1478" height="1020" alt="image" src="https://github.com/user-attachments/assets/4772223c-b6ef-4476-8e72-31372e5d180d" />

<img width="1137" height="320" alt="image" src="https://github.com/user-attachments/assets/22277137-5263-4e95-9805-9afe673ad32a" />

https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&west={W}&east={E}&south={S}&north={N}&outputFormat=GTiff&API_Key={API_KEY} <- here delete {API_KEY} and input ur api key
