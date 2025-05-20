import os
import sys
# 현재 폴더의 모듈 불러오기
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cartopy.io.shapereader as shpreader
from shapely.geometry import LineString, MultiLineString

from icosahedral_mesh import get_regionally_refined_mesh
import model_utils

def latlon_to_xyz(lons, lats, radius=1.0):
    """경도·위도(도) 배열 → 구면 좌표계 XYZ."""
    phi = np.deg2rad(lats)
    lam = np.deg2rad(lons)
    x = radius * np.cos(phi) * np.cos(lam)
    y = radius * np.cos(phi) * np.sin(lam)
    z = radius * np.sin(phi)
    return x, y, z

def plot_mesh_with_coastlines(mesh, base_splits, region_splits):
    # 1) 지구본 wireframe
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.3, rstride=2, cstride=2)

    # 2) 해안선 불러와서 3D로 그리기
    shp = shpreader.natural_earth(resolution='110m',
                                 category='physical', name='coastline')
    reader = shpreader.Reader(shp)
    for geom in reader.geometries():
        if isinstance(geom, LineString):
            lon, lat = geom.xy
            x, y, z = latlon_to_xyz(lon, lat)
            ax.plot(x, y, z, color='k', linewidth=0.5)
        elif isinstance(geom, MultiLineString):
            for line in geom:
                lon, lat = line.xy
                x, y, z = latlon_to_xyz(lon, lat)
                ax.plot(x, y, z, color='k', linewidth=0.5)

    # 3) 메쉬 노드 scatter
    verts = mesh.vertices
    ax.scatter(verts[:,0], verts[:,1], verts[:,2], s=2, c='k')

    # 노드 개수 텍스트 표시
    num_nodes = verts.shape[0]
    ax.text2D(0.05, 0.95, f"Nodes: {num_nodes}", transform=ax.transAxes, color='black')

    ax.set_box_aspect([1,1,1])
    ax.set_title(f"Regionally Refined Mesh on Globe\n(base={base_splits}, region={region_splits})")
    ax.axis('off')
    plt.show()

if __name__ == "__main__":
    base_splits   = 5
    region_splits = 6
    region_bbox   = (-10, 40, 90, 180)

    meshes = get_regionally_refined_mesh(base_splits, region_splits, region_bbox)
    final_mesh = meshes[-1]
    plot_mesh_with_coastlines(final_mesh, base_splits, region_splits)
