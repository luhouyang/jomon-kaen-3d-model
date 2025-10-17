# VoxelNeXt
# https://arxiv.org/abs/2303.11301
# https://github.com/dvlab-research/VoxelNeXt

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchinfo

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback

import open3d as o3d
import trimesh

# yapf: disable
# This function is taken from the main repository: https://github.com/arch-inform-kaken-group/jomon-kaen-3d-heatmap/blob/main/src/dataset/processing/pottery.py
def voxelize_pottery_dogu(input_file, target_voxel_resolution):
    try:
        scene = trimesh.load(str(input_file), force="scene")
        if not scene.geometry:
            return None, None, None

        mesh_trimesh = trimesh.util.concatenate(list(scene.geometry.values()))
        if mesh_trimesh.vertices.shape[0] == 0 or mesh_trimesh.faces.shape[0] == 0:
            return None, None, None

        # Load the colors correctly
        vertex_color_trimesh = mesh_trimesh.visual.to_color().vertex.colors

        # Use o3d package for processing
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_color_trimesh[:, :3] / 255.0)

        min_bound, max_bound = mesh_o3d.get_min_bound(), mesh_o3d.get_max_bound()
        max_range = np.max(max_bound - min_bound) + 1e-6
        voxel_size = max_range / (target_voxel_resolution - 1)
        voxel_size_sq = voxel_size**2

        mesh_vertices_np = np.asarray(mesh_o3d.vertices)
        mesh_triangles_np = np.asarray(mesh_o3d.triangles)
        mesh_vertex_colors_np = np.asarray(mesh_o3d.vertex_colors)

        tri_vertices = mesh_vertices_np[mesh_triangles_np]
        tri_colors = mesh_vertex_colors_np[mesh_triangles_np]

        v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
        triangle_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        # Number of random samples
        num_samples_per_triangle = np.ceil(triangle_areas / voxel_size_sq).astype(int) + 20
        total_samples = np.sum(num_samples_per_triangle)
        if total_samples == 0:
            return None, None, None

        # Normalized random samples in 3D space
        triangle_indices = np.repeat(np.arange(len(mesh_triangles_np)), num_samples_per_triangle)
        r = np.random.rand(total_samples, 2)
        r_sum = np.sum(r, axis=1)
        r[r_sum > 1] = 1 - r[r_sum > 1]
        bary_coords = np.zeros((total_samples, 3))
        bary_coords[:, [1, 2]] = r
        bary_coords[:, 0] = 1 - np.sum(r, axis=1)

        # Transform the normalized sample to real coordinates
        all_sample_points = np.einsum('ij,ijk->ik', bary_coords, tri_vertices[triangle_indices])
        all_interp_colors = np.einsum('ik,ijk->ik', bary_coords, tri_colors[triangle_indices])

        # Group into voxels
        voxel_coords_all = np.floor((all_sample_points - min_bound) / voxel_size).astype(int)
        df = pd.DataFrame(voxel_coords_all, columns=['x', 'y', 'z'])
        df[['r', 'g', 'b']] = all_interp_colors
        # Get the mean color in each voxel, allow arbitary resolutions of pottery with acceptable color faithfulness
        voxel_data_df = df.groupby(['x', 'y', 'z'])[['r', 'g', 'b']].mean()
        final_coords = np.stack(voxel_data_df.index.to_numpy())
        final_colors_np = voxel_data_df.to_numpy()
        voxel_points = min_bound + (final_coords + 0.5) * voxel_size # Get the centers

        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
        voxel_pcd.colors = o3d.utility.Vector3dVector(final_colors_np)
        return voxel_pcd, min_bound, voxel_size
    except Exception as e:
        print(f"Could not voxelize '{input_file}': {e}")
        return None, None, None
# yapf: enable

def main():
    pass

if __name__ == "__main__":
    main()
