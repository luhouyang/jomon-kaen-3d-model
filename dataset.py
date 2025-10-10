import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
import pandas as pd
import os


class PreprocessJomonKaenDataset(Dataset):

    def __init__(self,
                 data,
                 pottery_path,
                 n_samples_global,
                 n_samples_local,
                 num_local_seeds=64,
                 local_radius=7.5):
        super(PreprocessJomonKaenDataset, self).__init__()
        self.data = data
        self.n_samples_global = n_samples_global
        self.n_samples_local = n_samples_local
        self.num_local_seeds = num_local_seeds
        self.local_radius = local_radius
        self.total_samples = n_samples_global + n_samples_local
        self.headers = [
            'CODE', 'HAS_FLAME_LIKE_DECORATION', 'HAS_CROWN_LIKE_DECORATION',
            'HAS_HANDLES', 'HAS_CORD_MARKED_PATTERN', 'HAS_NAIL_ENGRAVING',
            'HAS_SPIRAL_PATTERN', 'NUMBER_OF_PERTRUSIONS_0.0',
            'NUMBER_OF_PERTRUSIONS_1.0', 'NUMBER_OF_PERTRUSIONS_2.0',
            'NUMBER_OF_PERTRUSIONS_3.0', 'NUMBER_OF_PERTRUSIONS_4.0',
            'NUMBER_OF_PERTRUSIONS_6.0', 'NUMBER_OF_PERTRUSIONS_8.0'
        ]
        self.labels = pd.read_csv(pottery_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pottery_file = str(self.data[index]['processed_pottery_path'])
        pottery_name = os.path.basename(pottery_file)
        pottery_pcd = o3d.io.read_point_cloud(pottery_file)

        if len(pottery_pcd.points) < self.total_samples:
            return (torch.zeros((self.total_samples, 6), dtype=torch.float32),
                    torch.zeros(14, dtype=torch.float32))

        pottery_points = np.asarray(pottery_pcd.points, dtype=np.float32)
        pottery_colors = np.asarray(
            pottery_pcd.colors,
            dtype=np.float32) if pottery_pcd.has_colors() else np.full_like(
                pottery_points, 0.5, dtype=np.float32)

        shuffled_indices = np.random.permutation(len(pottery_points))
        shuffled_pcd = o3d.geometry.PointCloud()
        shuffled_pcd.points = o3d.utility.Vector3dVector(
            pottery_points[shuffled_indices])
        shuffled_pcd.colors = o3d.utility.Vector3dVector(
            pottery_colors[shuffled_indices])

        # 1. Global Sampling: Get the primary set of points for overall shape context.
        pcd_fps_global = shuffled_pcd.farthest_point_down_sample(
            self.n_samples_global)
        global_points = np.asarray(pcd_fps_global.points, dtype=np.float32)
        global_colors = np.asarray(pcd_fps_global.colors, dtype=np.float32)

        if self.n_samples_local <= 0:
            combined_xyz_rgb = np.hstack((global_points, global_colors))
            combined_tensor = torch.from_numpy(
                combined_xyz_rgb.astype(np.float32))
            target_tensor = torch.from_numpy(
                self.labels.loc[self.labels['CODE'] == pottery_name,
                                self.headers[1:]].values.astype(
                                    np.float32)).squeeze()
            return combined_tensor, target_tensor
            # If no local samples are needed, return early.

        # 2. Seed Selection:
        # From the global points, select a smaller, evenly-spaced subset to act as seeds.
        if self.n_samples_global <= self.num_local_seeds:
            pcd_seeds = pcd_fps_global
        else:
            pcd_seeds = pcd_fps_global.farthest_point_down_sample(
                self.num_local_seeds)

        # 3. Local Sampling: Use the sparse seeds for the radius search.
        pcd_tree = o3d.geometry.KDTreeFlann(pottery_pcd)
        local_indices = []
        for point in pcd_seeds.points:
            [_, idx,
             _] = pcd_tree.search_radius_vector_3d(point, self.local_radius)
            local_indices.extend(idx)

        unique_local_indices = np.unique(local_indices)

        if len(unique_local_indices) > 10:
            local_patch_pcd = pottery_pcd.select_by_index(unique_local_indices)
        else:
            rand_start_idx = np.random.randint(
                0,
                len(pottery_points) - self.n_samples_local)
            patch_indices = np.arange(rand_start_idx,
                                      rand_start_idx + self.n_samples_local)
            local_patch_pcd = pottery_pcd.select_by_index(patch_indices)

        pcd_fps_local = local_patch_pcd.farthest_point_down_sample(
            self.n_samples_local)
        local_points = np.asarray(pcd_fps_local.points, dtype=np.float32)
        local_colors = np.asarray(pcd_fps_local.colors, dtype=np.float32)

        if len(local_points) < self.n_samples_local:
            if len(local_points) == 0:
                idx = np.random.choice(len(pottery_points),
                                       self.n_samples_local,
                                       replace=True)
                local_points = pottery_points[idx, :]
                local_colors = pottery_colors[idx, :]
            else:
                idx = np.random.choice(len(local_points),
                                       self.n_samples_local,
                                       replace=True)
                local_points = local_points[idx, :]
                local_colors = local_colors[idx, :]

        # 4. Final Combination: Combine GLOBAL points and LOCAL points
        combined_points = np.vstack((global_points, local_points))
        combined_colors = np.vstack((global_colors, local_colors))
        combined_xyz_rgb = np.hstack((combined_points, combined_colors))

        combined_tensor = torch.from_numpy(combined_xyz_rgb.astype(np.float32))
        target_tensor = torch.from_numpy(
            self.labels.loc[self.labels['CODE'] == pottery_name,
                            self.headers[1:]].values.astype(
                                np.float32)).squeeze()

        return combined_tensor, target_tensor
