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

# try:
#     import spconv.pytorch as spconv
#     from spconv.pytorch import SubMConv3d, SparseConv3d, SparseInverseConv3d
# except ImportError:
#     raise ImportError("Please install spconv-cu118: `pip install spconv-cu118`")
import spconv.pytorch as spconv
from spconv.pytorch import SubMConv3d, SparseConv3d, SparseInverseConv3d


# yapf: disable
# This function is taken from the main repository: https://github.com/arch-inform-kaken-group/jomon-kaen-3d-heatmap/blob/main/src/dataset/processing/pottery.py
def voxelize_pottery_dogu(input_file, target_voxel_resolution):
    """Voxelizes a 3D mesh file into a colored point cloud."""
    try:
        scene = trimesh.load(str(input_file), force="scene")
        if not scene.geometry:
            return None, None, None

        mesh_trimesh = trimesh.util.concatenate(list(scene.geometry.values()))
        if mesh_trimesh.vertices.shape[0] == 0 or mesh_trimesh.faces.shape[0] == 0:
            return None, None, None

        vertex_color_trimesh = mesh_trimesh.visual.to_color().vertex_colors

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_color_trimesh[:, :3] / 255.0)

        min_bound, max_bound = mesh_o3d.get_min_bound(), mesh_o3d.get_max_bound()
        max_range = np.max(max_bound - min_bound) + 1e-6
        voxel_size = max_range / (target_voxel_resolution - 1)
        voxel_size_sq = voxel_size**2

        mesh_vertices_np = np.asarray(mesh_o3d.vertices)
        mesh_vertex_colors_np = np.asarray(mesh_o3d.vertex_colors)
        mesh_triangles_np = np.asarray(mesh_o3d.triangles)

        tri_vertices = mesh_vertices_np[mesh_triangles_np]
        tri_colors = mesh_vertex_colors_np[mesh_triangles_np]

        v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
        triangle_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        num_samples_per_triangle = np.ceil(triangle_areas / voxel_size_sq).astype(int) + 20
        total_samples = np.sum(num_samples_per_triangle)
        if total_samples == 0:
            return None, None, None

        triangle_indices = np.repeat(np.arange(len(mesh_triangles_np)), num_samples_per_triangle)
        r = np.random.rand(total_samples, 2)
        r_sum = np.sum(r, axis=1)
        r[r_sum > 1] = 1 - r[r_sum > 1]
        bary_coords = np.zeros((total_samples, 3))
        bary_coords[:, [1, 2]] = r
        bary_coords[:, 0] = 1 - np.sum(r, axis=1)

        all_sample_points = np.einsum('ij,ijk->ik', bary_coords, tri_vertices[triangle_indices])
        all_interp_colors = np.einsum('ij,ijk->ik', bary_coords, tri_colors[triangle_indices])

        voxel_coords_all = np.floor((all_sample_points - min_bound) / voxel_size).astype(int)
        df = pd.DataFrame(voxel_coords_all, columns=['x', 'y', 'z'])
        df[['r', 'g', 'b']] = all_interp_colors
        voxel_data_df = df.groupby(['x', 'y', 'z'])[['r', 'g', 'b']].mean()
        final_coords = np.stack(voxel_data_df.index.to_numpy())
        final_colors_np = voxel_data_df.to_numpy()
        voxel_points = min_bound + (final_coords + 0.5) * voxel_size

        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
        voxel_pcd.colors = o3d.utility.Vector3dVector(final_colors_np)
        return voxel_pcd, min_bound, voxel_size
    except Exception as e:
        print(f"Could not voxelize '{input_file}': {e}")
        return None, None, None
# yapf: enable


class VoxelDataset(Dataset):
    def __init__(
        self,
        data,
        labels_csv_path,
        voxel_resolution,
        cache_dir,
        feature_columns,
        augment_color_p=0.5,
        color_jitter_std=0.05,
        jitter_voxel_p=0.1,
    ):
        super().__init__()
        self.data = data
        self.labels_df = pd.read_csv(labels_csv_path)
        self.voxel_resolution = voxel_resolution
        self.cache_dir = cache_dir
        self.feature_columns = feature_columns
        self.augment_color_p = augment_color_p
        self.color_jitter_std = color_jitter_std
        self.jitter_voxel_p = jitter_voxel_p

    def __len__(self):
        return len(self.data)

    def color_jitter_pcd(self, voxel_pcd):
        if np.random.rand() < self.augment_color_p:
            colors = np.asarray(voxel_pcd.colors).astype(np.float32)
            num_points = colors.shape[0]
            jitter_mask = np.random.rand(num_points) < self.jitter_voxel_p
            if np.any(jitter_mask):
                noise = np.random.normal(0.0, self.color_jitter_std, colors[jitter_mask].shape).astype(np.float32)
                colors[jitter_mask] += noise
                colors = np.clip(colors, 0.0, 1.0)
                voxel_pcd.colors = o3d.utility.Vector3dVector(colors)
        return voxel_pcd

    def __getitem__(self, index):
        item_info = self.data[index]
        mesh_file_path = item_info['mesh_path']
        original_filename = os.path.basename(mesh_file_path)
        base_name = os.path.splitext(original_filename)[0]
        lookup_name = f"{base_name}.ply"

        cached_ply_path = os.path.join(self.cache_dir, f"{base_name}.ply")
        cached_meta_path = os.path.join(self.cache_dir, f"{base_name}.npz")
        voxel_pcd, min_bound, voxel_size = None, None, None

        if os.path.exists(cached_ply_path) and os.path.exists(cached_meta_path):
            try:
                voxel_pcd = o3d.io.read_point_cloud(cached_ply_path)
                meta = np.load(cached_meta_path)
                min_bound, voxel_size = meta['min_bound'], meta['voxel_size']
            except Exception:
                voxel_pcd = None

        if voxel_pcd is None:
            voxel_pcd, min_bound, voxel_size = voxelize_pottery_dogu(mesh_file_path, self.voxel_resolution)
            if voxel_pcd is not None and voxel_pcd.has_points():
                o3d.io.write_point_cloud(cached_ply_path, voxel_pcd)
                np.savez(cached_meta_path, min_bound=min_bound, voxel_size=voxel_size)

        if voxel_pcd is not None and voxel_pcd.has_points():
            voxel_pcd = self.color_jitter_pcd(voxel_pcd)

        dense_voxel_tensor = torch.zeros((3, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution), dtype=torch.float32)

        if voxel_pcd is not None and voxel_pcd.has_points():
            points, colors = np.asarray(voxel_pcd.points), np.asarray(voxel_pcd.colors).astype(np.float32)
            indices = np.floor((points - min_bound) / voxel_size).astype(int)
            indices = np.clip(indices, 0, self.voxel_resolution - 1)
            Dx, Dy, Dz = indices[:, 0], indices[:, 1], indices[:, 2]
            Cr, Cg, Cb = colors[:, 0], colors[:, 1], colors[:, 2]
            dense_voxel_tensor[0, Dz, Dy, Dx] = torch.from_numpy(Cr)
            dense_voxel_tensor[1, Dz, Dy, Dx] = torch.from_numpy(Cg)
            dense_voxel_tensor[2, Dz, Dy, Dx] = torch.from_numpy(Cb)

        target_row = self.labels_df[self.labels_df['CODE'] == lookup_name]
        if target_row.empty:
            raise ValueError(f"WARNING: No labels found for '{lookup_name}' in the CSV.")
        else:
            target_values = target_row[self.feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values

        target_tensor = torch.from_numpy(target_values).squeeze().float()
        return dense_voxel_tensor, target_tensor


class JomonKaenVoxelDataModule(pl.LightningDataModule):
    def __init__(self, labels_csv_path, mesh_dir, cache_dir, feature_columns, voxel_resolution, batch_size=8, num_workers=4, use_weighted_loss=False):
        super().__init__()
        self.save_hyperparameters()
        self.pos_weight = None

    def setup(self, stage=None):
        os.makedirs(self.hparams.cache_dir, exist_ok=True)
        print(f"Voxel cache is located at: {os.path.abspath(self.hparams.cache_dir)}")
        all_files = [{'mesh_path': os.path.join(self.hparams.mesh_dir, p), 'POTTERY': p} for p in os.listdir(self.hparams.mesh_dir) if p.endswith('.glb')]
        if not all_files:
            raise FileNotFoundError(f"'.glb' files not found in the directory '{os.path.abspath(self.hparams.mesh_dir)}'")
        np.random.shuffle(all_files)
        test_groups = {"IN0009(5).glb", "NM0049(30).glb", "UD0005(71).glb", "NM0015(27).glb", "NM0066(31).glb", "NM0099(37).glb", "NM0154(42).glb", "NM0168(45).glb", "SI0001(53).glb", "SJ0503(54).glb"}
        common_params = {"labels_csv_path": self.hparams.labels_csv_path, "voxel_resolution": self.hparams.voxel_resolution, "cache_dir": self.hparams.cache_dir, "feature_columns": self.hparams.feature_columns}
        train_data = [d for d in all_files if d['POTTERY'] not in test_groups]
        val_data = [d for d in all_files if d['POTTERY'] in test_groups]
        if not train_data:
            print("WARNING: The training dataset is empty.")
        self.train_dataset = VoxelDataset(train_data, **common_params)
        self.val_dataset = VoxelDataset(val_data, **common_params)
        if stage in ('fit', None) and self.hparams.use_weighted_loss:
            print("Calculating positive class weights for weighted loss")
            train_files_for_lookup = [f"{os.path.splitext(d['POTTERY'])[0]}.ply" for d in self.train_dataset.data]
            all_labels_df = pd.read_csv(self.hparams.labels_csv_path)
            train_labels_df = all_labels_df[all_labels_df['CODE'].isin(train_files_for_lookup)]
            labels = train_labels_df[self.hparams.feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values
            num_samples, num_positives = len(labels), np.sum(labels, axis=0)
            num_negatives = num_samples - num_positives
            pos_weight_np = num_negatives / (num_positives + 1e-6)
            self.pos_weight = torch.tensor(pos_weight_np.astype(np.float32))
        elif stage in ('fit', None):
            print("Weighted loss is disabled")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True)


# --------------------------------------------------------------------------- #
#                      NEW VOXELNEXT-BASED MODEL DEFINITION                   #
# --------------------------------------------------------------------------- #

class SparseResidualBlock(spconv.SparseModule):
    """A residual block for sparse convolutions."""
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        self.conv1 = SubMConv3d(in_channels, out_channels, kernel_size=3, bias=False, indice_key=indice_key)
        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = SubMConv3d(out_channels, out_channels, kernel_size=3, bias=False, indice_key=indice_key)
        self.bn2 = norm_fn(out_channels)

        if in_channels != out_channels:
            self.downsample = spconv.SparseSequential(
                SparseConv3d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_fn(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        return out


class VoxelNeXtBackbone(nn.Module):
    """
    Implements the sparse backbone architecture inspired by VoxelNeXt,
    featuring multiple downsampling stages to increase receptive field.
    """
    def __init__(self, input_channels, grid_size):
        super().__init__()
        self.sparse_shape = grid_size
        norm_fn = nn.BatchNorm1d

        # Input layer
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm0'),
            norm_fn(16),
            nn.ReLU(),
        )

        # Backbone stages
        self.conv1 = spconv.SparseSequential(
            SparseResidualBlock(16, 16, norm_fn, indice_key='subm1'),
            SparseResidualBlock(16, 16, norm_fn, indice_key='subm1'),
        )
        self.down1 = spconv.SparseSequential(
            SparseConv3d(16, 32, 3, 2, padding=1, bias=False, indice_key='spconv2'),
            norm_fn(32), nn.ReLU(),
        )
        self.conv2 = spconv.SparseSequential(
            SparseResidualBlock(32, 32, norm_fn, indice_key='subm2'),
            SparseResidualBlock(32, 32, norm_fn, indice_key='subm2'),
        )
        self.down2 = spconv.SparseSequential(
            SparseConv3d(32, 64, 3, 2, padding=1, bias=False, indice_key='spconv3'),
            norm_fn(64), nn.ReLU(),
        )
        self.conv3 = spconv.SparseSequential(
            SparseResidualBlock(64, 64, norm_fn, indice_key='subm3'),
            SparseResidualBlock(64, 64, norm_fn, indice_key='subm3'),
        )
        self.down3 = spconv.SparseSequential(
            SparseConv3d(64, 128, 3, 2, padding=1, bias=False, indice_key='spconv4'),
            norm_fn(128), nn.ReLU(),
        )
        self.conv4 = spconv.SparseSequential(
            SparseResidualBlock(128, 128, norm_fn, indice_key='subm4'),
            SparseResidualBlock(128, 128, norm_fn, indice_key='subm4'),
        )
        self.down4 = spconv.SparseSequential(
            SparseConv3d(128, 128, 3, 2, padding=1, bias=False, indice_key='spconv5'),
            norm_fn(128), nn.ReLU(),
        )
        self.conv5 = spconv.SparseSequential(
            SparseResidualBlock(128, 128, norm_fn, indice_key='subm5'),
            SparseResidualBlock(128, 128, norm_fn, indice_key='subm5'),
        )
        self.down5 = spconv.SparseSequential(
            SparseConv3d(128, 128, 3, 2, padding=1, bias=False, indice_key='spconv6'),
            norm_fn(128), nn.ReLU(),
        )
        self.conv6 = spconv.SparseSequential(
            SparseResidualBlock(128, 128, norm_fn, indice_key='subm6'),
            SparseResidualBlock(128, 128, norm_fn, indice_key='subm6'),
        )
        
        # Upsampling and feature aggregation layers
        self.upsample5 = SparseInverseConv3d(128, 128, 3, indice_key='spconv5', bias=False)
        self.upsample6 = spconv.SparseSequential(
            SparseInverseConv3d(128, 128, 3, indice_key='spconv6', bias=False),
            SparseInverseConv3d(128, 128, 3, indice_key='spconv5', bias=False),
        )

        self.output_channels = 128 * 3 # From concatenating stages 4, 5, 6

    def forward(self, voxel_features, voxel_coords, batch_size):
        sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords,
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x0 = self.conv_input(sp_tensor)
        
        x1 = self.conv1(x0)
        x2_in = self.down1(x1)
        
        x2 = self.conv2(x2_in)
        x3_in = self.down2(x2)
        
        x3 = self.conv3(x3_in)
        x4_in = self.down3(x3)
        
        x4 = self.conv4(x4_in)
        x5_in = self.down4(x4)
        
        x5 = self.conv5(x5_in)
        x6_in = self.down5(x5)
        
        x6 = self.conv6(x6_in)

        # Feature aggregation
        x5_up = self.upsample5(x5)
        x6_up = self.upsample6(x6)
        
        # Concatenate features
        features_cat = torch.cat([x4.features, x5_up.features, x6_up.features], dim=1)
        
        # Create a new sparse tensor with the concatenated features
        # The indices from x4 define the final sparse tensor structure
        out_sp_tensor = x4.replace_feature(features_cat)
        
        return out_sp_tensor


class VoxelNeXtForClassification(nn.Module):
    """
    Main model that combines the VoxelNeXt backbone with a classification head.
    Handles dense-to-sparse conversion and global pooling.
    """
    def __init__(self, num_outputs, resolution):
        super().__init__()
        self.resolution = resolution
        grid_size = [resolution, resolution, resolution]
        
        self.backbone = VoxelNeXtBackbone(input_channels=3, grid_size=grid_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.output_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Convert dense input to sparse format
        coords = torch.nonzero(x.sum(dim=1))  # (N, 4) -> (batch, z, y, x)
        
        if coords.shape[0] == 0:
            # Handle empty batch
            return torch.zeros((batch_size, self.classifier[-1].out_features), device=x.device)
            
        features = x[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]]
        
        # 2. Pass through sparse backbone
        backbone_out = self.backbone(features, coords.int(), batch_size)
        
        # 3. Global Average Pooling
        pooled_features = []
        for i in range(batch_size):
            batch_mask = backbone_out.indices[:, 0] == i
            if batch_mask.sum() > 0:
                pooled_features.append(backbone_out.features[batch_mask].mean(dim=0))
            else:
                # Append a zero tensor if a sample in the batch is empty
                pooled_features.append(torch.zeros(self.backbone.output_channels, device=x.device))
        
        pooled_features = torch.stack(pooled_features, dim=0)

        # 4. Final classification
        logits = self.classifier(pooled_features)
        return logits


class SaveValidationPredictions(Callback):
    def __init__(self, output_dir, feature_columns, save_every_n_epochs=10):
        super().__init__()
        self.output_dir = output_dir
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(self.output_dir, exist_ok=True)
        self.best_val_loss = float('inf')
        self.headers = feature_columns

    def on_validation_epoch_start(self, trainer, pl_module):
        self.preds, self.targets = [], []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, dict) and 'preds' in outputs and 'targets' in outputs:
            self.preds.append(outputs['preds'].cpu())
            self.targets.append(outputs['targets'].cpu())

    def _save_to_file(self, pl_module, filename):
        all_preds = torch.cat(self.preds).int().numpy()
        all_targets = torch.cat(self.targets).int().numpy()
        df_preds = pd.DataFrame(all_preds, columns=self.headers)
        df_targets = pd.DataFrame(all_targets, columns=self.headers)
        df_preds['source'] = 'prediction'
        df_targets['source'] = 'ground_truth'
        df_preds['sample_id'] = range(len(df_preds))
        df_targets['sample_id'] = range(len(df_targets))
        df_combined = pd.concat([df_targets, df_preds], ignore_index=True).sort_values(by=['sample_id', 'source'], ascending=[True, False])
        df_combined = df_combined[['sample_id', 'source'] + self.headers]
        filepath = os.path.join(self.output_dir, filename)
        df_combined.to_csv(filepath, index=False)
        pl_module.print(f"\nSaved validation predictions to {filepath}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self.preds:
            return
        epoch = trainer.current_epoch + 1
        current_val_loss = trainer.callback_metrics.get('val_loss')
        is_interval_save = (epoch > 0 and epoch % self.save_every_n_epochs == 0)
        is_best_loss_save = (current_val_loss is not None and current_val_loss < self.best_val_loss)
        if is_interval_save:
            self._save_to_file(pl_module, filename=f"eval_predictions_epoch_{epoch}.csv")
        if is_best_loss_save:
            self.best_val_loss = current_val_loss.item()
            self._save_to_file(pl_module, filename="eval_prediction_best.csv")
            if not is_interval_save:
                self._save_to_file(pl_module, filename=f"eval_predictions_epoch_{epoch}_best_loss.csv")


class VoxelNetLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, lr_final=1e-6, num_outputs=13, pos_weight=None, resolution=128, l1_lambda=0.0, use_weighted_loss=False, warmup_start_factor=0.01, warmup_total_iters=100):
        super().__init__()
        self.save_hyperparameters()
        
        # --- Use the new VoxelNeXt-based model ---
        self.model = VoxelNeXtForClassification(
            num_outputs=self.hparams.num_outputs,
            resolution=self.hparams.resolution,
        )

        if self.hparams.use_weighted_loss:
            print("Using weighted loss in criterion")
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.hparams.pos_weight)
        else:
            print("Not using weighted loss in criterion")
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def compute_l1_loss(self):
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return l1_norm

    def _shared_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        if self.hparams.l1_lambda > 0:
            l1_loss = self.compute_l1_loss()
            loss = loss + self.hparams.l1_lambda * l1_loss
        preds = torch.sigmoid(outputs) > 0.5
        accuracy = (preds == targets.bool()).float().mean()
        return loss, accuracy, preds, targets

    def training_step(self, batch, batch_idx):
        loss, accuracy, _, _ = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        if self.hparams.l1_lambda > 0:
            l1_loss = self.compute_l1_loss()
            self.log('train_l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, preds, targets = self._shared_step(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)
        if self.hparams.l1_lambda > 0:
            l1_loss = self.compute_l1_loss()
            self.log('val_l1_loss', l1_loss, on_epoch=True, prog_bar=False)
        return {'preds': preds, 'targets': targets}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams.warmup_start_factor, total_iters=self.hparams.warmup_total_iters)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs - self.hparams.warmup_total_iters, eta_min=self.hparams.lr_final)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[self.hparams.warmup_total_iters])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    FEATURE_COLUMNS = ['HAS_FLAME_LIKE_DECORATION', 'HAS_CROWN_LIKE_DECORATION', 'HAS_HANDLES', 'HAS_CORD_MARKED_PATTERN', 'HAS_NAIL_ENGRAVING', 'HAS_SPIRAL_PATTERN', 'NUMBER_OF_PERTRUSIONS_0.0', 'NUMBER_OF_PERTRUSIONS_1.0', 'NUMBER_OF_PERTRUSIONS_2.0', 'NUMBER_OF_PERTRUSIONS_3.0', 'NUMBER_OF_PERTRUSIONS_4.0', 'NUMBER_OF_PERTRUSIONS_6.0', 'NUMBER_OF_PERTRUSIONS_8.0']
    NUM_OUTPUTS = len(FEATURE_COLUMNS)
    VOXEL_RESOLUTION = 128
    BATCH_SIZE = 4 # Reduced batch size for potentially larger model
    MAX_EPOCHS = 1000
    NUM_WORKERS = 4 # Adjust based on your machine's capability
    LEARNING_RATE_INITIAL = 1e-4 # May need a smaller LR for larger models
    LEARNING_RATE_FINAL = 1e-6
    USE_WEIGHTED_LOSS = False
    VISUALIZE_SAMPLES = False
    L1_LAMBDA = 1e-5

    # --- PLEASE UPDATE THESE PATHS ---
    LABELS_CSV_PATH = r"path/to/your/DS_Labels_Cleaned.csv"
    MESH_DIR = r"path/to/your/pottery_only"
    CACHE_DIR = r"voxel_cache_voxelnext"
    # ---------------------------------

    # Check if paths are valid
    if not os.path.exists(LABELS_CSV_PATH) or not os.path.exists(MESH_DIR):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Please update LABELS_CSV_PATH and MESH_DIR before running! !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()


    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    print(f"Using {NUM_WORKERS} workers. Voxel Resolution: {VOXEL_RESOLUTION}. Weighted Loss: {USE_WEIGHTED_LOSS}. L1 Lambda: {L1_LAMBDA}")

    datamodule = JomonKaenVoxelDataModule(labels_csv_path=LABELS_CSV_PATH, mesh_dir=MESH_DIR, cache_dir=CACHE_DIR, feature_columns=FEATURE_COLUMNS, voxel_resolution=VOXEL_RESOLUTION, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, use_weighted_loss=USE_WEIGHTED_LOSS)
    datamodule.setup('fit')

    model = VoxelNetLightningModule(learning_rate=LEARNING_RATE_INITIAL, lr_final=LEARNING_RATE_FINAL, num_outputs=NUM_OUTPUTS, pos_weight=datamodule.pos_weight, resolution=VOXEL_RESOLUTION, l1_lambda=L1_LAMBDA, use_weighted_loss=USE_WEIGHTED_LOSS, warmup_start_factor=0.01, warmup_total_iters=100)

    # Note: torchinfo.summary may not work perfectly with spconv models.
    # We can try to summarize, but it might throw an error or give incomplete info.
    try:
        torchinfo.summary(model, input_size=(BATCH_SIZE, 3, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION))
    except Exception as e:
        print(f"\nCould not generate model summary with torchinfo: {e}")
        print("This is common for sparse models. Continuing with training.\n")


    if VISUALIZE_SAMPLES:
        print("\nVisualizing Data Samples (Close window to continue)")
        num_samples_to_show = 3
        if len(datamodule.train_dataset) > 0:
            for i in range(min(num_samples_to_show, len(datamodule.train_dataset))):
                dense_voxel_tensor, label = datamodule.train_dataset[i]
                voxel_coords = torch.nonzero(dense_voxel_tensor.sum(dim=0) > 0.0, as_tuple=False)
                if voxel_coords.numel() == 0:
                    print(f"Sample {i+1}/{num_samples_to_show} is empty/failed. Skipping.")
                    continue
                colors = dense_voxel_tensor[:, voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]].T
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(voxel_coords.numpy().astype(float))
                pcd.colors = o3d.utility.Vector3dVector(colors.numpy())
                print(f"Showing sample {i+1}/{num_samples_to_show}")
                o3d.visualization.draw_geometries([pcd], window_name=f"Voxelized Sample {i+1}")
        else:
            print("Skipping visualization because the training dataset is empty.")

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints_voxelnext/', filename='voxelnext-best-{epoch:02d}-{val_loss:.4f}', save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=1000, verbose=True, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    prediction_saver = SaveValidationPredictions(output_dir='eval_prediction_voxelnext', feature_columns=FEATURE_COLUMNS, save_every_n_epochs=100)

    trainer = pl.Trainer(accelerator='gpu', devices=1, precision="16-mixed", max_epochs=MAX_EPOCHS, callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, prediction_saver], log_every_n_steps=10)

    print("\nStarting Training")
    trainer.fit(model, datamodule=datamodule)
    print("Training Finished.")
