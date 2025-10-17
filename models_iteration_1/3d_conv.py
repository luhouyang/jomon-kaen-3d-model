import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
import open3d as o3d
import numpy as np
import torchinfo
import trimesh


# This function is taken from the main repository: https://github.com/arch-inform-kaken-group/jomon-kaen-3d-heatmap/blob/main/src/dataset/processing/pottery.py
def voxelize_pottery_dogu(input_file, target_voxel_resolution):
    """
    Converts a 3D mesh file into a voxelized point cloud representation.
    
    Args:
        input_file: Path to the input mesh file (.glb format)
        target_voxel_resolution: Target resolution for voxelization (e.g., 64, 80)
    
    Returns:
        tuple: (voxel_pcd, min_bound, voxel_size) or (None, None, None) on error
    """
    try:
        scene = trimesh.load(str(input_file), force="scene")
        if not scene.geometry: return None, None, None
        mesh_trimesh = trimesh.util.concatenate(list(scene.geometry.values()))
        if mesh_trimesh.vertices.shape[0] == 0 or mesh_trimesh.faces.shape[
                0] == 0:
            return None, None, None
        vertex_color_trimesh = mesh_trimesh.visual.to_color().vertex_colors
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
            vertex_color_trimesh[:, :3] / 255.0)
        min_bound, max_bound = mesh_o3d.get_min_bound(
        ), mesh_o3d.get_max_bound()
        max_range = np.max(max_bound - min_bound) + 1e-6
        voxel_size = max_range / (target_voxel_resolution - 1)
        voxel_size_sq = voxel_size**2
        mesh_vertices_np, mesh_vertex_colors_np, mesh_triangles_np = np.asarray(
            mesh_o3d.vertices), np.asarray(mesh_o3d.vertex_colors), np.asarray(
                mesh_o3d.triangles)
        tri_vertices, tri_colors = mesh_vertices_np[
            mesh_triangles_np], mesh_vertex_colors_np[mesh_triangles_np]
        v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
        triangle_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0,
                                                       v2 - v0),
                                              axis=1)
        num_samples_per_triangle = np.ceil(
            triangle_areas / voxel_size_sq).astype(int) + 20
        total_samples = np.sum(num_samples_per_triangle)
        if total_samples == 0: return None, None, None
        triangle_indices = np.repeat(np.arange(len(mesh_triangles_np)),
                                     num_samples_per_triangle)
        r = np.random.rand(total_samples, 2)
        r_sum = np.sum(r, axis=1)
        r[r_sum > 1] = 1 - r[r_sum > 1]
        bary_coords = np.zeros((total_samples, 3))
        bary_coords[:, [1, 2]] = r
        bary_coords[:, 0] = 1 - np.sum(r, axis=1)
        all_sample_points = np.einsum('ij,ijk->ik',
                                      bary_coords,
                                      tri_vertices[triangle_indices])
        all_interp_colors = np.einsum('ij,ijk->ik',
                                      bary_coords,
                                      tri_colors[triangle_indices])
        voxel_coords_all = np.floor(
            (all_sample_points - min_bound) / voxel_size).astype(int)
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
        print(f"Error processing file {input_file}: {e}")
        return None, None, None


class VoxelDataset(Dataset):
    """
    PyTorch Dataset for loading and voxelizing 3D pottery meshes.
    Implements caching to avoid re-voxelizing meshes on subsequent runs.
    """

    def __init__(self,
                 data,
                 labels_csv_path,
                 voxel_resolution,
                 cache_dir,
                 feature_columns):
        super().__init__()
        self.data, self.labels_df = data, pd.read_csv(labels_csv_path)
        self.voxel_resolution, self.cache_dir = voxel_resolution, cache_dir
        self.feature_columns = feature_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_info = self.data[index]
        mesh_file_path = item_info['mesh_path']
        original_filename = os.path.basename(mesh_file_path)
        base_name = os.path.splitext(original_filename)[0]

        # Construct the .ply filename for the CSV lookup
        lookup_name = f"{base_name}.ply"

        cached_ply_path = os.path.join(self.cache_dir, f"{base_name}.ply")
        cached_meta_path = os.path.join(self.cache_dir, f"{base_name}.npz")
        voxel_pcd, min_bound, voxel_size = None, None, None

        # Check for cached files, delete the voxel_cache if planning to change voxel resolution
        if os.path.exists(cached_ply_path) and os.path.exists(
                cached_meta_path):
            try:
                voxel_pcd = o3d.io.read_point_cloud(cached_ply_path)
                meta = np.load(cached_meta_path)
                min_bound, voxel_size = meta['min_bound'], meta['voxel_size']
            except Exception:
                voxel_pcd = None

        # If not cached, voxelize and cache the result
        if voxel_pcd is None:
            voxel_pcd, min_bound, voxel_size = voxelize_pottery_dogu(
                mesh_file_path, self.voxel_resolution)
            if voxel_pcd is not None and voxel_pcd.has_points():
                o3d.io.write_point_cloud(cached_ply_path, voxel_pcd)
                np.savez(cached_meta_path,
                         min_bound=min_bound,
                         voxel_size=voxel_size)

        # Create dense voxel grid (RGB_CHANNEL 3, X, Y, Z) for PyTorch channel-first inputs
        dense_voxel_tensor = torch.zeros((3,
                                          self.voxel_resolution,
                                          self.voxel_resolution,
                                          self.voxel_resolution),
                                         dtype=torch.float32)
        if voxel_pcd is not None and voxel_pcd.has_points():
            points, colors = np.asarray(voxel_pcd.points), np.asarray(
                voxel_pcd.colors).astype(np.float32)
            indices = np.floor((points - min_bound) / voxel_size).astype(int)
            indices = np.clip(indices, 0, self.voxel_resolution - 1)
            Dx, Dy, Dz = indices[:, 0], indices[:, 1], indices[:, 2]
            Cr, Cg, Cb = colors[:, 0], colors[:, 1], colors[:, 2]
            dense_voxel_tensor[0, Dx, Dy, Dz] = torch.from_numpy(Cr)
            dense_voxel_tensor[1, Dx, Dy, Dz] = torch.from_numpy(Cg)
            dense_voxel_tensor[2, Dx, Dy, Dz] = torch.from_numpy(Cb)

        # Load classification labels from CSV
        target_row = self.labels_df[self.labels_df['CODE'] == lookup_name]
        if target_row.empty:
            raise (ValueError(
                f"WARNING: No label found for '{lookup_name}' in the CSV."))
        else:
            target_values = target_row[self.feature_columns].apply(
                pd.to_numeric,
                errors='coerce').fillna(0).values

        target_tensor = torch.from_numpy(target_values).squeeze()
        return dense_voxel_tensor, target_tensor


class JomonKaenVoxelDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for managing train/validation splits,
    data loading, and optional weighted loss calculation.
    """

    def __init__(self,
                 labels_csv_path,
                 mesh_dir,
                 cache_dir,
                 feature_columns,
                 voxel_resolution,
                 batch_size=8,
                 num_workers=4,
                 use_weighted_loss=False):
        super().__init__()
        self.save_hyperparameters()
        self.pos_weight = None

    def setup(self, stage=None):
        os.makedirs(self.hparams.cache_dir, exist_ok=True)
        print(
            f"Voxel cache is located at: {os.path.abspath(self.hparams.cache_dir)}"
        )

        # Look for original .glb files
        all_files = [{
            'mesh_path': os.path.join(self.hparams.mesh_dir,
                                      p),
            'POTTERY': p
        } for p in os.listdir(self.hparams.mesh_dir) if p.endswith('.glb')]

        if not all_files:
            raise FileNotFoundError(
                f"'.glb' files found in the directory '{os.path.abspath(self.hparams.mesh_dir)}'. Please check the path."
            )

        np.random.shuffle(all_files)

        # Define test set (held-out samples for final evaluation)
        test_groups = {
            "IN0009(5).glb",
            "NM0049(30).glb",
            "UD0005(71).glb",
            "NM0015(27).glb",
            "NM0066(31).glb",
            "NM0099(37).glb",
            "NM0154(42).glb",
            "NM0168(45).glb",
            "SI0001(53).glb",
            "SJ0503(54).glb"
        }

        common_params = {
            "labels_csv_path": self.hparams.labels_csv_path,
            "voxel_resolution": self.hparams.voxel_resolution,
            "cache_dir": self.hparams.cache_dir,
            "feature_columns": self.hparams.feature_columns
        }

        # Split data into train and validation sets
        train_data, val_data = [
            d for d in all_files if d['POTTERY'] not in test_groups
        ], [d for d in all_files if d['POTTERY'] in test_groups]

        if not train_data:
            print(
                "WARNING: The training dataset is empty. This might be because all found .glb files are part of the test_groups."
            )

        self.train_dataset = VoxelDataset(train_data, **common_params)
        self.val_dataset = VoxelDataset(val_data, **common_params)

        # Calculate positive class weights for handling class imbalance
        if stage in ('fit', None) and self.hparams.use_weighted_loss:
            print("Calculating positive class weights for weighted loss.")
            # Use the .ply names for lookup in the CSV
            train_files_for_lookup = [
                f"{os.path.splitext(d['POTTERY'])[0]}.ply"
                for d in self.train_dataset.data
            ]
            all_labels_df = pd.read_csv(self.hparams.labels_csv_path)
            train_labels_df = all_labels_df[all_labels_df['CODE'].isin(
                train_files_for_lookup)]
            labels = train_labels_df[self.hparams.feature_columns].apply(
                pd.to_numeric,
                errors='coerce').fillna(0).values
            num_samples, num_positives = len(labels), np.sum(labels, axis=0)
            num_negatives = num_samples - num_positives
            pos_weight_np = num_negatives / (num_positives + 1e-6)
            self.pos_weight = torch.tensor(pos_weight_np.astype(np.float32))
        elif stage in ('fit', None):
            print("Weighted loss is disabled.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True)


class Conv3DNetwork(nn.Module):
    """
    3D Convolutional Neural Network for voxel-based classification.
    Architecture: 4 conv blocks with batch norm, ReLU, and max pooling,
    followed by fully connected layers with dropout.
    """

    def __init__(self, num_outputs, resolution=64):
        super().__init__()
        # Calculate flattened dimension after 4 max pooling layers (each divides by 2)
        final_size = resolution // (2**4)
        flattened_dim = 128 * (final_size**3)

        self.net = nn.Sequential(
            # Conv Block 1: 3 -> 16 channels
            nn.Conv3d(3,
                      16,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            # Conv Block 2: 16 -> 32 channels
            nn.Conv3d(16,
                      32,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            # Conv Block 3: 32 -> 64 channels
            nn.Conv3d(32,
                      64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            # Conv Block 4: 64 -> 128 channels
            nn.Conv3d(64,
                      128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),

            # Fully connected layers
            nn.Flatten(),
            nn.Linear(flattened_dim,
                      256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,
                      num_outputs))

    def forward(self, x):
        return self.net(x)


class SaveValidationPredictions(Callback):
    """
    PyTorch Lightning callback to save validation predictions to CSV files.
    Saves predictions at regular intervals and when best validation loss is achieved.
    """

    def __init__(self, output_dir, feature_columns, save_every_n_epochs=10):
        super().__init__()
        self.output_dir, self.save_every_n_epochs = output_dir, save_every_n_epochs
        os.makedirs(self.output_dir, exist_ok=True)
        self.best_val_loss = float('inf')
        self.headers = feature_columns

    def on_validation_epoch_start(self, trainer, pl_module):
        # Initialize lists to collect predictions and targets
        self.preds, self.targets = [], []

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx=0):
        # Collect predictions and targets from each batch
        if isinstance(outputs,
                      dict) and 'preds' in outputs and 'targets' in outputs:
            self.preds.append(outputs['preds'].cpu())
            self.targets.append(outputs['targets'].cpu())

    def _save_to_file(self, pl_module, filename):
        """Helper method to save predictions and ground truth to CSV"""
        all_preds, all_targets = torch.cat(
            self.preds).int().numpy(), torch.cat(self.targets).int().numpy()
        df_preds, df_targets = pd.DataFrame(
            all_preds,
            columns=self.headers), pd.DataFrame(all_targets,
                                                columns=self.headers)
        df_preds['source'], df_targets['source'] = 'prediction', 'ground_truth'
        df_preds['sample_id'], df_targets['sample_id'] = range(
            len(df_preds)), range(len(df_targets))
        df_combined = pd.concat([df_targets,
                                 df_preds],
                                ignore_index=True).sort_values(
                                    by=['sample_id',
                                        'source'],
                                    ascending=[True,
                                               False])
        df_combined = df_combined[['sample_id', 'source'] + self.headers]
        filepath = os.path.join(self.output_dir, filename)
        df_combined.to_csv(filepath, index=False)
        pl_module.print(f"\nSaved validation predictions to {filepath}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self.preds: return
        epoch, current_val_loss = trainer.current_epoch + 1, trainer.callback_metrics.get(
            'val_loss')
        is_interval_save, is_best_loss_save = (
            epoch > 0 and epoch % self.save_every_n_epochs
            == 0), (current_val_loss is not None
                    and current_val_loss < self.best_val_loss)

        # Save at regular intervals
        if is_interval_save:
            self._save_to_file(pl_module,
                               filename=f'eval_predictions_epoch_{epoch}.csv')

        # Save when best validation loss is achieved
        if is_best_loss_save:
            self.best_val_loss = current_val_loss.item()
            self._save_to_file(pl_module, filename='eval_predictions_best.csv')
            if not is_interval_save:
                self._save_to_file(
                    pl_module,
                    filename=f'eval_predictions_epoch_{epoch}_best_loss.csv')


class VoxelNetLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module wrapping the 3D CNN model.
    Handles training loop, validation, optimization, and optional L1 regularization.
    """

    def __init__(self,
                 learning_rate=1e-3,
                 lr_final=1e-6,
                 num_outputs=13,
                 pos_weight=None,
                 resolution=64,
                 l1_lambda=0.0,
                 use_weighted_loss=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = Conv3DNetwork(num_outputs=self.hparams.num_outputs,
                                   resolution=self.hparams.resolution)
        # Binary cross-entropy loss for multi-label classification
        # if self.hparams.use_weighted_loss:
        #     self.criterion = nn.BCEWithLogitsLoss(
        #         pos_weight=self.hparams.pos_weight)
        # else:
        #     self.criterion = nn.BCEWithLogitsLoss()

        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.hparams.pos_weight)

    def forward(self, x):
        return self.model(x)

    def compute_l1_loss(self):
        """
        Compute L1 regularization term (sum of absolute values of all parameters).
        L1 regularization encourages sparsity in model weights.
        """
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return l1_norm

    def _shared_step(self, batch):
        """Shared logic for training and validation steps"""
        inputs, targets = batch
        outputs = self.forward(inputs)

        # Compute base loss (BCE)
        loss = self.criterion(outputs, targets)

        # Add L1 regularization if enabled (l1_lambda > 0)
        if self.hparams.l1_lambda > 0:
            l1_loss = self.compute_l1_loss()
            loss = loss + self.hparams.l1_lambda * l1_loss

        # Compute accuracy (convert logits to binary predictions)
        preds = torch.sigmoid(outputs) > 0.5
        accuracy = (preds == targets.bool()).float().mean()

        return loss, accuracy, preds, targets

    def training_step(self, batch, batch_idx):
        loss, accuracy, _, _ = self._shared_step(batch)

        # Log metrics to progress bar and logger
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)
        self.log('train_acc',
                 accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        # Log L1 loss separately if enabled
        if self.hparams.l1_lambda > 0:
            l1_loss = self.compute_l1_loss()
            self.log('train_l1_loss',
                     l1_loss,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, preds, targets = self._shared_step(batch)

        # Log validation metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)

        # Log L1 loss separately if enabled
        if self.hparams.l1_lambda > 0:
            l1_loss = self.compute_l1_loss()
            self.log('val_l1_loss', l1_loss, on_epoch=True, prog_bar=False)

        # Return predictions for SaveValidationPredictions callback
        return {'preds': preds, 'targets': targets}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)

        # # Cosine annealing: gradually decreases learning rate from initial to final
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.trainer.max_epochs,
        #     eta_min=self.hparams.lr_final)

        # Linear warmup for the first 100 epochs (adjust total_iters as needed)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                             start_factor=0.01,
                                                             total_iters=100)

        # Then, cosine annealing for the rest of the training
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - 100)

        # Combine the schedulers using SequentialLR for seamless transition
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler,
                        main_scheduler],
            milestones=[100])

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == '__main__':
    # ========== FEATURE COLUMNS ==========
    # Define the output labels for multi-label classification
    FEATURE_COLUMNS = [
        'HAS_FLAME_LIKE_DECORATION',
        'HAS_CROWN_LIKE_DECORATION',
        'HAS_HANDLES',
        'HAS_CORD_MARKED_PATTERN',
        'HAS_NAIL_ENGRAVING',
        'HAS_SPIRAL_PATTERN',
        'NUMBER_OF_PERTRUSIONS_0.0',
        'NUMBER_OF_PERTRUSIONS_1.0',
        'NUMBER_OF_PERTRUSIONS_2.0',
        'NUMBER_OF_PERTRUSIONS_3.0',
        'NUMBER_OF_PERTRUSIONS_4.0',
        'NUMBER_OF_PERTRUSIONS_6.0',
        'NUMBER_OF_PERTRUSIONS_8.0'
    ]
    NUM_OUTPUTS = len(FEATURE_COLUMNS)

    # ========== HYPERPARAMETERS ==========
    VOXEL_RESOLUTION = 80  # Resolution of voxel grid (80x80x80)
    BATCH_SIZE = 8
    MAX_EPOCHS = 1000
    NUM_WORKERS = 8  # Number of parallel data loading workers
    LEARNING_RATE_INITIAL = 1e-3
    LEARNING_RATE_FINAL = 1e-6
    NUM_OUTPUTS = 13
    USE_WEIGHTED_LOSS = True  # Weight loss by class frequency
    VISUALIZE_SAMPLES = False  # Show sample voxelizations before training

    # L1 Regularization parameter
    # Set to 0.0 to disable, or try values like 1e-5, 1e-4, 1e-3
    # Higher values = stronger regularization (more sparse weights)
    L1_LAMBDA = 1e-4  # Default: no L1 regularization

    # ========== PATHS ==========
    LABELS_CSV_PATH = r"C:\Users\User\Desktop\Python\jomon-kaen-3d-model\DS_Labels_Cleaned.csv"
    MESH_DIR = r"D:\storage\jomon_kaen\pottery_only"
    CACHE_DIR = r"voxel_cache"

    # ========== SETUP ==========
    pl.seed_everything(42)  # Set random seed for reproducibility
    torch.set_float32_matmul_precision(
        'high')  # Use TensorFloat-32 for faster training
    print(
        f"Using {NUM_WORKERS} workers. Voxel Resolution: {VOXEL_RESOLUTION}. "
        f"Weighted Loss: {USE_WEIGHTED_LOSS}. L1 Lambda: {L1_LAMBDA}")

    # ========== DATAMODULE AND MODEL ==========
    # Initialize data module (handles data loading and preprocessing)
    datamodule = JomonKaenVoxelDataModule(labels_csv_path=LABELS_CSV_PATH,
                                          mesh_dir=MESH_DIR,
                                          cache_dir=CACHE_DIR,
                                          feature_columns=FEATURE_COLUMNS,
                                          voxel_resolution=VOXEL_RESOLUTION,
                                          batch_size=BATCH_SIZE,
                                          num_workers=NUM_WORKERS,
                                          use_weighted_loss=USE_WEIGHTED_LOSS)
    datamodule.setup('fit')

    # Initialize model with L1 regularization
    model = VoxelNetLightningModule(learning_rate=LEARNING_RATE_INITIAL,
                                    lr_final=LEARNING_RATE_FINAL,
                                    num_outputs=NUM_OUTPUTS,
                                    pos_weight=datamodule.pos_weight,
                                    resolution=VOXEL_RESOLUTION,
                                    l1_lambda=L1_LAMBDA,
                                    use_weighted_loss=USE_WEIGHTED_LOSS)

    print(
        f"\nModel configured with {NUM_OUTPUTS} outputs and L1 lambda={L1_LAMBDA}."
    )
    torchinfo.summary(model,
                      input_size=(BATCH_SIZE,
                                  3,
                                  VOXEL_RESOLUTION,
                                  VOXEL_RESOLUTION,
                                  VOXEL_RESOLUTION))

    # ========== VISUALIZATION BLOCK ==========
    # Optionally visualize some training samples before training
    if VISUALIZE_SAMPLES:
        print("\nVisualizing Data Samples (Close window to continue)")
        num_samples_to_show = 5
        if len(datamodule.train_dataset) > 0:
            for i in range(
                    min(num_samples_to_show,
                        len(datamodule.train_dataset))):
                dense_voxel_tensor, label = datamodule.train_dataset[i]
                # Find occupied voxels (where color sum > 0)
                voxel_coords = torch.nonzero(dense_voxel_tensor.sum(dim=0)
                                             > 0.0,
                                             as_tuple=False)
                if voxel_coords.numel() == 0:
                    print(
                        f"Sample {i+1}/{num_samples_to_show} is empty/failed. Skipping."
                    )
                    continue
                colors = dense_voxel_tensor[:,
                                            voxel_coords[:,
                                                         0],
                                            voxel_coords[:,
                                                         1],
                                            voxel_coords[:,
                                                         2]].T
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    voxel_coords.numpy().astype(float))
                pcd.colors = o3d.utility.Vector3dVector(colors.numpy())
                print(f"Showing sample {i+1}/{num_samples_to_show}")
                o3d.visualization.draw_geometries(
                    [pcd],
                    window_name=f"Voxelized Sample {i+1}")
        else:
            print(
                "Skipping visualization because the training dataset is empty."
            )

    # ========== CALLBACKS & TRAINER ==========
    # Callback to save best model checkpoint based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints_3dcnn/',
        filename='conv3d-best-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min')

    # Early stopping if validation loss doesn't improve
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=1000,
                                        verbose=True,
                                        mode='min')

    # Monitor learning rate changes
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Save validation predictions to CSV files
    prediction_saver = SaveValidationPredictions(
        output_dir='eval_predictions_3dcnn',
        feature_columns=FEATURE_COLUMNS,
        save_every_n_epochs=10)

    # Initialize trainer with GPU acceleration and mixed precision
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         precision="16-mixed",  # Use mixed precision for faster training
                         max_epochs=MAX_EPOCHS,
                         callbacks=[
                             checkpoint_callback,
                             early_stop_callback,
                             lr_monitor,
                             prediction_saver
                         ],
                         log_every_n_steps=10)

    # ========== TRAINING ==========
    print("\nStarting Training")
    trainer.fit(model, datamodule=datamodule)
    print("Training Finished.")
