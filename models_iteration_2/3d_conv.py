import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
        vertex_color_trimesh = mesh_trimesh.visual.to_color().vertex_colors

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
        mesh_vertex_colors_np = np.asarray(mesh_o3d.vertex_colors)
        mesh_triangles_np = np.asarray(mesh_o3d.triangles)

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
        all_interp_colors = np.einsum('ij,ijk->ik', bary_coords, tri_colors[triangle_indices])

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


# yapf: disable
class VoxelDataset(Dataset):
    def __init__(
        self,
        data,
        labels_csv_path,
        voxel_resolution,
        cache_dir,
        feature_columns,
        augment_color_p=0.5,  # Probability of applying color augmentation
        color_jitter_std=0.05, # Standard deviation for the random color noise
        jitter_voxel_p=0.1,    # Probability of jittering an *individual* voxel's color
    ):
        super().__init__()
        self.data = data
        self.labels_df = pd.read_csv(labels_csv_path)
        self.voxel_resolution = voxel_resolution
        self.cache_dir = cache_dir
        self.feature_columns = feature_columns

        # Augmentation parameters
        self.augment_color_p = augment_color_p
        self.color_jitter_std = color_jitter_std
        self.jitter_voxel_p = jitter_voxel_p

    def __len__(self):
        return len(self.data)

    # --- Helper function for color augmentation ---
    def color_jitter_pcd(self, voxel_pcd):
        if np.random.rand() < self.augment_color_p:
            colors = np.asarray(voxel_pcd.colors).astype(np.float32)
            
            # 1. Create a boolean mask for voxels to augment
            num_points = colors.shape[0]
            jitter_mask = np.random.rand(num_points) < self.jitter_voxel_p
            
            if np.any(jitter_mask):
                # 2. Generate random noise (Gaussian)
                noise = np.random.normal(0.0, self.color_jitter_std, colors[jitter_mask].shape).astype(np.float32)
                
                # 3. Apply noise only to selected voxels
                colors[jitter_mask] += noise
                
                # 4. Clip colors to [0, 1] range
                colors = np.clip(colors, 0.0, 1.0)
                
                # Update the point cloud's colors
                voxel_pcd.colors = o3d.utility.Vector3dVector(colors)
        return voxel_pcd

    def __getitem__(self, index):
        item_info = self.data[index]
        mesh_file_path = item_info['mesh_path']
        original_filename = os.path.basename(mesh_file_path) # Has extension of .glb
        base_name = os.path.splitext(original_filename)[0] # Like AS0001(1)

        lookup_name = f"{base_name}.ply" # The voxelized pottery files

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

        dense_voxel_tensor = torch.zeros((3,
                                        self.voxel_resolution,
                                        self.voxel_resolution,
                                        self.voxel_resolution),
                                    dtype=torch.float32)

        # Prepare input for 3D Conv operations, group by channels
        if voxel_pcd is not None and voxel_pcd.has_points():
            points, colors = np.asarray(voxel_pcd.points), np.asarray(voxel_pcd.colors).astype(np.float32)
            indices = np.floor((points - min_bound) / voxel_size).astype(int)
            indices = np.clip(indices, 0, self.voxel_resolution - 1)
            Dx, Dy, Dz = indices[:, 0], indices[:, 1], indices[:, 2]
            Cr, Cg, Cb = colors[:, 0], colors[:, 1], colors[:, 2]
            dense_voxel_tensor[0, Dx, Dy, Dz] = torch.from_numpy(Cr)
            dense_voxel_tensor[1, Dx, Dy, Dz] = torch.from_numpy(Cg)
            dense_voxel_tensor[2, Dx, Dy, Dz] = torch.from_numpy(Cb)

        target_row = self.labels_df[self.labels_df['CODE'] == lookup_name]
        if target_row.empty:
            raise(ValueError(
                f"WARNING: No labels found for '{lookup_name}' in the CSV."
            ))
        else:
            target_values = target_row[self.feature_columns].apply(
                pd.to_numeric,
                errors='coerce'
            ).fillna(0).values

        target_tensor = torch.from_numpy(target_values).squeeze()
        return dense_voxel_tensor, target_tensor
# yapf: enable
 

class JomonKaenVoxelDataModule(pl.LightningDataModule):

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

        # Look for original .glb pottery fiels
        all_files = [{
            'mesh_path': os.path.join(self.hparams.mesh_dir,
                                      p),
            'POTTERY': p
        } for p in os.listdir(self.hparams.mesh_dir) if p.endswith('.glb')]

        if not all_files:
            raise FileNotFoundError(
                f"'.glb' files not found in the directory '{os.path.abspath(self.hparams.mesh_dir)}'"
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

        # test_groups = {
        #     "IN0008(4).ply", "IN0009(5).ply", "IN0081(7).ply", 
        #     "IN0220(11).ply", "IN0239(14).ply", "MY0007(20).ply", 
        #     "NM0002(23).ply", "NM0010(25).ply", "NM0041(29).ply", 
        #     "NM0049(30).ply", "NM0079(35).ply", "NM0099(37).ply", 
        #     "NM0144(41).ply", "NM0168(45).ply", "NM0173(46).ply", 
        #     "NM0191(49).ply", "SB0002(51).ply", "SI0001(53).ply", 
        #     "SK0001(56).ply", "SK0004(59).ply", "UD0003(70).ply", 
        #     "UD0005(71).ply", "UD0016(76).ply", "UD0302(78).ply", 
        #     "UD0318(81).ply"
        # }

        common_params = {
            "labels_csv_path": self.hparams.labels_csv_path,
            "voxel_resolution": self.hparams.voxel_resolution,
            "cache_dir": self.hparams.cache_dir,
            "feature_columns": self.hparams.feature_columns
        }

        # Split data into train and validation sets
        train_data = [d for d in all_files if d['POTTERY'] not in test_groups]
        val_data = [d for d in all_files if d['POTTERY'] in test_groups]

        if not train_data:
            print("WARNING: The training dataset is empty. Might be because all pottery found is in test group!")

        self.train_dataset = VoxelDataset(train_data, **common_params)
        self.val_dataset = VoxelDataset(val_data, **common_params)

        # Calculate positive class weights for handling class imbalance
        if stage in ('fit', None) and self.hparams.use_weighted_loss:
            print("Calculating positive class weights for weighted loss")
            train_files_for_lookup = [
                f"{os.path.splitext(d["POTTERY"])[0]}.ply"
                for d in self.train_dataset.data
            ]
            all_labels_df = pd.read_csv(self.hparams.labels_csv_path)
            train_labels_df = all_labels_df[all_labels_df['CODE'].isin(train_files_for_lookup)] # Because the CODE column is "POTTERY.ply"
            labels = train_labels_df[self.hparams.feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values 
            num_samples, num_positives = len(labels), np.sum(labels, axis=0)
            num_negatives = num_samples - num_positives
            pos_weight_np = num_negatives / (num_positives + 1e-6)
            self.pos_weight = torch.tensor(pos_weight_np.astype(np.float32))
        elif stage in ('fit', None):
            print("Weighted loss is disabled")

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        batch_size=self.hparams.batch_size,
                        shuffle=True,
                        num_workers=self.hparams.num_workers,
                        pin_memory=True,
                        persistent_workers=True,     
                )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                        batch_size=self.hparams.batch_size,
                        shuffle=False,
                        num_workers=self.hparams.num_workers,
                        pin_memory=True,
                        persistent_workers=True,     
                )

class Conv3DNetwork(nn.Module):
    def __init__(self, num_outputs, conv_dims, resolution=80):
        super().__init__()
        # Calculate flattened dimension after 4 max pooling layers (each divides by 2)
        final_size = resolution // (2**(len(conv_dims) - 1))
        flatten_dim = conv_dims[-1] * (final_size**3)

        self.feature_extraction = nn.Sequential(
            *[self.conv_block(in_dim, conv_dims[i+1]) for i, in_dim in enumerate(conv_dims[:-1])]
        )

        self.flatten = nn.Flatten()

        self.output_head = nn.Sequential(
            nn.Linear(flatten_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, num_outputs)
        )
    
    def conv_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    # def conv_block(self, in_dim, out_dim):
    #     return nn.Sequential(
    #         nn.Conv3d(in_dim, out_dim, kernel_size=3, padding=1),
    #         nn.BatchNorm3d(out_dim),
    #         nn.ReLU(inplace=True),
    #         nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1),
    #         nn.BatchNorm3d(out_dim),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool3d(2),
    #     )
        
    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.flatten(x)
        return self.output_head(x)


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
        df_combined = pd.concat([df_targets, df_preds], ignore_index=[True, False]).sort_values(by=['sample_id', 'source'], ascending=[True, False])
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
            self._save_to_file(pl_module, filename=f"eval_predictions_spoch_{epoch}.csv")

        if is_best_loss_save:
            self.best_val_loss = current_val_loss.item()
            self._save_to_file(pl_module, filename="eval_prediction_best.csv")
            if not is_interval_save:
                self._save_to_file(
                    pl_module,
                    filename=f"eval_predictions_epoch_{epoch}_best_loss.csv"
                )

class VoxelNetLightningModule(pl.LightningModule):
    def __init__(self,
                learning_rate=1e-3,
                lr_final=1e-6,
                num_outputs=13,
                pos_weight=None,
                resolution=80,
                l1_lambda=0.0,
                use_weighted_loss=False,
                warmup_start_factor=0.01,
                warmup_total_iters=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = Conv3DNetwork(num_outputs=self.hparams.num_outputs,
                                resolution=self.hparams.resolution,
                                conv_dims=[3, 4, 8, 16, 32, 64, 64]
                                )

        if self.hparams.use_weighted_loss:
            print("Using weighted loss in criterion")
            print(self.hparams.pos_weight)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                            start_factor=self.hparams.warmup_start_factor, 
                                                            total_iters=self.hparams.warmup_total_iters)

        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - self.hparams.warmup_total_iters,
            eta_min=self.hparams.lr_final
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.hparams.warmup_total_iters]
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
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

    VOXEL_RESOLUTION = 256
    BATCH_SIZE = 4
    MAX_EPOCHS = 1000
    NUM_WORKERS = 4
    LEARNING_RATE_INITIAL = 1e-3
    LEARNING_RATE_FINAL = 1e-6
    NUM_OUTPUTS = 13
    USE_WEIGHTED_LOSS = False
    VISUALIZE_SAMPLES = False

    L1_LAMBDA = 1e-4

    LABELS_CSV_PATH = r"C:\Users\User\Desktop\Python\jomon-kaen-3d-model\DS_Labels_Cleaned.csv"
    MESH_DIR = r"D:\storage\jomon_kaen\pottery_only"
    CACHE_DIR = r"voxel_cache"

    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    print(
        f"Using {NUM_WORKERS} workers. Voxel Resolution: {VOXEL_RESOLUTION}. "
        f"Weighted Loss: {USE_WEIGHTED_LOSS}. L1 Lambda: {L1_LAMBDA}")

    datamodule = JomonKaenVoxelDataModule(
        labels_csv_path=LABELS_CSV_PATH,
        mesh_dir=MESH_DIR,
        cache_dir=CACHE_DIR,
        feature_columns=FEATURE_COLUMNS,
        voxel_resolution=VOXEL_RESOLUTION,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_weighted_loss=USE_WEIGHTED_LOSS
    )
    datamodule.setup('fit')

    model = VoxelNetLightningModule(
        learning_rate=LEARNING_RATE_INITIAL,
        lr_final=LEARNING_RATE_FINAL,
        num_outputs=NUM_OUTPUTS,
        pos_weight=datamodule.pos_weight,
        resolution=VOXEL_RESOLUTION,
        l1_lambda=L1_LAMBDA,
        use_weighted_loss=USE_WEIGHTED_LOSS,
        warmup_start_factor=0.01,
        warmup_total_iters=100
    )

    torchinfo.summary(
        model,
        input_size=(BATCH_SIZE, 3, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
    )

    if VISUALIZE_SAMPLES:
        print("\nVisualizing Data Samples (Close window to continue)")
        num_samples_to_show = 3
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
                                            voxel_coords[:, 0],
                                            voxel_coords[:, 1],
                                            voxel_coords[:, 2]].T
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

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints_3cnn/',
        filename='conv3d-best-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=1000,
                                        verbose=True,
                                        mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    prediction_saver = SaveValidationPredictions(
        output_dir='eval_prediction_3dcnn',
        feature_columns=FEATURE_COLUMNS,
        save_every_n_epochs=100
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision="16-mixed",
        max_epochs=MAX_EPOCHS,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            prediction_saver
        ],
        log_every_n_steps=10
    )

    print("\nStarting Training")
    trainer.fit(model, datamodule=datamodule)
    print("Training Finished.")
