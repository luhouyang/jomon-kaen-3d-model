import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
import open3d as o3d
import numpy as np
import torchinfo

# This import is assumed to contain the PreprocessJomonKaenDataset class
from models_iteration_1.dataset import PreprocessJomonKaenDataset

# =================================================================================
# MODEL DEFINITIONS: T-NET and POINTNET-TRANSFORMER
# =================================================================================


class TNet(nn.Module):
    """A Transformation Network (T-Net) that learns a KxK transformation matrix."""

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).view(
            1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetTransformerRegressorUnifiedInput(nn.Module):
    """
    A PointNet model with a Transformer encoder layer that processes a single
    unified point cloud input.
    """

    def __init__(self,
                 num_outputs=13,
                 feature_dim=125,
                 nhead=4,
                 num_encoder_layers=2,
                 dim_feedforward=128):
        super().__init__()

        # Single Input Stream
        self.input_tnet = TNet(k=6)
        self.conv1 = nn.Conv1d(6, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.feature_tnet = TNet(k=32)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, feature_dim, 1)
        self.bn3 = nn.BatchNorm1d(feature_dim)

        # Smaller Transformer Encoder
        self.embedding_dim = feature_dim + 3  # Add XYZ back
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,  # Expects (N, B, E)
            dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        # MLP Head
        self.fc1 = nn.Linear(self.embedding_dim, 128)
        self.bn1_head = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.bn2_head = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_outputs)

    def forward(self, inputs):
        # 1. Input Transformation
        inputs_permuted = inputs.permute(0, 2, 1)  # (B, 6, N)
        transform = self.input_tnet(inputs_permuted)
        x_transformed = torch.bmm(inputs, transform).permute(0, 2, 1)

        # 2. Shared Backbone for Feature Extraction
        x_feat = F.relu(self.bn1(self.conv1(x_transformed)))

        # This is the matrix needed for regularization
        feature_transform = self.feature_tnet(x_feat)

        x = torch.bmm(x_feat.transpose(2, 1),
                      feature_transform).transpose(2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(
            self.conv3(x))  # Final per-point features: (B, feature_dim, N)

        # 3. Prepare for Transformer by adding back transformed XYZ coordinates
        xyz_coords = x_transformed[:, :3, :]
        combined_for_transformer = torch.cat([xyz_coords, x],
                                             dim=1)  # (B, 3+feature_dim, N)
        transformer_input = combined_for_transformer.permute(
            2, 0, 1)  # (N, B, embedding_dim)

        # 4. Transformer and MLP Head
        transformer_output = self.transformer_encoder(transformer_input)
        aggregated_features = torch.max(transformer_output,
                                        0)[0]  # (B, embedding_dim)

        # 5. Final prediction layers
        x = self.drop1(F.relu(self.bn1_head(self.fc1(aggregated_features))))
        x = self.drop2(F.relu(self.bn2_head(self.fc2(x))))
        x = self.fc3(x)

        # MODIFIED: Return both prediction and the transform matrix for the loss
        return x, feature_transform


class JomonKaenDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_root,
                 pottery_path,
                 batch_size=8,
                 num_workers=4,
                 n_samples_global=512,
                 n_samples_local=1536,
                 num_local_seeds=64,
                 local_radius_ratio=7.5,
                 merge_points=True):
        super().__init__()
        self.save_hyperparameters()

    def get_jomon_kaen_data(self, data_root, pottery_path, n_samples_global,
                            n_samples_local, num_local_seeds,
                            local_radius_ratio, test_groups, merge_points):
        data = [{
            'processed_pottery_path': f"{pottery_path}/{p}",
            'POTTERY': p
        } for p in os.listdir(pottery_path)]
        np.random.shuffle(data)
        train_data = [d for d in data if d['POTTERY'] not in test_groups]
        test_data = [d for d in data if d['POTTERY'] in test_groups]
        train_dataset = PreprocessJomonKaenDataset(
            data=train_data,
            pottery_path=data_root,
            n_samples_global=n_samples_global,
            n_samples_local=n_samples_local,
            num_local_seeds=num_local_seeds,
            local_radius_ratio=local_radius_ratio,
            merge_points=merge_points)
        test_dataset = PreprocessJomonKaenDataset(
            data=test_data,
            pottery_path=data_root,
            n_samples_global=n_samples_global,
            n_samples_local=n_samples_local,
            num_local_seeds=num_local_seeds,
            local_radius_ratio=local_radius_ratio,
            merge_points=merge_points)
        return train_dataset, test_dataset

    def setup(self, stage=None):
        common_params = {
            "data_root": self.hparams.data_root,
            "pottery_path": self.hparams.pottery_path,
            "n_samples_global": self.hparams.n_samples_global,
            "n_samples_local": self.hparams.n_samples_local,
            "num_local_seeds": self.hparams.num_local_seeds,
            "local_radius_ratio": self.hparams.local_radius_ratio,
            "merge_points": self.hparams.merge_points
        }
        test_groups = [
            "IN0009(5).ply", "NM0049(30).ply", "UD0005(71).ply",
            "NM0015(27).ply", "NM0066(31).ply", "NM0099(37).ply",
            "NM0154(42).ply", "NM0168(45).ply", "SI0001(53).ply",
            "SJ0503(54).ply"
        ]
        if stage in ('fit', None):
            self.train_dataset, self.val_dataset = self.get_jomon_kaen_data(
                test_groups=test_groups, **common_params)
            train_files = [d['POTTERY'] for d in self.train_dataset.data]
            all_labels_df = self.train_dataset.labels
            train_labels_df = all_labels_df[all_labels_df['CODE'].isin(
                train_files)]
            feature_columns = self.train_dataset.headers[1:]
            labels = train_labels_df[feature_columns].values
            num_samples = len(labels)
            num_positives = np.sum(labels, axis=0)
            num_negatives = num_samples - num_positives
            pos_weight = num_negatives / (num_positives + 1e-6)
            self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        if stage == 'predict':
            _, self.predict_dataset = self.get_jomon_kaen_data(
                test_groups=test_groups, **common_params)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=3,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
                          batch_size=3,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)


class SaveValidationPredictions(Callback):

    def __init__(self, output_dir, save_every_n_epochs=10):
        super().__init__()
        self.output_dir = output_dir
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(self.output_dir, exist_ok=True)
        self.headers = [
            'HAS_FLAME_LIKE_DECORATION', 'HAS_CROWN_LIKE_DECORATION',
            'HAS_HANDLES', 'HAS_CORD_MARKED_PATTERN', 'HAS_NAIL_ENGRAVING',
            'HAS_SPIRAL_PATTERN', 'NUMBER_OF_PERTRUSIONS_0.0',
            'NUMBER_OF_PERTRUSIONS_1.0', 'NUMBER_OF_PERTRUSIONS_2.0',
            'NUMBER_OF_PERTRUSIONS_3.0', 'NUMBER_OF_PERTRUSIONS_4.0',
            'NUMBER_OF_PERTRUSIONS_6.0', 'NUMBER_OF_PERTRUSIONS_8.0'
        ]

    def on_validation_epoch_start(self, trainer, pl_module):
        self.preds, self.targets = [], []

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx=0):
        if isinstance(outputs,
                      dict) and 'preds' in outputs and 'targets' in outputs:
            self.preds.append(outputs['preds'].cpu())
            self.targets.append(outputs['targets'].cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch > 0 and epoch % self.save_every_n_epochs == 0 and self.preds:
            all_preds = torch.cat(self.preds).int().numpy()
            all_targets = torch.cat(self.targets).int().numpy()
            df_preds = pd.DataFrame(all_preds, columns=self.headers)
            df_targets = pd.DataFrame(all_targets, columns=self.headers)
            df_preds['source'], df_targets[
                'source'] = 'prediction', 'ground_truth'
            df_preds['sample_id'], df_targets['sample_id'] = range(
                len(df_preds)), range(len(df_targets))
            df_combined = pd.concat([df_targets, df_preds], ignore_index=True)
            df_combined = df_combined.sort_values(by=['sample_id', 'source'],
                                                  ascending=[True, False])
            final_cols = ['sample_id', 'source'] + self.headers
            df_combined = df_combined[final_cols]
            filename = os.path.join(self.output_dir,
                                    f'eval_predictions_epoch_{epoch}.csv')
            df_combined.to_csv(filename, index=False)
            pl_module.print(
                f"\nSaved validation predictions for epoch {epoch} to {filename}"
            )


class PointNetLightningModule(pl.LightningModule):

    def __init__(self,
                 learning_rate=1e-3,
                 lr_final=1e-5,
                 pos_weight=None,
                 reg_weight=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointNetTransformerRegressorUnifiedInput()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.hparams.pos_weight)
        # self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inputs):
        return self.model(inputs)

    def compute_regularization_loss(self, mat):
        k = mat.size(1)
        identity = torch.eye(k, device=self.device).unsqueeze(0).repeat(
            mat.size(0), 1, 1)
        mat_mult = torch.bmm(mat, mat.transpose(2, 1))
        loss = torch.nn.functional.mse_loss(mat_mult, identity)
        return loss

    def _shared_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, feature_transform = self.forward(inputs)
        classification_loss = self.criterion(outputs, targets)
        regularization_loss = self.compute_regularization_loss(
            feature_transform)
        total_loss = classification_loss + self.hparams.reg_weight * regularization_loss
        preds = torch.sigmoid(outputs) > 0.5
        accuracy = (preds == targets.bool()).float().mean()
        return total_loss, classification_loss, regularization_loss, accuracy, preds, targets

    def training_step(self, batch, batch_idx):
        total_loss, class_loss, reg_loss, accuracy, _, _ = self._shared_step(
            batch, batch_idx)
        self.log('train_loss',
                 total_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('train_class_loss',
                 class_loss,
                 on_step=True,
                 on_epoch=True,
                 logger=True)
        self.log('train_reg_loss',
                 reg_loss,
                 on_step=True,
                 on_epoch=True,
                 logger=True)
        self.log('train_acc',
                 accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, class_loss, reg_loss, accuracy, preds, targets = self._shared_step(
            batch, batch_idx)
        self.log('val_loss',
                 total_loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('val_class_loss', class_loss, on_epoch=True, logger=True)
        self.log('val_reg_loss', reg_loss, on_epoch=True, logger=True)
        self.log('val_acc',
                 accuracy,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return {'preds': preds, 'targets': targets}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr_final)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


if __name__ == '__main__':
    N_SAMPLES_GLOBAL = 4096
    N_SAMPLES_LOCAL = 4096
    N_TOTAL_SAMPLES = N_SAMPLES_GLOBAL + N_SAMPLES_LOCAL
    LOCAL_RADIUS_RATIO = 10.0
    NUM_LOCAL_SEEDS = 64
    MERGE_POINTS = True
    BATCH_SIZE = 4
    MAX_EPOCHS = 100
    NUM_GPUS = torch.cuda.device_count()
    NUM_WORKERS = int(os.cpu_count() / 3) if os.cpu_count() else 0
    LEARNING_RATE_INITIAL = 1e-3
    LEARNING_RATE_FINAL = 1e-5

    torch.set_float32_matmul_precision('high')
    print(
        f"Total points per sample: {N_TOTAL_SAMPLES} ({N_SAMPLES_GLOBAL} global + {N_SAMPLES_LOCAL} local)"
    )
    print(
        f"Found {NUM_GPUS} GPUs and {os.cpu_count()} CPUs. Using {NUM_WORKERS} workers."
    )

    datamodule = JomonKaenDataModule(data_root=r".\DS_Labels_Cleaned.csv",
                                     pottery_path=r".\voxel_pottery",
                                     batch_size=BATCH_SIZE,
                                     num_workers=NUM_WORKERS,
                                     n_samples_global=N_SAMPLES_GLOBAL,
                                     n_samples_local=N_SAMPLES_LOCAL,
                                     local_radius_ratio=LOCAL_RADIUS_RATIO,
                                     num_local_seeds=NUM_LOCAL_SEEDS,
                                     merge_points=MERGE_POINTS)
    datamodule.setup('fit')

    model = PointNetLightningModule(learning_rate=LEARNING_RATE_INITIAL,
                                    lr_final=LEARNING_RATE_FINAL,
                                    pos_weight=datamodule.pos_weight,
                                    reg_weight=0.001)

    print("\nModel Summary:")
    torchinfo.summary(model, input_size=[(BATCH_SIZE, N_TOTAL_SAMPLES, 6)])

    best_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='pointnet-transformer-fused-best-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
        save_last=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            patience=100,
                                            verbose=True,
                                            mode='min')
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    prediction_saver_callback = SaveValidationPredictions(
        output_dir='eval_predictions', save_every_n_epochs=10)

    trainer = pl.Trainer(accelerator='gpu' if NUM_GPUS > 0 else 'cpu',
                         devices=NUM_GPUS if NUM_GPUS > 0 else 1,
                         precision="16-mixed" if NUM_GPUS > 0 else "32-true",
                         max_epochs=MAX_EPOCHS,
                         callbacks=[
                             best_checkpoint_callback, early_stopping_callback,
                             lr_monitor_callback, prediction_saver_callback
                         ],
                         log_every_n_steps=10)

    VISUALIZE_SAMPLES = True
    try:
        if VISUALIZE_SAMPLES:
            print("\n--- Visualizing Data Samples")
            datamodule.setup('fit')
            num_samples_to_show = 5
            for i in range(num_samples_to_show):
                if not MERGE_POINTS:
                    fps_tensor, local_tensor, _ = datamodule.train_dataset[i]
                    pcd_fps = o3d.geometry.PointCloud()
                    pcd_fps.points = o3d.utility.Vector3dVector(
                        fps_tensor[:, :3].numpy())
                    pcd_fps.colors = o3d.utility.Vector3dVector(
                        fps_tensor[:, 3:].numpy())
                    pcd_local = o3d.geometry.PointCloud()
                    pcd_local.points = o3d.utility.Vector3dVector(
                        local_tensor[:, :3].numpy())
                    pcd_local.colors = o3d.utility.Vector3dVector(
                        local_tensor[:, 3:].numpy())
                    x_shift = pcd_fps.get_max_bound()[0] - pcd_local.get_min_bound(
                    )[0] + 0.3
                    pcd_local.translate((x_shift, 0, 0))
                    print(
                        f"Showing sample {i+1}/{num_samples_to_show}. Close the window to continue..."
                    )
                    o3d.visualization.draw_geometries(
                        [pcd_fps, pcd_local],
                        window_name=
                        f"Sample {i+1} | Global FPS (Left) vs Local FPS (Right)")
                else:
                    combined_tensor, _ = datamodule.train_dataset[i]

                    pcd_combined = o3d.geometry.PointCloud()
                    pcd_combined.points = o3d.utility.Vector3dVector(
                        combined_tensor[:, :3].numpy())
                    pcd_combined.colors = o3d.utility.Vector3dVector(
                        combined_tensor[:, 3:].numpy())

                    print(
                        f"Showing sample {i+1}/{num_samples_to_show}. Close the window to continue"
                    )

                    o3d.visualization.draw_geometries(
                        [pcd_combined],
                        window_name=f"Sample {i+1} | Fused Point Cloud")
    except Exception as e:
        print(f"Could not visualize samples: {e}")

    print("\nStarting Training")
    trainer.fit(model, datamodule=datamodule)
    print("Training Finished")
