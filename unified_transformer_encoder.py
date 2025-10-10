# Implmentation of PointNet and Transformer
# Inspired by https://arxiv.org/abs/2109.08141
# Using only PointNet for now, since the data pointcloud is mostly uniform, no need to account for different densities
# 2 sampling methods:
#      1. FPS for global context
#      2. Select N seed points from the first FPS, then using a ball query to get nearby points within LOCAL_RADIUS.
#         The points in the raduis are downsampled with FPS to get a dense local context
# The local sampling can be disabled by setting N_SAMPLES_LOCAL=0

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
import open3d as o3d
import numpy as np
import torchinfo

from dataset import PreprocessJomonKaenDataset
from model import PointNetTransformerRegressor

class JomonKaenDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_root,
                 pottery_path,
                 batch_size=8,
                 num_workers=4,
                 n_samples_global=512,
                 n_samples_local=1536,
                 num_local_seeds=64,
                 local_radius=7.5):
        super().__init__()
        self.save_hyperparameters()

    def get_jomon_kaen_data(self, data_root, pottery_path, n_samples_global,
                            n_samples_local, num_local_seeds, local_radius,
                            test_groups):
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
            local_radius=local_radius)
        test_dataset = PreprocessJomonKaenDataset(
            data=test_data,
            pottery_path=data_root,
            n_samples_global=n_samples_global,
            n_samples_local=n_samples_local,
            num_local_seeds=num_local_seeds,
            local_radius=local_radius)

        return train_dataset, test_dataset

    def setup(self, stage=None):
        common_params = {
            "data_root": self.hparams.data_root,
            "pottery_path": self.hparams.pottery_path,
            "n_samples_global": self.hparams.n_samples_global,
            "n_samples_local": self.hparams.n_samples_local,
            "num_local_seeds": self.hparams.num_local_seeds,
            "local_radius": self.hparams.local_radius,
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
        if stage == 'predict':
            _, self.predict_dataset = self.get_jomon_kaen_data(
                test_groups=test_groups, **common_params)

    # Dataloader methods (train_dataloader, etc.) remain unchanged
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
    """
    Saves the model's predictions and the ground truth labels on the validation set
    to a CSV file every N epochs. For each sample, two rows are created: one for the
    ground truth and one for the prediction, identified by a 'source' column.
    """

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
        ]  # Make sure it is same as the dataset labels

    def on_validation_epoch_start(self, trainer, pl_module):
        """Reset the storage at the beginning of each validation epoch."""
        self.preds = []
        self.targets = []

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx=0):
        """Gather predictions and targets from each validation batch."""
        # Ensure 'outputs' is a dictionary containing 'preds' and 'targets'
        if isinstance(outputs,
                      dict) and 'preds' in outputs and 'targets' in outputs:
            self.preds.append(outputs['preds'].cpu())
            self.targets.append(outputs['targets'].cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Save predictions at the end of the validation epoch if the condition is met."""
        epoch = trainer.current_epoch + 1
        # Proceed only if the epoch is a save interval and we have collected predictions
        if epoch > 0 and epoch % self.save_every_n_epochs == 0 and self.preds:
            # Concatenate all batch results and convert to integer (0 or 1) numpy arrays
            all_preds = torch.cat(self.preds).int().numpy()
            all_targets = torch.cat(self.targets).int().numpy()

            # Create DataFrames for predictions and targets using the same feature headers
            df_preds = pd.DataFrame(all_preds, columns=self.headers)
            df_targets = pd.DataFrame(all_targets, columns=self.headers)

            # Add a 'source' column to identify predictions vs. ground truth
            df_preds['source'] = 'prediction'
            df_targets['source'] = 'ground_truth'

            # Add a 'sample_id' to link the pairs of rows together
            df_preds['sample_id'] = range(len(df_preds))
            df_targets['sample_id'] = range(len(df_targets))

            # Combine the two dataframes vertically
            df_combined = pd.concat([df_targets, df_preds], ignore_index=True)

            # Sort by sample_id to group each sample's truth and prediction together
            df_combined = df_combined.sort_values(by=['sample_id', 'source'],
                                                  ascending=[True, False])

            # Reorder columns to make the output cleaner
            final_cols = ['sample_id', 'source'] + self.headers
            df_combined = df_combined[final_cols]

            # Save the combined DataFrame to a CSV file
            filename = os.path.join(self.output_dir,
                                    f'eval_predictions_epoch_{epoch}.csv')
            df_combined.to_csv(filename, index=False)
            pl_module.print(
                f"\nSaved validation predictions for epoch {epoch} to {filename}"
            )


class PointNetLightningModule(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, lr_final=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointNetTransformerRegressor()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inputs):
        return self.model(inputs)

    def _shared_step(self, batch, batch_idx):
        # Unpack single input tensor
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        preds = torch.sigmoid(outputs) > 0.5
        accuracy = (preds == targets.bool()).float().mean()
        return loss, accuracy, preds, targets

    def training_step(self, batch, batch_idx):
        loss, accuracy, _, _ = self._shared_step(batch, batch_idx)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('train_acc',
                 accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, preds, targets = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
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
    N_SAMPLES_GLOBAL = 2048  # Number of seed points from FPS
    N_SAMPLES_LOCAL = 6144  # Number of points sampled around the seeds
    N_TOTAL_SAMPLES = N_SAMPLES_GLOBAL + N_SAMPLES_LOCAL
    LOCAL_RADIUS = 8.0
    NUM_LOCAL_SEEDS = 48

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

    datamodule = JomonKaenDataModule(
        data_root=r".\DS_Labels_Cleaned.csv",
        pottery_path=
        r".\voxel_pottery",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        n_samples_global=N_SAMPLES_GLOBAL,
        n_samples_local=N_SAMPLES_LOCAL,
        local_radius=LOCAL_RADIUS,
        num_local_seeds=NUM_LOCAL_SEEDS)

    model = PointNetLightningModule(learning_rate=LEARNING_RATE_INITIAL,
                                    lr_final=LEARNING_RATE_FINAL)

    print("\nModel Summary:")
    # Update input size for torchinfo summary
    torchinfo.summary(model, input_size=[(BATCH_SIZE, N_TOTAL_SAMPLES, 6)])

    best_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='pointnet-fused-best-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
        save_last=True)

    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            patience=30,
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
            num_samples_to_show = 3
            for i in range(num_samples_to_show):
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
