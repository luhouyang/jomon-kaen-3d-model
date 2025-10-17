import torch
import torch.nn as nn
import torch.nn.functional as F


# Using a smaller T-Net than original, can try other configuraations
class TNet(nn.Module):
    """
    A Transformation Network (T-Net) that learns a KxK transformation matrix.
    Slightly smaller version to reduce model capacity.
    """

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

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


# yapf: disable
class PointNetTransformerRegressorUnifiedInput(nn.Module):
    def __init__(self,
                 num_outputs=13, # 13 with pertursions, 6 without
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
        self.embedding_dim = feature_dim + 3  # Add XYZ back, from 3DETR
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,
            dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # MLP Head
        self.fc1 = nn.Linear(self.embedding_dim, 128)
        self.bn1_head = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.bn2_head = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_outputs)

    def forward(self, inputs):
        # inputs shape: (B, N, 6)

        # 1. Input Transformation
        inputs = inputs.permute(0, 2, 1)  # (B, 6, N)
        transform = self.input_tnet(inputs)
        x = torch.bmm(inputs.transpose(2, 1), transform).transpose(2, 1)

        # 2. Shared Backbone for Feature Extraction
        x_feat = F.relu(self.bn1(self.conv1(x)))
        feature_transform = self.feature_tnet(x_feat)
        x = torch.bmm(x_feat.transpose(2, 1),
                      feature_transform).transpose(2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # Final per-point features: (B, feature_dim, N)

        # 3. Prepare for Transformer by adding back XYZ coordinates
        xyz_coords = x[:, :3, :]  # Get transformed XYZ
        combined_for_transformer = torch.cat([xyz_coords, x], dim=1)  # (B, 3+feature_dim, N)
        transformer_input = combined_for_transformer.permute(2, 0, 1)  # (N, B, 3+feature_dim)

        # 4. Transformer and MLP Head
        transformer_output = self.transformer_encoder(transformer_input)  # (N, B, embedding_dim)
        aggregated_features = torch.max(transformer_output, 0)[0]  # (B, embedding_dim)

        x = self.drop1(F.relu(self.bn1_head(self.fc1(aggregated_features))))
        x = self.drop2(F.relu(self.bn2_head(self.fc2(x))))
        x = self.fc3(x)
        return x
# yapf: enable


# yapf: disable
class PointNetTransformerRegressorSeperatedInput(nn.Module):
    def __init__(self,
                 num_outputs=13,
                 feature_dim=125,
                 nhead=4,
                 num_encoder_layers=2,
                 dim_feedforward=128):
        super().__init__()

        # --- Separate Initial Layers ---
        self.input_tnet1 = TNet(k=6)
        self.input_tnet2 = TNet(k=6)

        # --- Shared Layers ---
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.feature_tnet = TNet(k=64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, feature_dim, 1)
        self.bn3 = nn.BatchNorm1d(feature_dim)

        # Smaller Transformer Encoder
        self.embedding_dim = feature_dim + 3  # Add XYZ back, from 3DETR
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,
            dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # MLP Head
        self.fc1 = nn.Linear(self.embedding_dim, 128)
        self.bn1_head = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.bn2_head = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_outputs)

    def forward(self, inputs_global, inputs_local):
        # 1. Separate Input Transformations
        input1 = inputs_global.permute(0, 2, 1)
        transform1 = self.input_tnet1(input1)
        x1 = torch.bmm(input1.transpose(2, 1), transform1).transpose(2, 1)

        input2 = inputs_local.permute(0, 2, 1)
        transform2 = self.input_tnet2(input2)
        x2 = torch.bmm(input2.transpose(2, 1), transform2).transpose(2, 1)

        # 2. Extract initial features from each branch using the shared conv1
        # We apply the batch norm separately to the concatenated tensor for stable statistics
        x_cat = torch.cat([x1, x2], dim=2)
        x_feat = F.relu(self.bn1(self.conv1(x_cat)))

        # 3. Pass combined features through shared backbone
        feature_transform = self.feature_tnet(x_feat)
        x = torch.bmm(x_feat.transpose(2, 1),feature_transform).transpose(2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # Final per-point features for the combined set 

        # 3. Prepare for Transformer by adding back XYZ coordinates
        xyz_coords = x[:, :3, :]  # Get transformed XYZ
        combined_for_transformer = torch.cat([xyz_coords, x], dim=1)  # (B, 3+feature_dim, N)
        transformer_input = combined_for_transformer.permute(2, 0, 1)  # (N, B, 3+feature_dim)

        # 4. Transformer and MLP Head
        transformer_output = self.transformer_encoder(transformer_input)  # (N, B, embedding_dim)
        aggregated_features = torch.max(transformer_output, 0)[0]  # (B, embedding_dim)

        x = self.drop1(F.relu(self.bn1_head(self.fc1(aggregated_features))))
        x = self.drop2(F.relu(self.bn2_head(self.fc2(x))))
        x = self.fc3(x)
        return x
# yapf: enable


# yapf: disable
class PointNetRegressor(nn.Module):
    def __init__(self,
                 num_outputs=13,
                 ):
        super().__init__()

        # Single Input Stream
        self.input_tnet = TNet(k=6)
        self.conv1 = nn.Conv1d(6, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.feature_tnet = TNet(k=32)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 3, 1)
        self.bn3 = nn.BatchNorm1d(3)

        self.flatten = nn.Flatten()

        # MLP Head
        self.fc1 = nn.Linear(8192*3, 128)
        self.bn1_head = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.bn2_head = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_outputs)

    def forward(self, inputs):
        # inputs shape: (B, N, 6)

        # 1. Input Transformation
        inputs = inputs.permute(0, 2, 1)  # (B, 6, N)
        transform = self.input_tnet(inputs)
        x = torch.bmm(inputs.transpose(2, 1), transform).transpose(2, 1)

        # 2. Shared Backbone for Feature Extraction
        x_feat = F.relu(self.bn1(self.conv1(x)))
        feature_transform = self.feature_tnet(x_feat)
        x = torch.bmm(x_feat.transpose(2, 1),
                      feature_transform).transpose(2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # Final per-point features: (B, feature_dim, N)

        x = self.drop1(F.relu(self.bn1_head(self.fc1(self.flatten(x)))))
        x = self.drop2(F.relu(self.bn2_head(self.fc2(x))))
        x = self.fc3(x)
        return x
# yapf: enable
