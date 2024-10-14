import torch
import torch.nn as nn

class Add_Integration(nn.Module):
    def forward(self, x, sensor_data):
        # Expand sensor data to match spatial dimensions of x
        sensor_data_expanded = sensor_data.unsqueeze(-1).unsqueeze(-1)  # [batch_size, channels, T', 1, 1]
        spatial_size = x.shape[-2:]  # (height, width)
        sensor_data_expanded = sensor_data_expanded.expand(-1, -1, -1, *spatial_size)
        
        # Add the expanded sensor data to x
        return x + sensor_data_expanded

class Cat_Integration(nn.Module):
    def __init__(self):
        super(Cat_Integration, self).__init__()
        # After concatenation, we have 128 channels, so we reduce it back to 64
        self.conv = nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x, sensor_data):
        # Expand sensor data to match spatial dimensions of x
        sensor_data_expanded = sensor_data.unsqueeze(-1).unsqueeze(-1)  # [batch_size, channels_s, T', 1, 1]
        spatial_size = x.shape[-2:]  # (height, width)
        sensor_data_expanded = sensor_data_expanded.expand(-1, -1, -1, *spatial_size)
        
        # Concatenate along the channel dimension
        x = torch.cat([x, sensor_data_expanded], dim=1)  # Concatenated to [batch_size, 128, T', width', height']
        
        # Apply convolution to reduce back to 64 channels
        x = self.conv(x)
        
        return x

class AdaptiveNormalizationIntegration(nn.Module):
    def __init__(self):
        super(AdaptiveNormalizationIntegration, self).__init__()
        self.bn = nn.BatchNorm3d(64, affine=False)
        self.gamma_generator = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.beta_generator = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, sensor_data):
        # Compute gamma and beta from sensor_data
        gamma = self.gamma_generator(sensor_data)  # [batch_size, 64, T']
        beta = self.beta_generator(sensor_data)    # [batch_size, 64, T']
        # Expand to match x dimensions
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 64, T', 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [batch_size, 64, T', 1, 1]
        gamma = gamma.expand(-1, -1, -1, x.size(-2), x.size(-1))
        beta = beta.expand(-1, -1, -1, x.size(-2), x.size(-1))
        # Apply batch normalization without affine parameters
        x = self.bn(x)
        # Apply adaptive scaling and shifting
        x = gamma * x + beta
        return x

class CrossAttentionIntegration(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super(CrossAttentionIntegration, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Spatial pooling to reduce spatial dimensions
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # Multi-head attention module with batch_first=True
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, sensor_data):
        batch_size, C, T, H, W = x.size()

        # Spatial pooling on visual features
        x_pooled = self.spatial_pool(x).view(batch_size, C, T).permute(0, 2, 1)  # [batch_size, T, C]

        # Sensor data: [batch_size, sensor_channels, T]
        sensor_proj = sensor_data.permute(0, 2, 1)  # [batch_size, T, sensor_channels]

        # Ensure that the feature dimensions match embed_dim
        if C != self.embed_dim or sensor_proj.size(2) != self.embed_dim:
            raise ValueError(f"Feature dimensions of visual ({C}) and sensor data ({sensor_proj.size(2)}) must match embed_dim ({self.embed_dim}). Adjust embed_dim or include projection layers.")

        # Perform multi-head attention with sensor data as queries
        # and visual features as keys and values
        attn_output, attn_weights = self.multihead_attn(query=sensor_proj, key=x_pooled, value=x_pooled)

        # Add the attended output to the original visual features
        # We need to align the dimensions appropriately
        attn_output = attn_output.permute(0, 2, 1)  # [batch_size, C, T]

        # Expand dimensions to match x's spatial dimensions
        attended_values = attn_output.unsqueeze(-1).unsqueeze(-1)  # [batch_size, C, T, 1, 1]
        attended_values = attended_values.expand(-1, -1, -1, H, W)  # [batch_size, C, T, H, W]

        # Integrate attended values into x
        x = x + attended_values  # [batch_size, C, T, H, W]

        return x

class AriaNet_Multimodal(nn.Module):
    def __init__(self, frames=128, integration_strategy='add', integration_layers=[2], sensor_type='imu', skip_connections=False):
        super(AriaNet_Multimodal, self).__init__()

        self.frames = frames
        self.integration_layers = integration_layers
        self.sensor_type = sensor_type
        sensor_in_channels = 6 if sensor_type == 'imu' else 4
        self.skip_connections = skip_connections

        # Visual stream convolutional blocks
        self.visual_modules = nn.ModuleList([
            # ConvBlock1 and MaxPool (spatial pooling only)
            nn.Sequential(
                nn.Conv3d(3, 16, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))  # Spatial pooling
            ),
            # ConvBlock2
            nn.Sequential(
                nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)
            ),
            # ConvBlock3 and MaxPool (spatial and temporal pooling)
            nn.Sequential(
                nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d((2, 2, 2), stride=2)  # Temporal and spatial pooling
            ),
            # ConvBlock4
            nn.Sequential(
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            ),
            # ConvBlock5 and MaxPool (spatial and temporal pooling)
            nn.Sequential(
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d((2, 2, 2), stride=2)  # Temporal and spatial pooling
            ),
            # ConvBlock6
            nn.Sequential(
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            ),
            # ConvBlock7 and MaxPool (spatial pooling only)
            nn.Sequential(
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))  # Spatial pooling
            ),
            # ConvBlock8
            nn.Sequential(
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            ),
            # ConvBlock9
            nn.Sequential(
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            ),
            # Upsample
            nn.Sequential(
                nn.ConvTranspose3d(64, 64, kernel_size=(4, 1, 1),
                                   stride=(2, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True)
            ),
            # Upsample2
            nn.Sequential(
                nn.ConvTranspose3d(64, 64, kernel_size=(4, 1, 1),
                                   stride=(2, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True)
            ),
            # PoolSpa and ConvBlock10
            nn.Sequential(
                nn.AdaptiveAvgPool3d((self.frames, 1, 1)),
                nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)
            )
        ])

        # Sensor stream convolutional blocks
        self.sensor_modules = nn.ModuleList([
            # ConvBlock1 (no temporal pooling)
            nn.Sequential(
                nn.Conv1d(sensor_in_channels, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True)
                # No temporal pooling here
            ),
            # ConvBlock2
            nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True)
            ),
            # ConvBlock3 and MaxPool (temporal pooling)
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)  # Temporal pooling
            ),
            # ConvBlock4
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            ),
            # ConvBlock5 and MaxPool (temporal pooling)
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)  # Temporal pooling
            ),
            # ConvBlock6
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            ),
            # ConvBlock7 (no temporal pooling)
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
                # No temporal pooling here
            ),
            # ConvBlock8
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            ),
            # ConvBlock9
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            ),
            # Upsample
            nn.Sequential(
                nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.ELU(inplace=True)
            ),
            # Upsample2
            nn.Sequential(
                nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.ELU(inplace=True)
            ),
            # Pooling and ConvBlock10
            nn.Sequential(
                nn.AdaptiveAvgPool1d(self.frames),
                nn.Conv1d(64, 1, kernel_size=1, stride=1, padding=0)
            )
        ])

        # Initialize the integration strategy (add or concat)
        self.integrate = nn.ModuleDict()
        if integration_strategy == 'add':
            for layer_idx in integration_layers:
                self.integrate[str(layer_idx)] = Add_Integration()
        elif integration_strategy == 'cat':
            for layer_idx in integration_layers:
                self.integrate[str(layer_idx)] = Cat_Integration()
        elif integration_strategy == 'adaptive_norm':
            for layer_idx in integration_layers:
                self.integrate[str(layer_idx)] = AdaptiveNormalizationIntegration()
        elif integration_strategy == 'cross_attention':
            for layer_idx in integration_layers:
                self.integrate[str(layer_idx)] = CrossAttentionIntegration()
        else:
            raise ValueError(f"Unknown integration strategy: {integration_strategy}")

    def forward(self, x, imu=None, quaternions=None):
        batch_size, C, T, H, W = x.size()
        
        # Select sensor data
        if self.sensor_type == 'imu':
            sensor_data = imu  # Expected shape [batch_size, length, 6]
        else:
            sensor_data = quaternions  # Expected shape [batch_size, length, 4]

        # Reshape sensor data to [batch_size, channels, length]
        sensor_data = sensor_data.permute(0, 2, 1)

        # Process both streams up to the integration layers
        for i in range(len(self.visual_modules)):
            # Store previous outputs for skip connections
            x_prev = x
            sensor_prev = sensor_data

            # Process visual and sensor modules
            x = self.visual_modules[i](x)
            sensor_data = self.sensor_modules[i](sensor_data)

            # Add skip connections if dimensions match
            if x.size() == x_prev.size() and self.skip_connections:
                x = x + x_prev
            if sensor_data.size() == sensor_prev.size() and self.skip_connections:
                sensor_data = sensor_data + sensor_prev

            # Perform integration if it's an integration layer
            if str(i) in self.integrate:
                x = self.integrate[str(i)](x, sensor_data)

        rPPG = x.view(batch_size, self.frames)  # [batch_size, frames]
        return rPPG