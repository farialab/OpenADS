import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product

def fast_conv3d_transpose_numpy(input_tensor, kernel_weights, bias, 
                              strides=(2, 2, 2), padding='SAME'):
    """
    Optimized NumPy implementation of 3D transposed convolution.
    
    Args:
        input_tensor: 5D numpy array (batch, depth, height, width, in_channels)
        kernel_weights: 5D numpy array (kernel_d, kernel_h, kernel_w, out_channels, in_channels)
        bias: 1D numpy array (out_channels,)
        strides: Tuple of 3 integers for depth, height, width strides
        padding: 'SAME' or 'VALID'
    """
    batch_size, in_depth, in_height, in_width, in_channels = input_tensor.shape
    kernel_d, kernel_h, kernel_w, out_channels, _ = kernel_weights.shape
    stride_d, stride_h, stride_w = strides

    out_depth = in_depth * stride_d
    out_height = in_height * stride_h
    out_width = in_width * stride_w

    output = np.zeros((batch_size, out_depth, out_height, out_width, out_channels))

    pad_d = max((kernel_d - stride_d), 0) // 2
    pad_h = max((kernel_h - stride_h), 0) // 2
    pad_w = max((kernel_w - stride_w), 0) // 2

    d_indices = np.arange(in_depth)
    h_indices = np.arange(in_height)
    w_indices = np.arange(in_width)

    D, H, W = np.meshgrid(d_indices, h_indices, w_indices, indexing='ij')
    out_d_start = D * stride_d - pad_d
    out_h_start = H * stride_h - pad_h
    out_w_start = W * stride_w - pad_w

    for b in range(batch_size):
        input_slice = input_tensor[b]
        for kd, kh, kw in product(range(kernel_d), range(kernel_h), range(kernel_w)):
            out_d = out_d_start + kd
            out_h = out_h_start + kh
            out_w = out_w_start + kw

            valid_mask = (out_d >= 0) & (out_d < out_depth) & \
                         (out_h >= 0) & (out_h < out_height) & \
                         (out_w >= 0) & (out_w < out_width)

            if not np.any(valid_mask):
                continue

            kernel_slice = kernel_weights[kd, kh, kw]  # shape: (out_channels, in_channels)

            valid_d = out_d[valid_mask]
            valid_h = out_h[valid_mask]
            valid_w = out_w[valid_mask]

            input_vals = input_slice[D[valid_mask], H[valid_mask], W[valid_mask]]  # (num_valid, in_channels)
            output_vals = np.dot(input_vals, kernel_slice.T)  # (num_valid, out_channels)

            output[b, valid_d, valid_h, valid_w] += output_vals

    output += bias
    return output

class CustomConv3DTranspose(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 strides=(2, 2, 2),
                 padding='SAME',
                 use_bias=True):
        super(CustomConv3DTranspose, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

        self.weight = nn.Parameter(
            torch.randn(kernel_size[0], kernel_size[1], kernel_size[2], 
                       out_channels, in_channels) / np.sqrt(in_channels * np.prod(kernel_size))
        )

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv3d_transpose_fn = fast_conv3d_transpose_numpy

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        x_np = np.transpose(x_np, (0, 2, 3, 4, 1))  # (B, D, H, W, C)
        weight_np = self.weight.detach().cpu().numpy()
        bias = self.bias if self.use_bias else torch.zeros(self.out_channels, device=x.device)
        bias_np = bias.detach().cpu().numpy()

        output_np = self.conv3d_transpose_fn(x_np, weight_np, bias_np, self.strides, self.padding)
        output_torch = torch.from_numpy(output_np).permute(0, 4, 1, 2, 3).to(x.device)
        return output_torch

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, '
                f'strides={self.strides}, '
                f'padding={self.padding}, '
                f'use_bias={self.use_bias}')

class DAGMNet(nn.Module):
    def __init__(self):
        super(DAGMNet, self).__init__()

        # Activation functions
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()

        # Encoder Block 1
        self.conv_98 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn_48 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)
        self.conv_99 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn_49 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)

        # Encoder Block 2
        self.conv_100 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn_50 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)
        self.conv_101 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn_51 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)

        # Encoder Block 3
        self.conv_102 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn_52 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)
        self.conv_103 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn_53 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)

        # Encoder Block 4
        self.conv_104 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn_54 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)
        self.conv_105 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn_55 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)

        # Encoder12 layers
        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv_106 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_56 = nn.BatchNorm3d(64, eps=1e-3, momentum=0.99)
        self.conv_107 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_57 = nn.BatchNorm3d(64, eps=1e-3, momentum=0.99)

        # Encoder23 layers
        self.conv_108 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn_58 = nn.BatchNorm3d(128, eps=1e-3, momentum=0.99)
        self.conv_109 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_59 = nn.BatchNorm3d(128, eps=1e-3, momentum=0.99)

        # Encoder34 layers
        self.conv_110 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn_60 = nn.BatchNorm3d(256, eps=1e-3, momentum=0.99)
        self.conv_111 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn_61 = nn.BatchNorm3d(256, eps=1e-3, momentum=0.99)

        # DAG related layers
        # sAG blocks for DAG1
        self.conv_136 = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv_138 = nn.Conv3d(3, 1, kernel_size=5, stride=1, padding=2)
        self.conv_137 = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv_139 = nn.Conv3d(3, 1, kernel_size=5, stride=1, padding=2)
        self.conv_140 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)
        
        # sAG Blocks for DAG2
        self.conv_128 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)
        self.conv_129 = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv_130 = nn.Conv3d(3, 1, kernel_size=5, stride=1, padding=2)
        self.conv_131 = nn.Conv3d(3, 1, kernel_size=5, stride=1, padding=2)
        self.conv_132 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)
        
        # sAG Blocks for DAG3
        self.conv_120 = nn.Conv3d(128, 1, kernel_size=1, stride=1, padding=0)
        self.conv_121 = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv_122 = nn.Conv3d(3, 1, kernel_size=5, stride=1, padding=2)
        self.conv_123 = nn.Conv3d(3, 1, kernel_size=5, stride=1, padding=2)
        self.conv_124 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)
        
        # sAG Blocks for DAG4
        self.conv_112 = nn.Conv3d(256, 1, kernel_size=1, stride=1, padding=0)
        self.conv_113 = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv_114 = nn.Conv3d(3, 1, kernel_size=5, stride=1, padding=2)
        self.conv_115 = nn.Conv3d(3, 1, kernel_size=5, stride=1, padding=2)
        self.conv_116 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)

        # cAG Blocks for encoder1
        self.dense_66 = nn.Linear(32, 2, bias=True)
        self.dense_67 = nn.Linear(32, 2, bias=True)
        self.dense_68 = nn.Linear(32, 2, bias=True)
        self.dense_69 = nn.Linear(32, 2, bias=True)
        self.dense_70 = nn.Linear(2, 32, bias=True)
        self.dense_71 = nn.Linear(32, 32, bias=True)
        
        # cAG Blocks for encoder2
        self.dense_60 = nn.Linear(64, 4, bias=True)
        self.dense_61 = nn.Linear(64, 4, bias=True)
        self.dense_62 = nn.Linear(32, 4, bias=True)
        self.dense_63 = nn.Linear(32, 4, bias=True)
        self.dense_64 = nn.Linear(4, 64, bias=True)
        self.dense_65 = nn.Linear(64, 64, bias=True)

        # cAG Blocks for encoder3
        self.dense_54 = nn.Linear(128, 8, bias=True)
        self.dense_55 = nn.Linear(128, 8, bias=True)
        self.dense_56 = nn.Linear(32, 8, bias=True)
        self.dense_57 = nn.Linear(32, 8, bias=True)
        self.dense_58 = nn.Linear(8, 128, bias=True)
        self.dense_59 = nn.Linear(128, 128, bias=True)

        # cAG Blocks for encoder4
        self.dense_48 = nn.Linear(256, 16, bias=True)
        self.dense_49 = nn.Linear(256, 16, bias=True)
        self.dense_50 = nn.Linear(32, 16, bias=True)
        self.dense_51 = nn.Linear(32, 16, bias=True)
        self.dense_52 = nn.Linear(16, 256, bias=True)
        self.dense_53 = nn.Linear(256, 256, bias=True)

        # Decoder layers
        # Output 4
        self.conv_117 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn_62 = nn.BatchNorm3d(256, eps=1e-3, momentum=0.99)
        self.conv_118 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn_63 = nn.BatchNorm3d(256, eps=1e-3, momentum=0.99)
        self.conv_119 = nn.Conv3d(256, 1, kernel_size=3, padding=1)

        # Output 3
        self.conv_transpose_12 = CustomConv3DTranspose(in_channels=256, out_channels=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.conv_125 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.bn_64 = nn.BatchNorm3d(128, eps=1e-3, momentum=0.99)
        self.conv_126 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn_65 = nn.BatchNorm3d(128, eps=1e-3, momentum=0.99)
        self.conv_127 = nn.Conv3d(128, 1, kernel_size=3, padding=1)

        # Output 2
        self.conv_transpose_15 = CustomConv3DTranspose(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.conv_133 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.bn_66 = nn.BatchNorm3d(64, eps=1e-3, momentum=0.99)
        self.conv_134 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn_67 = nn.BatchNorm3d(64, eps=1e-3, momentum=0.99)
        self.conv_135 = nn.Conv3d(64, 1, kernel_size=3, padding=1)

        # Output 1
        self.conv_transpose_17 = CustomConv3DTranspose(in_channels=64, out_channels=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.conv_141 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.bn_68 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)
        self.conv_142 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn_69 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)
        self.conv_143 = nn.Conv3d(32, 1, kernel_size=3, padding=1)

        # Output fused related
        self.conv_transpose_13 = CustomConv3DTranspose(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.conv_transpose_14 = CustomConv3DTranspose(in_channels=64, out_channels=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.conv_transpose_16 = CustomConv3DTranspose(in_channels=64, out_channels=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.conv_144 = nn.Conv3d(128, 32, kernel_size=3, padding=1)
        self.bn_70 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)
        self.conv_145 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn_71 = nn.BatchNorm3d(32, eps=1e-3, momentum=0.99)
        self.conv_146 = nn.Conv3d(32, 1, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoder Block 1
        x1 = self.conv_98(x)
        x1 = self.bn_48(x1)
        x1 = F.selu(x1)
        x1 = self.conv_99(x1)
        x1 = F.selu(self.bn_49(x1))
        encoder1_output = x1

        # Encoder Block 2
        x2 = x[:, :, ::2, ::2, ::2]
        x2 = self.conv_100(x2)
        x2 = F.selu(self.bn_50(x2))
        x2 = self.conv_101(x2)
        x2 = F.selu(self.bn_51(x2))
        encoder2_output = x2

        # Encoder Block 3
        x3 = x[:, :, ::4, ::4, ::4]
        x3 = self.conv_102(x3)
        x3 = F.selu(self.bn_52(x3))
        x3 = self.conv_103(x3)
        x3 = F.selu(self.bn_53(x3))
        encoder3_output = x3

        # Encoder Block 4
        x4 = x[:, :, ::8, ::8, ::8]
        x4 = self.conv_104(x4)
        x4 = F.selu(self.bn_54(x4))
        x4 = self.conv_105(x4)
        x4 = F.selu(self.bn_55(x4))
        encoder4_output = x4

        # Encoder12
        pooled_49 = self.max_pool(encoder1_output)
        concatenated = torch.cat([pooled_49, encoder2_output], dim=1)
        x = self.conv_106(concatenated)
        x = F.selu(self.bn_56(x))
        x = self.conv_107(x)
        x = F.selu(self.bn_57(x))
        activation_57 = x

        # Encoder23
        x = self.max_pool(activation_57)
        x = self.conv_108(x)
        x = F.selu(self.bn_58(x))
        x = self.conv_109(x)
        x = F.selu(self.bn_59(x))
        activation_59 = x

        # Encoder34
        x = self.max_pool(activation_59)
        x = self.conv_110(x)
        x = F.selu(self.bn_60(x))
        x = self.conv_111(x)
        x = F.selu(self.bn_61(x))
        activation_61 = x

        # ========================
        # DAG1 (Level 1 Attention)
        # ========================
        # Spatial Attention Gate for encoder1_output
        branch1_max = torch.max(encoder1_output, dim=1, keepdim=True)[0]
        branch1_avg = torch.mean(encoder1_output, dim=1, keepdim=True)
        branch1_conv = self.conv_136(encoder1_output)
        branch1_conv = self.selu(branch1_conv)
        branch1_concat = torch.cat([branch1_avg, branch1_max, branch1_conv], dim=1)
        branch1_output = self.selu(self.conv_138(branch1_concat))

        branch2_max = torch.max(encoder1_output, dim=1, keepdim=True)[0]
        branch2_avg = torch.mean(encoder1_output, dim=1, keepdim=True)
        branch2_conv = self.selu(self.conv_137(encoder1_output))
        branch2_concat = torch.cat([branch2_avg, branch2_max, branch2_conv], dim=1)
        branch2_output = self.selu(self.conv_139(branch2_concat))

        combined_output = branch1_output + branch2_output
        output_sAG_en1 = self.sigmoid(self.conv_140(combined_output))

        # Channel Attention Gate for encoder1_output
        input1_max_pool = torch.amax(encoder1_output, dim=(2, 3, 4))
        input1_avg_pool = torch.mean(encoder1_output, dim=(2, 3, 4))
        input2_max_pool = torch.amax(encoder1_output, dim=(2, 3, 4))
        input2_avg_pool = torch.mean(encoder1_output, dim=(2, 3, 4))

        branch1_max_out = self.selu(self.dense_69(input1_max_pool))
        branch1_avg_out = self.selu(self.dense_68(input1_avg_pool))
        branch2_max_out = self.selu(self.dense_67(input2_max_pool))
        branch2_avg_out = self.selu(self.dense_66(input2_avg_pool))

        combined_features = branch1_max_out + branch1_avg_out + branch2_max_out + branch2_avg_out
        fusion1_out = self.selu(self.dense_70(combined_features))
        output_cAG_en1 = self.sigmoid(self.dense_71(fusion1_out))

        dag1_output = output_sAG_en1 * encoder1_output * output_cAG_en1.view(-1, 32, 1, 1, 1)

        # ========================
        # DAG2 (Level 2 Attention)
        # ========================
        # Spatial Attention Gate: activation_57 vs encoder2_output
        branch1_max_57 = torch.max(activation_57, dim=1, keepdim=True)[0]
        branch1_avg_57 = torch.mean(activation_57, dim=1, keepdim=True)
        branch1_conv_57 = self.selu(self.conv_128(activation_57))
        branch1_concat_57 = torch.cat([branch1_avg_57, branch1_max_57, branch1_conv_57], dim=1)
        branch1_output_57 = self.selu(self.conv_130(branch1_concat_57))

        branch2_max_57 = torch.max(encoder2_output, dim=1, keepdim=True)[0]
        branch2_avg_57 = torch.mean(encoder2_output, dim=1, keepdim=True)
        branch2_conv_57 = self.selu(self.conv_129(encoder2_output))
        branch2_concat_57 = torch.cat([branch2_avg_57, branch2_max_57, branch2_conv_57], dim=1)
        branch2_output_57 = self.selu(self.conv_131(branch2_concat_57))

        combined_output_57 = branch1_output_57 + branch2_output_57
        output_sAG_57 = self.sigmoid(self.conv_132(combined_output_57))

        avg_57 = activation_57.mean(dim=(2, 3, 4))
        max_57 = activation_57.amax(dim=(2, 3, 4))
        avg_2 = encoder2_output.mean(dim=(2, 3, 4))
        max_2 = encoder2_output.amax(dim=(2, 3, 4))

        cAG_2 = self.selu(self.dense_60(avg_57)) + self.selu(self.dense_61(max_57)) + \
                self.selu(self.dense_62(avg_2)) + self.selu(self.dense_63(max_2))
        cAG_2 = self.selu(self.dense_64(cAG_2))
        cAG_2 = self.sigmoid(self.dense_65(cAG_2))  # shape: (N, 64)

        dag2_output = output_sAG_57 * activation_57 * cAG_2.view(-1, 64, 1, 1, 1)

        # ========================
        # DAG3 (Level 3 Attention)
        # ========================
        branch1_max_59 = torch.max(activation_59, dim=1, keepdim=True)[0]
        branch1_avg_59 = torch.mean(activation_59, dim=1, keepdim=True)
        branch1_conv_59 = self.selu(self.conv_120(activation_59))
        branch1_concat_59 = torch.cat([branch1_avg_59, branch1_max_59, branch1_conv_59], dim=1)
        branch1_output_59 = self.selu(self.conv_122(branch1_concat_59))

        branch2_max_59 = torch.max(encoder3_output, dim=1, keepdim=True)[0]
        branch2_avg_59 = torch.mean(encoder3_output, dim=1, keepdim=True)
        branch2_conv_59 = self.selu(self.conv_121(encoder3_output))
        branch2_concat_59 = torch.cat([branch2_avg_59, branch2_max_59, branch2_conv_59], dim=1)
        branch2_output_59 = self.selu(self.conv_123(branch2_concat_59))

        combined_output_59 = branch1_output_59 + branch2_output_59
        output_sAG_59 = self.sigmoid(self.conv_124(combined_output_59))

        avg_59 = activation_59.mean(dim=(2, 3, 4))
        max_59 = activation_59.amax(dim=(2, 3, 4))
        avg_3 = encoder3_output.mean(dim=(2, 3, 4))
        max_3 = encoder3_output.amax(dim=(2, 3, 4))

        cAG_3 = self.selu(self.dense_54(avg_59)) + self.selu(self.dense_55(max_59)) + \
                self.selu(self.dense_56(avg_3)) + self.selu(self.dense_57(max_3))
        cAG_3 = self.selu(self.dense_58(cAG_3))
        cAG_3 = self.sigmoid(self.dense_59(cAG_3))  # shape: (N, 128)

        dag3_output = output_sAG_59 * activation_59 * cAG_3.view(-1, 128, 1, 1, 1)

        # ========================
        # DAG4 (Level 4 Attention)
        # ========================
        branch1_max_61 = torch.max(activation_61, dim=1, keepdim=True)[0]
        branch1_avg_61 = torch.mean(activation_61, dim=1, keepdim=True)
        branch1_conv_61 = self.selu(self.conv_112(activation_61))
        branch1_concat_61 = torch.cat([branch1_avg_61, branch1_max_61, branch1_conv_61], dim=1)
        branch1_output_61 = self.selu(self.conv_114(branch1_concat_61))

        branch2_max_61 = torch.max(encoder4_output, dim=1, keepdim=True)[0]
        branch2_avg_61 = torch.mean(encoder4_output, dim=1, keepdim=True)
        branch2_conv_61 = self.selu(self.conv_113(encoder4_output))
        branch2_concat_61 = torch.cat([branch2_avg_61, branch2_max_61, branch2_conv_61], dim=1)
        branch2_output_61 = self.selu(self.conv_115(branch2_concat_61))

        combined_output_61 = branch1_output_61 + branch2_output_61
        output_sAG_61 = self.sigmoid(self.conv_116(combined_output_61))

        avg_61 = activation_61.mean(dim=(2, 3, 4))
        max_61 = activation_61.amax(dim=(2, 3, 4))
        avg_4 = encoder4_output.mean(dim=(2, 3, 4))
        max_4 = encoder4_output.amax(dim=(2, 3, 4))

        cAG_4 = self.selu(self.dense_48(avg_61)) + self.selu(self.dense_49(max_61)) + \
                self.selu(self.dense_50(avg_4)) + self.selu(self.dense_51(max_4))
        cAG_4 = self.selu(self.dense_52(cAG_4))
        cAG_4 = self.sigmoid(self.dense_53(cAG_4))  # shape: (N, 256)

        dag4_output = output_sAG_61 * activation_61 * cAG_4.view(-1, 256, 1, 1, 1)

        # ========================
        # Decoder for Output 4
        # ========================
        x = self.conv_117(dag4_output)
        x = F.selu(self.bn_62(x))
        x = self.conv_118(x)
        x = F.selu(self.bn_63(x))
        activation_63 = x
        output_4 = torch.sigmoid(self.conv_119(activation_63))

        # ========================
        # Decoder for Output 3
        # ========================
        transpose_12 = self.conv_transpose_12(activation_63)
        concat_37 = torch.cat([dag3_output, transpose_12], dim=1).float()
        x = self.conv_125(concat_37)
        x = F.selu(self.bn_64(x))
        x = self.conv_126(x)
        x = F.selu(self.bn_65(x))
        activation_65 = x
        output_3 = torch.sigmoid(self.conv_127(activation_65))

        # ========================
        # Decoder for Output 2
        # ========================
        transpose_15 = self.conv_transpose_15(activation_65)
        concat_40 = torch.cat([dag2_output, transpose_15], dim=1).float()
        x = self.conv_133(concat_40)
        x = F.selu(self.bn_66(x))
        x = self.conv_134(x)
        x = F.selu(self.bn_67(x))
        activation_67 = x
        output_2 = torch.sigmoid(self.conv_135(activation_67))

        # ========================
        # Decoder for Output 1
        # ========================
        transpose_17 = self.conv_transpose_17(activation_67)
        concat_43 = torch.cat([dag1_output, transpose_17], dim=1).float()
        x = self.conv_141(concat_43)
        x = F.selu(self.bn_68(x))
        x = self.conv_142(x)
        x = F.selu(self.bn_69(x))
        activation_69 = x
        output_1 = torch.sigmoid(self.conv_143(activation_69))

        # ========================
        # Final fused output
        # ========================
        transpose_16 = self.conv_transpose_16(transpose_15)
        transpose_13 = self.conv_transpose_13(transpose_12)
        transpose_14 = self.conv_transpose_14(transpose_13)

        concat_44 = torch.cat([activation_69, transpose_17, transpose_16, transpose_14], dim=1).float()
        x = self.conv_144(concat_44)
        x = F.selu(self.bn_70(x))
        x = self.conv_145(x)
        x = F.selu(self.bn_71(x))
        output_fused = self.sigmoid(self.conv_146(x))

        return output_fused, output_1, output_2, output_3, output_4