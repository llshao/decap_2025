import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from typing import Optional, Tuple



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


class CategoricalMasked(Categorical):
    """A categorical distribution that supports invalid actions via boolean masks."""
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None):
        # If no masks are supplied fall back to the default behaviour
        if masks is None:
            super().__init__(probs=probs, logits=logits, validate_args=validate_args)
            self.masks = None
        else:
            # Ensure mask is boolean and on the same device as logits
            self.masks = masks.to(dtype=torch.bool, device=logits.device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8, device=logits.device))
            super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def entropy(self):
        # Delegate to parent implementation when no mask is provided
        if self.masks is None:
            return super().entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0, device=self.logits.device))
        return -p_log_p.sum(-1)

class Encoder(nn.Module):
    
    def __init__(self, input_channels: int, use_batch_norm: bool = True):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # 第一个卷积块
        self.conv1 = layer_init(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1))
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 第二个卷积块
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 第三个卷积块
        self.conv3 = layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 第四个卷积块
        self.conv4 = layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.bn4 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()
        
        # 使用更稳定的激活函数
        self.activation = nn.ReLU(inplace=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 第一层：11x11 -> 6x6
        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        
        # 第二层：6x6 -> 3x3
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        
        # 第三层：3x3 -> 2x2
        x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        
        # 第四层
        x = self.activation(self.bn4(self.conv4(x)))
        
        return x  # 输出: [batch, 256, 2, 2]

class Decoder(nn.Module):
    
    def __init__(self, output_channels: int, use_batch_norm: bool = True):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        self.output_channels = output_channels
        
        # 上采样层1: 2x2 -> 4x4
        self.deconv1 = layer_init(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.bn1 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()
        
        # 上采样层2: 4x4 -> 8x8
        self.deconv2 = layer_init(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.bn2 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        
        # 上采样层3: 8x8 -> 11x11
        self.deconv3 = layer_init(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0))
        self.bn3 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        
        # 精细化层
        self.conv1 = layer_init(nn.Conv2d(64, 32, kernel_size=3, padding=1))
        self.bn4 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        
        # 输出层 - 对于双头架构，输出2个通道
        self.conv_out = layer_init(nn.Conv2d(32, output_channels, kernel_size=3, padding=1))
        
        # 全局平均池化层 - 将空间维度从11x11压缩到11维
        self.global_pool = nn.AdaptiveAvgPool2d((1, 11))  # [batch, output_channels, 1, 11]
        
        # 激活函数
        self.activation = nn.ReLU(inplace=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 上采样序列
        x = self.activation(self.bn1(self.deconv1(x)))  # 2x2 -> 4x4
        x = self.activation(self.bn2(self.deconv2(x)))  # 4x4 -> 8x8
        x = self.activation(self.bn3(self.deconv3(x)))  # 8x8 -> 11x11
        
        # 精细化
        x = self.activation(self.bn4(self.conv1(x)))
        
        # 输出层
        x = self.conv_out(x)  # [batch, output_channels, 11, 11]
        
        # 全局平均池化 - 压缩空间维度
        x = self.global_pool(x)  # [batch, output_channels, 1, 11]
        x = x.squeeze(2)  # [batch, output_channels, 11]
        
        return x


class PPONetwork(nn.Module):
    """
    优化后的PPO网络
    改进了内存使用、计算效率和数值稳定性
    """
    
    def __init__(self, 
                 env, 
                 input_channels: int = 5, 
                 use_batch_norm: bool = False,
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.action_space = env.ACTION_SPACE
        self.input_channels = input_channels
        self.use_batch_norm = use_batch_norm
        
        # 阻抗处理网络（添加dropout以防止过拟合）
        self.imped_fc = nn.Sequential(
            layer_init(nn.Linear(4 * 231, 512), std=1),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            layer_init(nn.Linear(512, 2 * 121), std=1)
        )
        
        # 编码器（输入通道数包括原始输入和阻抗特征）
        self.encoder = Encoder(input_channels + 2, use_batch_norm)
        
        # Actor网络（策略网络）- 使用双头架构：行和列
        self.actor = Decoder(2, use_batch_norm)  # 2 heads: row and column
        self.action_dim = 11  # Each head has 11 dimensions
        
        # Critic网络（价值网络)
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(256 * 2 * 2, 512), std=1),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            layer_init(nn.Linear(512, 256), std=1),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            layer_init(nn.Linear(256, 1), std=1),
        )
    
    def _process_impedance(self, imped: torch.Tensor) -> torch.Tensor:
        """处理阻抗输入"""
        batch_size = imped.shape[0]
        processed = self.imped_fc(imped)
        return processed.reshape(batch_size, 2, 11, 11)

    def _action_to_row_col(self, action_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将动作索引转换为行列坐标"""
        row = action_idx // self.action_dim
        col = action_idx % self.action_dim
        return row, col

    def _row_col_to_action(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """将行列坐标转换为动作索引"""
        return row * self.action_dim + col

    def _compute_joint_logits(self, row_logits: torch.Tensor, col_logits: torch.Tensor) -> torch.Tensor:
        """计算联合概率的logits (121维)"""
        batch_size = row_logits.shape[0]
        
        # 计算联合概率: p(i,j) = p_row(i) * p_col(j)
        # 在log空间: log(p(i,j)) = log(p_row(i)) + log(p_col(j))
        row_probs = torch.softmax(row_logits, dim=-1)  # [batch, 11]
        col_probs = torch.softmax(col_logits, dim=-1)  # [batch, 11]
        
        # 计算所有121个组合的联合概率
        # 使用广播机制处理批次维度
        row_probs_expanded = row_probs.unsqueeze(2)  # [batch, 11, 1]
        col_probs_expanded = col_probs.unsqueeze(1)  # [batch, 1, 11]
        joint_probs = row_probs_expanded * col_probs_expanded  # [batch, 11, 11]
        joint_probs = joint_probs.reshape(batch_size, -1)  # [batch, 121]
        
        # 转换回logits空间
        joint_logits = torch.log(joint_probs + 1e-8)  # 添加小常数避免log(0)
        
        return joint_logits


    def _encode_features(self, x: torch.Tensor, imped: torch.Tensor) -> torch.Tensor:
        """
            编码特征
        """
        # 处理阻抗特征
        imped_features = self._process_impedance(imped)
        
        # 拼接输入特征
        combined_input = torch.cat((x, imped_features), dim=1)
        
        # 编码
        encoded_features = self.encoder(combined_input)
        
        return encoded_features

    def get_value(self, x: torch.Tensor, imped: torch.Tensor) -> torch.Tensor:
        """
        获取状态价值
        
        Args:
            x: 状态特征 [batch, channels, height, width]
            imped: 阻抗特征 [batch, 4*231]
            
        Returns:
            状态价值 [batch, 1]
        """
        encoded_features = self._encode_features(x, imped)
        return self.critic(encoded_features)

    def get_action_and_value(self, 
                           x: torch.Tensor, 
                           imped: torch.Tensor, 
                           action_mask: torch.Tensor, 
                           action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作和价值 - 使用双头架构
        
        Args:
            x: 状态特征
            imped: 阻抗特征
            action_mask: 动作掩码 (121维)
            action: 可选的指定动作
            
        Returns:
            (动作, 对数概率, 熵, 价值)
        """
        # 编码特征
        encoded_features = self._encode_features(x, imped)
        
        # 获取双头logits [batch, 2, 11]
        dual_head_logits = self.actor(encoded_features)
        
        # 分离行和列的logits
        row_logits = dual_head_logits[:, 0, :]  # [batch, 11]
        col_logits = dual_head_logits[:, 1, :]  # [batch, 11]
        
        # 计算联合概率的logits (121维)
        joint_logits = self._compute_joint_logits(row_logits, col_logits)  # [batch, 121]
        
        # 应用原始121维掩码
        masked_logits = torch.where(action_mask.bool(), joint_logits, torch.tensor(-1e8, device=joint_logits.device))
        
        # 创建掩码分类分布
        categorical = CategoricalMasked(logits=masked_logits, masks=action_mask.bool())
        
        # 采样或使用指定动作
        if action is None:
            action = categorical.sample()
        
        # 计算对数概率和熵
        logprob = categorical.log_prob(action)
        entropy = categorical.entropy()
        
        # 获取价值
        value = self.critic(encoded_features)
        
        return action, logprob, entropy, value
    

