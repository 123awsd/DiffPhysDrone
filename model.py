# DiffDrone Neural Network Controller
# DiffDrone 神经网络控制器
# 
# This module implements the neural network controller for vision-based quadrotor flight.
# It uses a CNN+GRU architecture to process depth images and state vectors,
# outputting control actions (acceleration + velocity prediction).
#
# 该模块实现了基于视觉的四轴飞行器神经网络控制器。
# 它使用CNN+GRU架构来处理深度图像和状态向量，
# 输出控制动作（加速度 + 速度预测）。

import torch
from torch import nn
 
def g_decay(x, alpha):
    """
    Gradient decay function / 梯度衰减函数
    Implements gradient truncation by mixing x with detached version of x
    实现梯度截断，通过将x与detached后的x混合
    
    Forward: x (unchanged) / 前向：x（不变）
    Backward: x * alpha (gradient scaled by alpha) / 反向：x * alpha（梯度按alpha缩放）
    
    Args:
        x: Input tensor / 输入张量
        alpha: Decay coefficient in range [0, 1] / 衰减系数，范围[0, 1]
              - Closer to 1: less decay / 接近1：衰减较小
              - Closer to 0: more decay / 接近0：衰减较大
    
    Returns:
        Mixed tensor for gradient decay / 用于梯度衰减的混合张量
    """
    # Mix x with detached x: x * alpha + x.detach() * (1 - alpha)
    # 将x与detached x混合：x * alpha + x.detach() * (1 - alpha)
    # Forward: returns x unchanged / 前向：返回x不变
    # Backward: gradient flows only through first term, scaled by alpha
    # 反向：梯度只流过第一项，按alpha缩放
    return x * alpha + x.detach() * (1 - alpha)
 
class Model(nn.Module):
    """
    Neural network model for quadrotor obstacle avoidance control
    用于四轴飞行器避障控制的神经网络模型
    
    Architecture:
        CNN Stem (visual feature extraction)
        -> State Projection
        -> Feature Fusion
        -> GRU (temporal modeling)
        -> FC (action output)
    """
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        """
        Initialize neural network controller
        初始化神经网络控制器
        
        Args:
            dim_obs: Dimension of observation vector / 观测向量的维度
                     Default: 9 (actual values: 10 with odom, 7 without odom)
                     默认：9（实际值：有里程计10，无里程计7）
                     - With odom: [local_v(3) + target_v(3) + up_vec(3) + margin(1)]
                     - Without odom: [target_v(3) + up_vec(3) + margin(1)]
            dim_action: Dimension of action vector / 动作向量的维度
                       Default: 4 (actual output is 6: acceleration(3) + velocity(3))
                       默认：4（实际输出为6：加速度3维 + 速度3维）
        """
        super().__init__()
        
        # CNN Stem: Visual feature extraction network
        # 图像特征提取网络(CNN)
        # Extracts spatial features from depth images using convolutional layers
        # 使用卷积层从深度图像中提取空间特征
        self.stem = nn.Sequential(
            # Layer 1: First convolutional layer / 第一卷积层
            # Input: [B, 1, 64, 48] (depth image)
            # Output: [B, 32, 32, 24]
            # Kernel: 2x2, Stride: 2, Bias: False
            # Large stride for fast downsampling and computational efficiency
            # 大步长用于快速降采样和提高计算效率
            nn.Conv2d(1, 32, 2, 2, bias=False),
            # LeakyReLU activation with negative_slope=0.05
            # LeakyReLU激活，负斜率为0.05
            # Prevents dying ReLU problem / 防止ReLU死亡问题
            nn.LeakyReLU(0.05),
            
            # Layer 2: Second convolutional layer / 第二卷积层
            # Input: [B, 32, 32, 24]
            # Output: [B, 64, 30, 22]
            # Kernel: 3x3, Stride: 1, Bias: False
            # No stride to preserve more spatial information
            # 无步长以保留更多空间信息
            nn.Conv2d(32, 64, 3, bias=False),
            nn.LeakyReLU(0.05),
            
            # Layer 3: Third convolutional layer / 第三卷积层
            # Input: [B, 64, 30, 22]
            # Output: [B, 128, 28, 20]
            # Kernel: 3x3, Stride: 1, Bias: False
            nn.Conv2d(64, 128, 3, bias=False),
            nn.LeakyReLU(0.05),
            
            # Flatten: Convert 3D feature maps to 1D vector
            # 展平：将3D特征图转换为1D向量
            # Shape: [B, 128, 28, 20] -> [B, 71680]
            # Note: Actual usage applies max_pool2d first to reduce to [B, 128, 2, 4]
            # 注意：实际使用时先应用max_pool2d降维到[B, 128, 2, 4]
            nn.Flatten(),
            
            # Linear projection: Reduce high-dimensional features to compact representation
            # 线性投影：将高维特征降维为紧凑表示
            # Input: [B, 128*2*4] = [B, 1024] (after pooling)
            # Output: [B, 192]
            nn.Linear(128*2*4, 192, bias=False),
        )
        
        # State projection layer: Project state vector to same dimension as visual features
        # 状态向量投影层：将状态向量投影到与视觉特征相同的维度
        # Input: [B, dim_obs] (e.g., [B, 10] or [B, 7])
        # Output: [B, 192]
        # This enables fusion of visual and state information
        # 这使得视觉和状态信息能够融合
        self.v_proj = nn.Linear(dim_obs, 192)
        # Initialize with smaller weights (0.5x) for stable training
        # 使用较小的权重(0.5倍)初始化，保证训练稳定
        self.v_proj.weight.data.mul_(0.5)
 
        # GRU: Gated Recurrent Unit for temporal modeling
        # GRU循环神经网络：用于时间建模
        # Captures temporal dependencies and maintains memory across timesteps
        # 捕获时间依赖关系，保持跨时间步的记忆
        # Input: [B, 192] (fused features)
        # Hidden state: [B, 192]
        self.gru = nn.GRUCell(192, 192)
        
        # FC (Fully Connected): Final output layer to produce control actions
        # 输出全连接层：生成控制动作的最终输出层
        # Input: [B, 192] (GRU hidden state)
        # Output: [B, dim_action] (e.g., [B, 4] or [B, 6])
        # Projects hidden state to action space
        # 将隐藏状态投影到动作空间
        self.fc = nn.Linear(192, dim_action, bias=False)
        # Initialize with very small weights (0.01x) to prevent large initial actions
        # 使用非常小的权重(0.01倍)初始化，防止初始动作过大
        self.fc.weight.data.mul_(0.01)
        
        # Activation function for intermediate layers
        # 中间层的激活函数
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        """
        Reset model state / 重置模型状态
        Currently a no-op, but can be used to reset model-specific states
        当前为空操作，但可用于重置模型特定状态
        (e.g., RNN hidden states if managed internally)
        （例如，如果在内部管理RNN隐藏状态）
        """
        pass
 
    def forward(self, x: torch.Tensor, v, hx=None):
        """
        Forward pass through the neural network / 网络前向传播
        
        Args:
            x: Depth image tensor / 深度图像张量
               Shape: [B, 1, H, W] or [B, H, W] (will add channel dim)
               形状：[B, 1, H, W] 或 [B, H, W]（会添加通道维度）
               B: batch size / 批次大小
               H, W: height and width of depth image / 深度图像的高度和宽度
               Typically [B, 64, 48] after max_pool2d(4, 4) in main_cuda.py
               通常在main_cuda.py中经过max_pool2d(4, 4)后为[B, 64, 48]
            v: State vector / 状态向量
               Shape: [B, dim_obs]
               形状：[B, dim_obs]
               Contains:
               包含：
               - local_velocity [3]: Local velocity in body frame / 机体坐标系下的本地速度
               - target_velocity [3]: Target velocity / 目标速度
               - up_vector [3]: Up vector from rotation matrix / 旋转矩阵的上向量
               - margin [1]: Safety margin / 安全距离
            hx: Previous GRU hidden state / 前一个GRU隐藏状态
                Shape: [B, 192] or None for initial timestep
                形状：[B, 192] 或 None（初始时间步）
                Used to maintain temporal memory across timesteps
                用于在时间步之间保持时间记忆
        
        Returns:
            act: Control action / 控制动作
                  Shape: [B, dim_action]
                  形状：[B, dim_action]
                  Contains acceleration [3] and velocity prediction [3] when decoded
                  解码后包含加速度[3]和速度预测[3]
            None: Placeholder for value function (not used in this architecture)
                  值函数占位符（在此架构中未使用）
            hx: New GRU hidden state / 新的GRU隐藏状态
                 Shape: [B, 192]
                 形状：[B, 192]
                 Should be passed as hx in next timestep
                 应该在下一个时间步中作为hx传递
        """
        # Extract visual features from depth image using CNN stem
        # 使用CNN stem从深度图像提取视觉特征
        # Shape: [B, H, W] -> [B, 192]
        img_feat = self.stem(x)
        
        # Project state vector to same dimension as visual features
        # 将状态向量投影到与视觉特征相同的维度
        # Shape: [B, dim_obs] -> [B, 192]
        v_proj = self.v_proj(v)
        
        # Fuse visual and state features by element-wise addition
        # 通过逐元素相加融合视觉和状态特征
        # This allows the network to learn how to combine visual and state information
        # 这允许网络学习如何结合视觉和状态信息
        # Shape: [B, 192] + [B, 192] -> [B, 192]
        x = self.act(img_feat + v_proj)
        
        # Process fused features through GRU to capture temporal dependencies
        # 通过GRU处理融合特征，捕获时间依赖关系
        # The GRU maintains a hidden state that carries information across timesteps
        # GRU维护一个隐藏状态，在时间步之间携带信息
        # Shape: [B, 192] -> [B, 192]
        hx = self.gru(x, hx)
        
        # Generate control action from GRU hidden state using FC layer
        # 使用FC层从GRU隐藏状态生成控制动作
        # Shape: [B, 192] -> [B, dim_action]
        act = self.fc(self.act(hx))
        
        return act, None, hx


if __name__ == '__main__':
    Model()
