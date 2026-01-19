# DiffDrone 使用指南 / DiffDrone Usage Guide

## 1. 环境搭建 / Environment Setup

### 1.1 系统要求 / System Requirements

#### 硬件要求 / Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support
  - 推荐 / Recommended: RTX 3090/4090 或更好
  - 最低 / Minimum: GTX 1080Ti 或同等性能
- **内存**: 至少 16GB RAM
- **存储**: 至少 10GB 可用空间

#### 软件要求 / Software Requirements

- **操作系统**: Linux (Ubuntu 18.04+) 或 Windows 10+
- **Python**: 3.11 (已测试 / Tested)
- **CUDA**: 11.8 (已测试 / Tested，其他版本可能兼容）
- **PyTorch**: 2.2.2 (已测试 / Tested，其他版本可能兼容）

### 1.2 依赖安装 / Dependency Installation

#### 1.2.1 创建虚拟环境 / Create Virtual Environment

```bash
# 使用 conda 创建环境 / Create environment using conda
conda create -n diffdrone python=3.11
conda activate diffdrone

# 或使用 venv / Or use venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 / or
venv\Scripts\activate  # Windows
```

#### 1.2.2 安装 PyTorch / Install PyTorch

```bash
# CUDA 11.8 / CUDA 11.8
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# 或根据你的CUDA版本 / Or based on your CUDA version
# 访问 https://pytorch.org/ 获取最新安装命令
# Visit https://pytorch.org/ for latest install commands
```

#### 1.2.3 安装其他依赖 / Install Other Dependencies

```bash
# 安装基本依赖 / Install basic dependencies
pip install matplotlib tqdm tensorboard

# 或使用 requirements.txt（如果提供）
# Or use requirements.txt (if provided)
pip install -r requirements.txt
```

### 1.3 编译 CUDA 扩展 / Compile CUDA Extensions

#### 1.3.1 安装编译工具 / Install Build Tools

**Ubuntu / Linux:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
```

**Windows:**
- 安装 Visual Studio 2019 或更新版本
- 确保安装了 C++ 开发工具
- 安装 CUDA Toolkit 11.8 或更新版本

#### 1.3.2 编译 quadsim_cuda / Compile quadsim_cuda

```bash
# 进入 src 目录 / Enter src directory
cd src

# 安装 CUDA 扩展 / Install CUDA extension
pip install -e .

# 验证安装 / Verify installation
python -c "import quadsim_cuda; print('CUDA extension loaded successfully!')"
```

**如果编译失败 / If compilation fails:**

1. 检查 CUDA 版本是否正确
2. 检查 nvcc 是否在 PATH 中
3. 检查 PyTorch 和 CUDA 版本是否匹配

```bash
# 检查 CUDA 版本 / Check CUDA version
nvcc --version

# 检查 PyTorch CUDA 版本 / Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

#### 1.3.3 测试 CUDA 扩展 / Test CUDA Extension

```bash
# 运行测试脚本 / Run test script
cd src
python test.py

# 预期输出 / Expected output:
# No errors, all assertions pass
```

### 1.4 验证安装 / Verify Installation

```bash
# 运行完整的验证 / Run full verification
python -c "
import torch
import quadsim_cuda
import numpy as np
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('quadsim_cuda available:', quadsim_cuda is not None)
"
```

---

## 2. 训练脚本 / Training Scripts

### 2.1 单智能体训练 / Single Agent Training

#### 2.1.1 基础训练命令 / Basic Training Command

```bash
# 使用单智能体配置 / Use single agent configuration
python main_cuda.py $(cat configs/single_agent.args)
```

#### 2.1.2 单智能体配置文件 / Single Agent Configuration File

`configs/single_agent.args`:
```
--single
--speed_mtp 4
--coef_d_acc 0.01
--coef_d_jerk 0.001
--ground_voxels
--random_rotation
--yaw_drift
--coef_collide 7.5
--coef_obj_avoidance 3.0
--cam_angle 20
--fov_x_half_tan 0.82
```

**参数说明 / Parameter Description:**

| 参数 | 值 | 说明 |
|------|-----|------|
| `--single` | | 启用单智能体模式 |
| `--speed_mtp` | 4 | 速度乘数，影响目标速度范围 |
| `--coef_d_acc` | 0.01 | 加速度平滑损失权重 |
| `--coef_d_jerk` | 0.001 | Jerk平滑损失权重 |
| `--ground_voxels` | | 添加地面障碍物 |
| `--random_rotation` | | 随机旋转场景 |
| `--yaw_drift` | | 添加偏航漂移扰动 |
| `--coef_collide` | 7.5 | 碰撞损失权重 |
| `--coef_obj_avoidance` | 3.0 | 避障损失权重 |
| `--cam_angle` | 20 | 相机俯仰角（度）|
| `--fov_x_half_tan` | 0.82 | 水平视场角半切值 |

#### 2.1.3 自定义训练参数 / Custom Training Parameters

```bash
# 自定义学习率和批次大小 / Custom learning rate and batch size
python main_cuda.py \
    --single \
    --batch_size 128 \
    --num_iters 100000 \
    --lr 5e-4 \
    --speed_mtp 4
```

**常用自定义参数 / Common Custom Parameters:**

```bash
--batch_size <N>           # 批次大小 / Batch size
--num_iters <N>            # 训练迭代次数 / Training iterations
--lr <float>               # 学习率 / Learning rate
--timesteps <N>            # 每个episode的时间步数 / Timesteps per episode
--fov_x_half_tan <float>   # 视场角 / Field of view
--cam_angle <float>         # 相机角度（度）/ Camera angle (degrees)
--speed_mtp <float>        # 速度乘数 / Speed multiplier
--grad_decay <float>        # 梯度衰减系数 / Gradient decay coefficient
```

#### 2.1.4 从检查点恢复训练 / Resume from Checkpoint

```bash
# 从检查点恢复 / Resume from checkpoint
python main_cuda.py \
    --resume checkpoint0001.pth \
    $(cat configs/single_agent.args)
```

### 2.2 多智能体训练 / Multi-Agent Training

#### 2.2.1 基础训练命令 / Basic Training Command

```bash
# 使用多智能体配置 / Use multi-agent configuration
python main_cuda.py $(cat configs/multi_agent.args)
```

#### 2.2.2 多智能体配置文件 / Multi Agent Configuration File

`configs/multi_agent.args`:
```
--gate
--coef_collide 5.0
--coef_obj_avoidance 2.0
--batch 256
--fov_x_half_tan 0.82
--timesteps 180
```

**参数说明 / Parameter Description:**

| 参数 | 值 | 说明 |
|------|-----|------|
| `--gate` | | 添加门任务 |
| `--coef_collide` | 5.0 | 碰撞损失权重 |
| `--coef_obj_avoidance` | 2.0 | 避障损失权重 |
| `--batch` | 256 | 批次大小（多智能体需要更大批次）|
| `--fov_x_half_tan` | 0.82 | 水平视场角 |
| `--timesteps` | 180 | 每个episode的时间步数 |

#### 2.2.3 自定义多智能体训练 / Custom Multi-Agent Training

```bash
# 自定义多智能体训练参数 / Custom multi-agent training
python main_cuda.py \
    --gate \
    --batch_size 512 \
    --timesteps 200 \
    --coef_collide 7.0 \
    --coef_obj_avoidance 3.0 \
    --fov_x_half_tan 0.9
```

---

## 3. 监控训练 / Monitoring Training

### 3.1 TensorBoard 可视化 / TensorBoard Visualization

#### 3.1.1 启动 TensorBoard / Start TensorBoard

```bash
# 启动 TensorBoard / Start TensorBoard
tensorboard --logdir runs

# 或指定端口 / Or specify port
tensorboard --logdir runs --port 6006

# 在浏览器中访问 / Access in browser
# http://localhost:6006
```

#### 3.1.2 查看训练指标 / View Training Metrics

**可用的指标 / Available Metrics:**

- **损失指标 / Loss Metrics:**
  - `loss`: 总损失
  - `loss_v`: 速度跟踪损失
  - `loss_v_pred`: 速度预测损失
  - `loss_obj_avoidance`: 避障损失
  - `loss_collide`: 碰撞损失
  - `loss_d_acc`: 加速度平滑损失
  - `loss_d_jerk`: Jerk平滑损失
  - `loss_bias`: 轨迹偏差损失

- **性能指标 / Performance Metrics:**
  - `success`: 成功率（无碰撞）|
  - `max_speed`: 最大速度
  - `avg_speed`: 平均速度
  - `ar`: 平均奖励（success × avg_speed）

- **可视化图表 / Visualization Charts:**
  - `p_history`: 位置历史轨迹
  - `v_history`: 速度历史
  - `a_reals`: 真实加速度历史

### 3.2 训练进度监控 / Training Progress Monitoring

#### 3.2.1 命令行输出 / Command Line Output

训练时会显示进度条：

```bash
loss: 2.345: 100%|██████████| 50000/50000 [02:15<00:00, 367.34it/s]
```

进度条显示：
- 当前损失值
- 当前进度
- 预计剩余时间
- 每秒迭代次数

#### 3.2.2 检查点保存 / Checkpoint Saving

```bash
# 检查点自动保存在当前目录 / Checkpoints auto-saved in current directory
ls -la checkpoint*.pth

# 输出示例 / Output example:
# checkpoint0001.pth
# checkpoint0002.pth
# checkpoint0003.pth
# ...
```

---

## 4. 评估和测试 / Evaluation and Testing

### 4.1 模型评估 / Model Evaluation

#### 4.1.1 评估脚本 / Evaluation Script

```bash
# 评估训练好的模型 / Evaluate trained model
python eval.py --resume checkpoint0005.pth --target_speed 2.5
```

**参数说明 / Parameter Description:**

| 参数 | 说明 |
|------|------|
| `--resume` | 模型检查点路径 |
| `--target_speed` | 目标速度（m/s）|

#### 4.1.2 评估指标 / Evaluation Metrics

评估脚本会输出：
- 成功率（无碰撞完成任务的百分比）
- 平均速度
- 最大速度
- 碰撞次数
- 任务完成时间

### 4.2 仿真器评估 / Simulator Evaluation

#### 4.2.1 下载仿真验证代码 / Download Simulation Validation Code

```bash
# 从GitHub Release页面下载
# Download from GitHub Release page
# 访问: https://github.com/henryhuyu/DiffDrone/releases
```

#### 4.2.2 启动仿真器 / Launch Simulator

**Linux:**
```bash
cd <path to multi agent code supplementary>
./LinuxNoEditor/Blocks.sh \
    -ResX=896 \
    -ResY=504 \
    -windowed \
    -WinX=512 \
    -WinY=304 \
    -settings=$PWD/settings.json
```

**Windows:**
```bash
cd <path to multi agent code supplementary>
.\Blocks.exe \
    -ResX=896 \
    -ResY=504 \
    -windowed \
    -WinX=512 \
    -WinY=304 \
    -settings=$PWD/settings.json
```

**参数说明 / Parameter Description:**

| 参数 | 说明 |
|------|------|
| `-ResX` | 水平分辨率 |
| `-ResY` | 垂直分辨率 |
| `-windowed` | 窗口模式运行 |
| `-WinX` | 窗口位置X |
| `-WinY` | 窗口位置Y |
| `-settings` | 设置文件路径 |

---

## 5. 参数调优 / Parameter Tuning

### 5.1 损失权重调优 / Loss Weight Tuning

#### 5.1.1 调优策略 / Tuning Strategy

**步骤 / Steps:**

1. **初始设置**: 使用默认参数
2. **观察训练**: 监控各损失项
3. **调整权重**: 根据观察结果调整
4. **验证效果**: 评估调整后的模型

**示例 / Example:**

```bash
# 如果避障损失太高，增加避障权重
# If obstacle avoidance loss is too high, increase obstacle avoidance weight
python main_cuda.py \
    --coef_obj_avoidance 5.0 \
    --coef_collide 10.0 \
    $(cat configs/single_agent.args)

# 如果轨迹不够平滑，增加平滑权重
# If trajectory is not smooth enough, increase smoothing weights
python main_cuda.py \
    --coef_d_acc 0.05 \
    --coef_d_jerk 0.005 \
    $(cat configs/single_agent.args)
```

#### 5.1.2 损失权重参考 / Loss Weight Reference

| 场景 | coef_obj_avoidance | coef_collide | coef_d_acc | coef_d_jerk |
|------|-------------------|--------------|------------|-------------|
| 简单环境 | 1.0 | 1.0 | 0.01 | 0.001 |
| 复杂障碍物 | 3.0-5.0 | 5.0-10.0 | 0.01 | 0.001 |
| 高速飞行 | 2.0 | 5.0 | 0.02 | 0.002 |
| 精细控制 | 1.5 | 2.0 | 0.05 | 0.005 |

### 5.2 学习率调优 / Learning Rate Tuning

#### 5.2.1 学习率选择 / Learning Rate Selection

**常用学习率 / Common Learning Rates:**

- `1e-3`: 默认，适用于大多数情况
- `5e-4`: 更保守，收敛更稳定
- `2e-3`: 更激进，收敛更快但不稳定
- `1e-4`: 非常保守，用于微调

**调整建议 / Tuning Suggestions:**

```bash
# 损失不下降，尝试降低学习率
# If loss doesn't decrease, try lowering learning rate
python main_cuda.py --lr 5e-4 $(cat configs/single_agent.args)

# 损下降太慢，尝试提高学习率
# If loss decreases too slowly, try increasing learning rate
python main_cuda.py --lr 2e-3 $(cat configs/single_agent.args)
```

#### 5.2.2 学习率调度 / Learning Rate Scheduling

代码中已经实现了余弦退火调度：

```python
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)
```

默认行为：
- 初始学习率: `args.lr`
- 最终学习率: `args.lr * 0.01`
- 调度方式: 余弦退火

### 5.3 批次大小调优 / Batch Size Tuning

#### 5.3.1 批次大小选择 / Batch Size Selection

**考虑因素 / Considerations:**

- **GPU内存**: 批次越大，内存占用越高
- **训练稳定性**: 批次越大，梯度估计越稳定
- **训练速度**: 批次越大，每步耗时越长，但总步数可能更少

**推荐配置 / Recommended Configurations:**

| GPU内存 | 批次大小 | 适用场景 |
|---------|----------|----------|
| 8GB | 32-64 | 单智能体，简单环境 |
| 12GB | 64-128 | 单智能体，复杂环境 |
| 16GB | 128-256 | 多智能体，中等环境 |
| 24GB+ | 256-512 | 多智能体，复杂环境 |

#### 5.3.2 调整批次大小 / Adjust Batch Size

```bash
# 根据GPU内存调整 / Adjust based on GPU memory
python main_cuda.py --batch_size 128 $(cat configs/single_agent.args)
```

### 5.4 时间步数调优 / Timesteps Tuning

#### 5.4.1 时间步数选择 / Timesteps Selection

**考虑因素 / Considerations:**

- **任务长度**: 任务越长，需要的时间步越多
- **训练成本**: 时间步越多，每步耗时越长
- **策略学习**: 时间步太少，难以学习长期策略

**推荐配置 / Recommended Configurations:**

| 任务类型 | 推荐时间步数 |
|----------|--------------|
| 短距离导航 | 100-150 |
| 中距离导航 | 150-200 |
| 长距离导航 | 200-300 |
| 复杂任务 | 300+ |

#### 5.4.2 调整时间步数 / Adjust Timesteps

```bash
# 根据任务复杂度调整 / Adjust based on task complexity
python main_cuda.py --timesteps 200 $(cat configs/single_agent.args)
```

---

## 6. 故障排除 / Troubleshooting

### 6.1 常见问题 / Common Issues

#### 6.1.1 CUDA 相关问题 / CUDA Related Issues

**问题**: `RuntimeError: CUDA out of memory`

**解决方案 / Solution:**
```bash
# 减小批次大小 / Reduce batch size
python main_cuda.py --batch_size 32 $(cat configs/single_agent.args)

# 或减小时间步数 / Or reduce timesteps
python main_cuda.py --timesteps 100 $(cat configs/single_agent.args)
```

**问题**: `ImportError: cannot import name 'quadsim_cuda'`

**解决方案 / Solution:**
```bash
# 重新编译CUDA扩展 / Re-compile CUDA extension
cd src
pip install -e .

# 验证CUDA版本 / Verify CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

#### 6.1.2 训练相关问题 / Training Related Issues

**问题**: 损失不下降或震荡

**解决方案 / Solution:**
```bash
# 降低学习率 / Lower learning rate
python main_cuda.py --lr 5e-4 $(cat configs/single_agent.args)

# 检查损失权重 / Check loss weights
# 确保各项损失平衡 / Ensure loss terms are balanced
```

**问题**: 训练成功率高但速度慢

**解决方案 / Solution:**
```bash
# 增加速度乘数 / Increase speed multiplier
python main_cuda.py --speed_mtp 5.0 $(cat configs/single_agent.args)

# 减小平滑损失权重 / Reduce smoothing loss weights
python main_cuda.py \
    --coef_d_acc 0.005 \
    --coef_d_jerk 0.0005 \
    $(cat configs/single_agent.args)
```

#### 6.1.3 环境相关问题 / Environment Related Issues

**问题**: 碰撞率过高

**解决方案 / Solution:**
```bash
# 增加避障损失权重 / Increase obstacle avoidance loss weights
python main_cuda.py \
    --coef_obj_avoidance 5.0 \
    --coef_collide 10.0 \
    $(cat configs/single_agent.args)

# 降低速度 / Reduce speed
python main_cuda.py --speed_mtp 2.0 $(cat configs/single_agent.args)
```

### 6.2 性能优化 / Performance Optimization

#### 6.2.1 训练速度优化 / Training Speed Optimization

**建议 / Suggestions:**

1. **使用更大的批次大小**
   - 充分利用GPU并行能力
   - 减少总迭代次数

2. **减少时间步数**
   - 初期训练可以使用较少时间步
   - 后期再增加

3. **减少保存频率**
   - 修改代码中的保存逻辑
   - 减少I/O开销

#### 6.2.2 内存优化 / Memory Optimization

**建议 / Suggestions:**

1. **使用混合精度训练**
   - 虽然代码中未实现，但可以添加
   - 显著减少内存占用

2. **梯度累积**
   - 小批次累积梯度
   - 模拟大批次效果

3. **及时清理不需要的变量**
   - 使用 `del` 释放内存
   - 定期调用 `torch.cuda.empty_cache()`

---

## 7. 高级用法 / Advanced Usage

### 7.1 自定义环境 / Custom Environment

#### 7.1.1 修改环境参数 / Modify Environment Parameters

```python
# 在 env_cuda.py 中修改 / Modify in env_cuda.py
def reset(self):
    # 自定义障碍物数量 / Customize number of obstacles
    self.balls = torch.rand((B, 50, 4), device=device)  # 30 -> 50

    # 自定义速度范围 / Customize speed range
    self.max_speed = (1.0 + 3.0 * rd) * self.speed_mtp

    # 自定义障碍物位置范围 / Customize obstacle position range
    self.ball_w = torch.tensor([10., 20, 8, 0.3], device=device)
```

#### 7.1.2 添加新的障碍物类型 / Add New Obstacle Types

```python
# 在 quadsim_kernel.cu 中添加 / Add in quadsim_kernel.cu
// 添加新的障碍物类型相交测试
// Add intersection test for new obstacle type
for (int i = 0; i < new_obstacles.size(1); i++) {
    // 实现相交测试 / Implement intersection test
}
```

### 7.2 自定义损失函数 / Custom Loss Functions

#### 7.2.1 添加新的损失项 / Add New Loss Term

```python
# 在 main_cuda.py 中添加 / Add in main_cuda.py

# 示例: 添加高度保持损失
# Example: Add altitude maintenance loss
target_height = 1.5  # 目标高度 / Target altitude
loss_height = F.smooth_l1_loss(
    p_history[..., 2],
    torch.ones_like(p_history[..., 2]) * target_height
)

# 添加到总损失 / Add to total loss
loss = loss + args.coef_height * loss_height
```

#### 7.2.2 修改现有损失函数 / Modify Existing Loss Functions

```python
# 示例: 修改避障损失为指数函数
# Example: Change obstacle avoidance loss to exponential
# 原始 / Original:
# loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)

# 修改后 / Modified:
loss_obj_avoidance = torch.exp(-distance[:, 1:]).mul(v_to_pt).mean()
```

### 7.3 自定义网络架构 / Custom Network Architecture

#### 7.3.1 修改CNN Stem / Modify CNN Stem

```python
# 在 model.py 中修改 / Modify in model.py
class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            # 添加更多卷积层 / Add more convolution layers
            nn.Conv2d(1, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(32),  # 添加BatchNorm
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*8*6, 192, bias=False),
        )
```

#### 7.3.2 替换为其他架构 / Replace with Other Architecture

```python
# 示例: 使用ResNet作为视觉编码器
# Example: Use ResNet as visual encoder
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        # 使用预训练的ResNet / Use pretrained ResNet
        resnet = models.resnet18(pretrained=False)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),  # 修改第一层 / Modify first layer
            *list(resnet.children())[1:-1]  # 移除最后的全连接层 / Remove final FC layer
        )
        # ... 其他部分保持不变 / Other parts unchanged
```

---

## 8. 最佳实践 / Best Practices

### 8.1 训练流程 / Training Workflow

#### 8.1.1 推荐的训练流程 / Recommended Training Workflow

1. **准备阶段**
   - 验证环境安装正确
   - 运行测试脚本确认CUDA扩展正常
   - 准备配置文件

2. **初步训练**
   - 使用默认参数进行初步训练
   - 监控TensorBoard观察训练动态
   - 确保损失正常下降

3. **调优阶段**
   - 根据训练效果调整参数
   - 重点调整损失权重
   - 优化学习率和批次大小

4. **评估阶段**
   - 使用验证集评估模型
   - 在仿真器中测试真实性能
   - 记录各项指标

5. **部署阶段**
   - 选择最佳检查点
   - 在真实环境中测试（如果可用）
   - 准备发布

### 8.2 数据管理 / Data Management

#### 8.2.1 检查点管理 / Checkpoint Management

```bash
# 创建备份 / Create backup
mkdir checkpoints_backup
cp checkpoint*.pth checkpoints_backup/

# 清理旧检查点 / Clean old checkpoints
# 保留最近的5个检查点
ls -t checkpoint*.pth | tail -n +6 | xargs rm -f
```

#### 8.2.2 TensorBoard日志管理 / TensorBoard Log Management

```bash
# 清理旧日志 / Clean old logs
# 保留最近10次运行的日志
ls -t runs/ | tail -n +11 | xargs rm -rf
```

### 8.3 资源监控 / Resource Monitoring

#### 8.3.1 GPU监控 / GPU Monitoring

```bash
# 实时监控GPU使用情况 / Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# 或使用 gpustat / Or use gpustat
pip install gpustat
gpustat -i
```

#### 8.3.2 系统监控 / System Monitoring

```bash
# 监控CPU和内存 / Monitor CPU and memory
htop

# 监控磁盘使用 / Monitor disk usage
df -h

# 监控网络 / Monitor network
iftop
```

---

## 9. 常见问题 FAQ / Frequently Asked Questions

### Q1: 训练需要多长时间？/ How long does training take?

**A**: 取决于硬件和配置。
- RTX 3090: ~2-3小时完成50,000次迭代
- RTX 4090: ~1-2小时完成50,000次迭代
- GTX 1080Ti: ~4-6小时完成50,000次迭代

### Q2: 可以在CPU上训练吗？/ Can I train on CPU?

**A**: 不推荐。虽然代码理论上可以在CPU上运行，但：
- 速度极慢（可能慢100倍以上）
- CUDA扩展无法使用
- 不支持深度图像渲染的加速

### Q3: 如何选择合适的配置文件？/ How to choose the right config file?

**A**:
- 单智能体简单任务: `configs/single_agent.args`
- 多智能体协同任务: `configs/multi_agent.args`
- 根据具体任务需求修改参数

### Q4: 训练成功率高但实际效果差？/ High success rate but poor actual performance?

**A**: 可能原因：
1. 过拟合训练环境
2. 使用了过多的域随机化
3. 损失权重不平衡
4. 需要在真实环境中微调

### Q5: 如何改进泛化能力？/ How to improve generalization?

**A**:
1. 增加域随机化（更多样化的环境参数）
2. 使用更大的数据集
3. 增加训练迭代次数
4. 在测试时进行在线微调

### Q6: 能否用于真实的四轴飞行器？/ Can this be used for real quadrotors?

**A**: 可以，但需要：
1. Sim-to-Real迁移策略
2. 在真实环境中微调
3. 处理传感器噪声和延迟
4. 安全验证和测试

---

## 10. 总结 / Summary

本指南涵盖了DiffDrone的完整使用流程：

This guide covers the complete usage workflow of DiffDrone:

1. **环境搭建**: 安装依赖和编译CUDA扩展
2. **训练脚本**: 单智能体和多智能体训练
3. **监控训练**: TensorBoard可视化和进度监控
4. **评估测试**: 模型评估和仿真器测试
5. **参数调优**: 损失权重、学习率、批次大小等
6. **故障排除**: 常见问题和解决方案
7. **高级用法**: 自定义环境和网络架构
8. **最佳实践**: 训练流程和数据管理

通过本指南，您应该能够：
- 成功搭建DiffDrone环境
- 训练自己的模型
- 调优和改进性能
- 解决常见问题

祝您使用愉快！Happy coding!

如有问题，请查阅：
- 论文: https://www.nature.com/articles/s42256-025-01048-0
- 项目网页: https://henryhuyu.github.io/DiffPhysDrone_Web/
- GitHub Issues: https://github.com/henryhuyu/DiffDrone/issues
