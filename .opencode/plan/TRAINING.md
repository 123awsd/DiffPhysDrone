# DiffDrone 训练算法文档 / DiffDrone Training Algorithm Document

## 1. 网络架构 / Network Architecture

### 1.1 整体架构 / Overall Architecture

DiffDrone使用**CNN + GRU**的序列到序列架构来处理视觉输入：

DiffDrone uses **CNN + GRU** sequence-to-sequence architecture to process visual inputs:

```
输入序列 (Input Sequence): [depth_t, state_t] for t = 0, 1, ..., T
    │
    ├─► 视觉路径 / Visual Path
    │   深度图像 [B,1,64,48]
    │       │
    │       ├─► CNN Stem (特征提取)
    │       │   Conv2d(1→32, 2×2, stride=2) → [B,32,32,24]
    │       │   LeakyReLU(0.05)
    │       │   Conv2d(32→64, 3×3) → [B,64,30,22]
    │       │   LeakyReLU(0.05)
    │       │   Conv2d(64→128, 3×3) → [B,128,28,20]
    │       │   LeakyReLU(0.05)
    │       │   Flatten + Linear → [B,192]
    │       │
    │       └─► 视觉特征 img_feat [B,192]
    │
    ├─► 状态路径 / State Path
    │   状态向量 [B,10]
    │       │
    │       └─► 线性投影 v_proj(dim_obs→192) → [B,192]
    │
    ├─► 特征融合 / Feature Fusion
    │   img_feat + v_proj → [B,192]
    │       │
    │       └─► LeakyReLU(0.05)
    │
    ├─► 序列建模 / Sequence Modeling
    │   GRUCell(192→192)
    │       │
    │       ├─► 输入: [B,192] (融合特征)
    │       ├─► 隐藏状态: [B,192]
    │       └─► 输出: [B,192] (新隐藏状态)
    │
    └─► 动作输出 / Action Output
        FC(192→6) → [B,6]
            │
            └─► 分解为:
                - 加速度 [B,3]
                - 速度预测 [B,3]
```

### 1.2 CNN Stem 详细设计 / CNN Stem Detailed Design

#### 1.2.1 第一卷积层 / First Convolution Layer

```python
nn.Conv2d(1, 32, 2, 2, bias=False)
```

**参数详解 / Parameter Details:**
- 输入通道: 1 (深度图像)
- 输出通道: 32 (特征图数量)
- 卷积核大小: 2×2
- 步长: 2 (下采样)
- 偏置: False (使用BatchNorm时不需要)

**输出维度 / Output Dimension:**
```
输入: [B, 1, 64, 48]
输出: [B, 32, 32, 24]
```

**设计理由 / Design Rationale:**
- 大步长(2)快速降低分辨率，减少计算量
- 小卷积核(2×2)保持局部特征
- 无偏置简化模型参数

#### 1.2.2 第二卷积层 / Second Convolution Layer

```python
nn.Conv2d(32, 64, 3, bias=False)
```

**参数详解 / Parameter Details:**
- 输入通道: 32
- 输出通道: 64
- 卷积核大小: 3×3
- 步长: 1 (默认，保持分辨率)

**输出维度 / Output Dimension:**
```
输入: [B, 32, 32, 24]
输出: [B, 64, 30, 22]
```

#### 1.2.3 第三卷积层 / Third Convolution Layer

```python
nn.Conv2d(64, 128, 3, bias=False)
```

**参数详解 / Parameter Details:**
- 输入通道: 64
- 输出通道: 128
- 卷积核大小: 3×3
- 步长: 1

**输出维度 / Output Dimension:**
```
输入: [B, 64, 30, 22]
输出: [B, 128, 28, 20]
```

#### 1.2.4 展平和线性层 / Flatten and Linear Layer

```python
nn.Flatten()
nn.Linear(128*2*4, 192, bias=False)
```

**参数详解 / Parameter Details:**
- 展平后维度: 128 × 28 × 20 = 71680 (太大！)
- 实际使用池化后: 128 × 2 × 4 = 1024

**代码实现 / Code Implementation:**
```python
# 在main_cuda.py中进行池化
x = F.max_pool2d(x[:, None], 4, 4)
# 128×28×20 → 128×7×5
# 然后展平为128×7×5=4480
# 但文档中写的是128×2×4，可能是不同的配置
```

### 1.3 GRU 详细设计 / GRU Detailed Design

#### 1.3.1 GRU 单元 / GRU Unit

```python
nn.GRUCell(192, 192)
```

**参数详解 / Parameter Details:**
- 输入维度: 192 (融合特征)
- 隐藏状态维度: 192

**GRU 公式 / GRU Formulas:**

更新门 / Update Gate:
$$ \mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z) $$

重置门 / Reset Gate:
$$ \mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r) $$

候选隐藏状态 / Candidate Hidden State:
$$ \tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) $$

新隐藏状态 / New Hidden State:
$$ \mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t $$

其中 $\odot$ 是逐元素乘法，$\sigma$ 是sigmoid函数。

Where $\odot$ is element-wise multiplication, $\sigma$ is sigmoid function.

**设计理由 / Design Rationale:**
- GRU比LSTM参数更少，训练更快
- 适合处理时间序列数据
- 隐藏状态保持历史信息

### 1.4 输出层详细设计 / Output Layer Detailed Design

```python
nn.Linear(192, 6, bias=False)
```

**参数详解 / Parameter Details:**
- 输入维度: 192 (GRU隐藏状态)
- 输出维度: 6
  - 前3维: 加速度 [ax, ay, az]
  - 后3维: 速度预测 [vx, vy, vz]
- 偏置: False

**权重初始化 / Weight Initialization:**
```python
self.fc.weight.data.mul_(0.01)  # 小权重初始化
```

**设计理由 / Design Rationale:**
- 小权重初始化防止梯度爆炸
- 无偏置简化模型
- 输出维度匹配物理控制需求

---

## 2. 损失函数设计 / Loss Function Design

### 2.1 总损失函数 / Total Loss Function

$$ L_{total} = w_v L_v + w_{v\_pred} L_{v\_pred} + w_{avoid} L_{avoid} + w_{collide} L_{collide} + w_{acc} L_{acc} + w_{jerk} L_{jerk} + w_{bias} L_{bias} $$

其中 $w_x$ 是各损失项的权重。

Where $w_x$ are weights for each loss term.

**代码实现 / Code Implementation:**
```python
loss = (args.coef_v * loss_v +
        args.coef_obj_avoidance * loss_obj_avoidance +
        args.coef_bias * loss_bias +
        args.coef_d_acc * loss_d_acc +
        args.coef_d_jerk * loss_d_jerk +
        args.coef_d_snap * loss_d_snap +
        args.coef_speed * loss_speed +
        args.coef_v_pred * loss_v_pred +
        args.coef_collide * loss_collide +
        loss_ground_affinity)
```

### 2.2 速度跟踪损失 / Velocity Tracking Loss

#### 2.2.1 平滑L1损失 / Smooth L1 Loss

$$ L_v = \text{SmoothL1}(\bar{\mathbf{v}} - \mathbf{v}_{target}) $$

其中 $\bar{\mathbf{v}}$ 是移动平均速度：

Where $\bar{\mathbf{v}}$ is moving average velocity:

$$ \bar{\mathbf{v}}_t = \frac{\sum_{i=t-30}^{t} \mathbf{v}_i}{30} $$

**代码实现 / Code Implementation:**
```python
# 计算移动平均速度 / Calculate moving average velocity
v_history_cum = v_history.cumsum(0)
v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30

# 计算速度跟踪损失 / Calculate velocity tracking loss
delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))
```

**设计理由 / Design Rationale:**
- 平滑损失减少对异常值的敏感度
- 移动平均减少瞬时噪声影响
- 鼓励四轴飞行器以目标速度飞行

### 2.3 速度预测损失 / Velocity Prediction Loss

#### 2.3.1 均方误差损失 / Mean Squared Error Loss

$$ L_{v\_pred} = \frac{1}{BT} \sum_{t=0}^{T-1} \sum_{i=1}^{B} ||\mathbf{v}_{pred}(t,i) - \mathbf{v}(t,i)||^2 $$

**代码实现 / Code Implementation:**
```python
loss_v_pred = F.mse_loss(v_preds, v_history.detach())
```

**设计理由 / Design Rationale:**
- 当没有里程计(no_odom)时，需要预测速度
- MSE损失鼓励准确的速度估计
- .detach()防止梯度回传到速度历史

### 2.4 避障损失 / Obstacle Avoidance Loss

#### 2.4.1 势能损失 / Potential Loss

$$ L_{avoid} = \frac{1}{BT} \sum_{t=1}^{T} \sum_{i=1}^{B} v_{to\_pt}(t,i) \cdot \max(0, 1 - d(t,i))^2 $$

其中 / Where:
- $d(t,i)$: 到最近障碍物的距离（减去安全距离）
- $v_{to\_pt}(t,i)$: 朝向障碍物的速度分量

**代码实现 / Code Implementation:**
```python
def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

# 计算到最近障碍物的向量 / Calculate vector to nearest obstacle
vec_to_pt_history = env.find_vec_to_nearest_pt()

# 计算距离 / Calculate distance
distance = torch.norm(vec_to_pt_history, 2, -1)
distance = distance - env.margin

# 计算接近速度 / Calculate approaching velocity
with torch.no_grad():
    v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)

# 计算避障损失 / Calculate obstacle avoidance loss
loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
```

**设计理由 / Design Rationale:**
- 使用势能函数，接近障碍物时损失指数增长
- $(1-d)^2$ 使得损失在接近障碍物时快速增加
- $v_{to\_pt}$ 确保只在朝向障碍物移动时施加惩罚
- max(0, d) 确保只有距离小于1时才有惩罚

### 2.5 碰撞损失 / Collision Loss

#### 2.5.1 Softplus损失 / Softplus Loss

$$ L_{collide} = \frac{1}{BT} \sum_{t=1}^{T} \sum_{i=1}^{B} \text{softplus}(-32 \cdot d(t,i)) \cdot v_{to\_pt}(t,i) $$

**代码实现 / Code Implementation:**
```python
loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()
```

**Softplus 函数 / Softplus Function:**

$$ \text{softplus}(x) = \ln(1 + e^x) $$

**设计理由 / Design Rationale:**
- Softplus提供平滑的指数增长
- -32系数使得损失在碰撞时非常大
- 与避障损失配合，形成双重保护
- 只对朝向障碍物移动的情况惩罚

### 2.6 加速度平滑损失 / Acceleration Smoothing Loss

#### 2.6.1 加速度惩罚 / Acceleration Penalty

$$ L_{acc} = \frac{1}{BT} \sum_{t=0}^{T-1} \sum_{i=1}^{B} ||\mathbf{a}(t,i)||^2 $$

**代码实现 / Code Implementation:**
```python
loss_d_acc = act_buffer.pow(2).sum(-1).mean()
```

**设计理由 / Design Rationale:**
- 惩罚过大的加速度，保护硬件
- 鼓励平滑的控制
- 减少能量消耗

### 2.7 加加速度平滑损失 / Jerk Smoothing Loss

#### 2.7.1 Jerk惩罚 / Jerk Penalty

$$ L_{jerk} = \frac{1}{B(T-1)} \sum_{t=0}^{T-2} \sum_{i=1}^{B} ||\mathbf{j}(t,i)||^2 $$

其中 jerk 是加速度的变化率：

Where jerk is the rate of change of acceleration:

$$ \mathbf{j}(t) = \frac{\mathbf{a}(t+1) - \mathbf{a}(t)}{dt} $$

**代码实现 / Code Implementation:**
```python
jerk_history = act_buffer.diff(1, 0).mul(15)
loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
```

**设计理由 / Design Rationale:**
- 惩罚加速度的剧烈变化
- 提高飞行舒适性
- 减少机械磨损

### 2.8 轨迹偏差损失 / Trajectory Bias Loss

#### 2.8.1 方向偏差惩罚 / Direction Bias Penalty

$$ L_{bias} = \frac{3}{BT} \sum_{t=0}^{T-1} \sum_{i=1}^{B} ||\mathbf{v}(t,i) - \hat{\mathbf{v}}_{target}(t,i)||^2 $$

其中 $\hat{\mathbf{v}}_{target}$ 是目标方向的单位向量乘以前向速度：

Where $\hat{\mathbf{v}}_{target}$ is unit vector in target direction multiplied by forward velocity:

$$ \hat{\mathbf{v}}_{target} = \frac{\mathbf{v}_{target}}{||\mathbf{v}_{target}||} \cdot (\mathbf{v} \cdot \frac{\mathbf{v}_{target}}{||\mathbf{v}_{target}||}) $$

**代码实现 / Code Implementation:**
```python
# 归一化目标速度 / Normalize target velocity
target_v_history_norm = torch.norm(target_v_history, 2, -1)
target_v_history_normalized = target_v_history / target_v_history_norm[..., None]

# 计算前向速度 / Calculate forward velocity
fwd_v = torch.sum(v_history * target_v_history_normalized, -1)

# 计算偏差损失 / Calculate bias loss
loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3
```

**设计理由 / Design Rationale:**
- 鼓励四轴飞行器朝目标方向飞行
- 惩罚侧向运动
- 系数3加强这个损失的影响

### 2.9 地面亲和损失 / Ground Affinity Loss

#### 2.9.1 高度惩罚 / Height Penalty

$$ L_{ground} = \frac{1}{BT} \sum_{t=0}^{T-1} \sum_{i=1}^{B} \max(0, p_z(t,i))^2 $$

**代码实现 / Code Implementation:**
```python
p_history = torch.stack(p_history)
loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
```

**设计理由 / Design Rationale:**
- 惩罚负高度（在地面以下）
- 鼓励四轴飞行器保持在地面以上
- 使用ReLU只对负高度惩罚

---

## 3. 训练流程 / Training Process

### 3.1 训练循环 / Training Loop

#### 3.1.1 外层循环 / Outer Loop

```python
for i in pbar:  # 迭代次数 / Number of iterations
    env.reset()
    model.reset()
    # ... 内层时间步循环
```

**步骤 / Steps:**
1. 重置环境状态
2. 重置模型隐藏状态
3. 执行多步前向模拟
4. 计算损失
5. 反向传播和优化

#### 3.1.2 内层时间步循环 / Inner Timestep Loop

```python
for t in range(args.timesteps):
    # 1. 渲染深度图像 / Render depth image
    depth, flow = env.render(ctl_dt)

    # 2. 准备状态 / Prepare state
    target_v_raw = env.p_target - env.p
    # ... 状态处理

    # 3. 前向传播 / Forward pass
    act, values, h = model(x, state, h)

    # 4. 执行物理模拟 / Execute physics simulation
    env.run(act_buffer[t], ctl_dt, target_v_raw)

    # 5. 收集历史数据 / Collect history
    # ... 保存位置、速度、动作等
```

### 3.2 数据预处理 / Data Preprocessing

#### 3.2.1 深度图像归一化 / Depth Image Normalization

```python
# 归一化深度值 / Normalize depth values
x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02

# 最大池化 / Max pooling
x = F.max_pool2d(x[:, None], 4, 4)
```

**归一化公式 / Normalization Formula:**

$$ x_{norm} = \frac{3}{\text{clamp}(d, 0.3, 24)} - 0.6 + \epsilon $$

其中 $\epsilon \sim \mathcal{N}(0, 0.02^2)$ 是噪声。

Where $\epsilon \sim \mathcal{N}(0, 0.02^2)$ is noise.

**设计理由 / Design Rationale:**
- 倒数映射使得近距离深度有更大的梯度
- 限制范围[0.3, 24]防止数值不稳定
- 添加噪声增强鲁棒性
- 池化降低分辨率，减少计算量

#### 3.2.2 状态向量构建 / State Vector Construction

```python
# 计算目标速度 / Calculate target velocity
target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
target_v_unit = target_v_raw / target_v_norm
target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)

# 构建状态向量 / Build state vector
state = [
    torch.squeeze(target_v[:, None] @ R, 1),  # 本地目标速度
    env.R[:, 2],  # 上向量
    env.margin[:, None]  # 安全距离
]
local_v = torch.squeeze(env.v[:, None] @ R, 1)
if not args.no_odom:
    state.insert(0, local_v)  # 本地速度
state = torch.cat(state, -1)
```

**状态向量维度 / State Vector Dimensions:**
- 有里程计: [10] = [local_v(3) + target_v(3) + up_vec(3) + margin(1)]
- 无里程计: [7] = [target_v(3) + up_vec(3) + margin(1)]

### 3.3 动作处理 / Action Processing

#### 3.3.1 动作解码 / Action Decoding

```python
# 从本地坐标系转换到世界坐标系 / Transform from local to world frame
a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)

# 应用推力估计误差 / Apply thrust estimation error
act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std

# 添加到动作缓冲区 / Add to action buffer
act_buffer.append(act)
```

**设计理由 / Design Rationale:**
- 分离加速度和速度预测
- 模拟推力估计误差，增强鲁棒性
- 使用动作缓冲区实现控制延迟

#### 3.3.2 动作延迟 / Action Delay

```python
act_lag = 1
act_buffer = [env.act] * (act_lag + 1)

# 在循环中 / In loop
env.run(act_buffer[t], ctl_dt, target_v_raw)
act_buffer.append(act)
```

**设计理由 / Design Rationale:**
- 模拟真实的控制延迟
- 提高训练策略的鲁棒性

### 3.4 梯度反向传播 / Gradient Backpropagation

#### 3.4.1 反向传播步骤 / Backpropagation Steps

```python
# 清零梯度 / Zero gradients
optim.zero_grad()

# 反向传播 / Backpropagate
loss.backward()

# 更新参数 / Update parameters
optim.step()

# 更新学习率 / Update learning rate
sched.step()
```

#### 3.4.2 梯度裁剪 / Gradient Clipping

虽然代码中没有显式的梯度裁剪，但使用梯度衰减机制：

Although there is no explicit gradient clipping, gradient decay mechanism is used:

```python
self.grad_decay = grad_decay  # 通常设置为 0.4

# 在RunFunction中 / In RunFunction
d_v[i][j] = d_v_next[i][j] * pow(grad_decay, ctl_dt)
d_p[i][j] = d_p_next[i][j] * pow(grad_decay, ctl_dt)
```

---

## 4. 优化策略 / Optimization Strategy

### 4.1 优化器选择 / Optimizer Selection

#### 4.1.1 AdamW优化器 / AdamW Optimizer

```python
optim = AdamW(model.parameters(), args.lr)
```

**AdamW 参数 / AdamW Parameters:**
- 学习率: 1e-3 (默认)
- Beta1: 0.9 (默认)
- Beta2: 0.999 (默认)
- Weight decay: 0.01 (默认)

**设计理由 / Design Rationale:**
- AdamW对超参数不敏感
- 权重衰减有助于防止过拟合
- 适合非凸优化问题

### 4.2 学习率调度 / Learning Rate Scheduling

#### 4.2.1 余弦退火 / Cosine Annealing

```python
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)
```

**余弦退火公式 / Cosine Annealing Formula:**

$$ \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi)) $$

其中 / Where:
- $\eta_{max} = 1e-3$ (初始学习率)
- $\eta_{min} = 1e-5$ (最小学习率)
- $T_{max} = 50000$ (总迭代次数)

**设计理由 / Design Rationale:**
- 平滑降低学习率
- 有助于收敛到更好的局部最优
- 最后100轮保持最小学习率进行精调

### 4.3 批处理策略 / Batching Strategy

#### 4.3.1 批大小选择 / Batch Size Selection

```python
--batch_size 64  # 单智能体 / Single agent
--batch_size 256  # 多智能体 / Multi-agent
```

**设计理由 / Design Rationale:**
- 较大批次提供稳定的梯度估计
- GPU内存限制决定了最大批次大小
- 多智能体训练需要更大批次（256）

#### 4.3.2 时间步数选择 / Timestep Selection

```python
--timesteps 150  # 单智能体 / Single agent
--timesteps 180  # 多智能体 / Multi-agent
```

**设计理由 / Design Rationale:**
- 足够长的轨迹学习长期策略
- 多智能体需要更长时间协调
- 平衡计算成本和训练效果

---

## 5. 监控和可视化 / Monitoring and Visualization

### 5.1 TensorBoard日志 / TensorBoard Logging

#### 5.1.1 记录的指标 / Recorded Metrics

```python
# 每25步记录一次 / Log every 25 steps
if (i + 1) % 25 == 0:
    for k, v in scaler_q.items():
        writer.add_scalar(k, sum(v) / len(v), i + 1)
    scaler_q.clear()
```

**记录的指标 / Recorded Metrics:**
- `loss`: 总损失
- `loss_v`: 速度跟踪损失
- `loss_v_pred`: 速度预测损失
- `loss_obj_avoidance`: 避障损失
- `loss_d_acc`: 加速度平滑损失
- `loss_d_jerk`: Jerk平滑损失
- `loss_bias`: 轨迹偏差损失
- `loss_speed`: 速度损失
- `loss_collide`: 碰撞损失
- `loss_ground_affinity`: 地面亲和损失
- `success`: 成功率
- `max_speed`: 最大速度
- `avg_speed`: 平均速度
- `ar`: 平均奖励 (success × avg_speed)

### 5.2 可视化 / Visualization

#### 5.2.1 保存可视化 / Save Visualization

```python
def is_save_iter(i):
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

if is_save_iter(i):
    # 保存位置历史 / Save position history
    fig_p, ax = plt.subplots()
    ax.plot(p_history[:, 0], label='x')
    ax.plot(p_history[:, 1], label='y')
    ax.plot(p_history[:, 2], label='z')
    writer.add_figure('p_history', fig_p, i + 1)

    # 保存速度历史 / Save velocity history
    fig_v, ax = plt.subplots()
    ax.plot(v_history[:, 0], label='x')
    ax.plot(v_history[:, 1], label='y')
    ax.plot(v_history[:, 2], label='z')
    writer.add_figure('v_history', fig_v, i + 1)

    # 保存动作历史 / Save action history
    fig_a, ax = plt.subplots()
    ax.plot(act_history[:, 0], label='x')
    ax.plot(act_history[:, 1], label='y')
    ax.plot(act_history[:, 2], label='z')
    writer.add_figure('a_reals', fig_a, i + 1)
```

### 5.3 模型保存 / Model Saving

#### 5.3.1 定期保存 / Regular Saving

```python
# 每10000次迭代保存一次 / Save every 10000 iterations
if (i + 1) % 10000 == 0:
    torch.save(model.state_dict(), f'checkpoint{i//10000:04d}.pth')
```

**设计理由 / Design Rationale:**
- 定期保存防止训练中断
- 便于回退到之前的检查点
- 节省磁盘空间（不每次都保存）

---

## 6. 训练技巧 / Training Tips

### 6.1 课程学习 / Curriculum Learning

#### 6.1.1 逐步增加难度 / Gradually Increase Difficulty

1. **初始阶段**: 简单环境，低速度
2. **中间阶段**: 添加障碍物，增加速度
3. **高级阶段**: 复杂环境，高速度

**实现方式 / Implementation:**
```python
# 随机化速度范围 / Randomize speed range
self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp

# 随机化障碍物密度 / Randomize obstacle density
self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b
```

### 6.2 域随机化 / Domain Randomization

#### 6.2.1 随机化物理参数 / Randomize Physics Parameters

```python
# 阻力系数 / Drag coefficient
self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3

# 控制延迟 / Control delay
self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)

# 推力估计误差 / Thrust estimation error
self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01
```

#### 6.2.2 随机化环境参数 / Randomize Environment Parameters

```python
# 随机化FOV / Randomize FOV
self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan

# 随机化相机角度 / Randomize camera angle
cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
```

### 6.3 数据增强 / Data Augmentation

#### 6.3.1 深度图像噪声 / Depth Image Noise

```python
x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
```

#### 6.3.2 状态扰动 / State Perturbation

```python
self.v = torch.randn((B, 3), device=device) * 0.2
self.a = torch.randn_like(self.v) * 0.1
```

---

## 7. 超参数调优 / Hyperparameter Tuning

### 7.1 关键超参数 / Key Hyperparameters

#### 7.1.1 损失权重 / Loss Weights

| 参数 | 默认值 | 作用 | 调优建议 |
|------|--------|------|----------|
| `coef_v` | 1.0 | 速度跟踪 | 根据任务调整 |
| `coef_v_pred` | 2.0 | 速度预测 | 无里程计时调大 |
| `coef_obj_avoidance` | 1.5 | 避障 | 障碍物多时调大 |
| `coef_collide` | 2.0 | 碰撞 | 安全要求高时调大 |
| `coef_d_acc` | 0.01 | 加速度平滑 | 硬件限制严格时调大 |
| `coef_d_jerk` | 0.001 | Jerk平滑 | 舒适性要求高时调大 |

#### 7.1.2 训练参数 / Training Parameters

| 参数 | 默认值 | 作用 | 调优建议 |
|------|--------|------|----------|
| `lr` | 1e-3 | 学习率 | 根据收敛情况调整 |
| `batch_size` | 64/256 | 批大小 | 根据GPU内存调整 |
| `num_iters` | 50000 | 迭代次数 | 根据任务复杂度调整 |
| `timesteps` | 150/180 | 时间步数 | 任务越长调大 |
| `grad_decay` | 0.4 | 梯度衰减 | 通常不需要调整 |

### 7.2 调优策略 / Tuning Strategy

#### 7.2.1 网格搜索 / Grid Search

```python
# 示例 / Example
for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
    for coef_obj in [1.0, 1.5, 2.0]:
        # 运行训练 / Run training
        train(lr=lr, coef_obj=coef_obj)
```

#### 7.2.2 手动调优 / Manual Tuning

1. 先调整学习率使损失下降
2. 再调整损失权重平衡各项
3. 最后微调其他参数

---

## 8. 总结 / Summary

DiffDrone训练算法的核心特点：

Key characteristics of DiffDrone training algorithm:

1. **端到端可微分训练**: 从视觉输入到控制输出端到端优化
2. **多损失函数设计**: 平衡速度跟踪、避障、平滑等多个目标
3. **序列建模**: 使用GRU处理时间序列数据
4. **课程学习和域随机化**: 逐步增加难度，提高泛化能力
5. **完善监控**: TensorBoard可视化训练过程

通过这些精心设计的算法和策略，DiffDrone实现了高效的视觉引导敏捷飞行学习。

Through these carefully designed algorithms and strategies, DiffDrone achieves efficient vision-guided agile flight learning.
