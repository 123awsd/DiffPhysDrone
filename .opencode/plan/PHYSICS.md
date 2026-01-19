# DiffDrone 物理模型文档 / DiffDrone Physics Model Document

## 1. 四轴飞行器动力学 / Quadrotor Dynamics

### 1.1 运动学方程 / Kinematic Equations

#### 1.1.1 坐标系定义 / Coordinate System Definitions

**世界坐标系 (World Frame)**: 惯性坐标系，固定不变
**机体系 (Body Frame)**: 附着在四轴飞行器上的坐标系
**相机坐标系 (Camera Frame)**: 相机的局部坐标系

```
世界坐标系 / World Frame (W):
  X_W: 向东 (East)
  Y_W: 向北 (North)
  Z_W: 向上 (Up)

机体系 / Body Frame (B):
  X_B: 向前 (Forward)
  Y_B: 向左 (Left)
  Z_B: 向上 (Up)

相机坐标系 / Camera Frame (C):
  X_C: 向前 (Forward)
  Y_C: 向右 (Right)
  Z_C: 向下 (Down)
```

#### 1.1.2 姿态表示 / Attitude Representation

DiffDrone使用**旋转矩阵 (Rotation Matrix)** $R \in SO(3)$ 表示姿态：

DiffDrone uses **Rotation Matrix** $R \in SO(3)$ to represent attitude:

$$ R = \begin{bmatrix} R_{11} & R_{12} & R_{13} \\ R_{21} & R_{22} & R_{23} \\ R_{31} & R_{32} & R_{33} \end{bmatrix} $$

其中列向量分别表示：
- 第1列: 前向量 (forward vector) $\mathbf{f} = [R_{11}, R_{21}, R_{31}]^T$
- 第2列: 左向量 (left vector) $\mathbf{l} = [R_{12}, R_{22}, R_{32}]^T$
- 第3列: 上向量 (up vector) $\mathbf{u} = [R_{13}, R_{23}, R_{33}]^T$

Where the column vectors represent:
- Column 1: Forward vector $\mathbf{f}$
- Column 2: Left vector $\mathbf{l}$
- Column 3: Up vector $\mathbf{u}$

### 1.2 动力学方程 / Dynamic Equations

#### 1.2.1 位置和速度更新 / Position and Velocity Update

使用**半隐式欧拉法 (Semi-Implicit Euler)** 进行数值积分：

Using **Semi-Implicit Euler** for numerical integration:

$$ \mathbf{p}_{next} = \mathbf{p} \cdot \alpha^{dt} + \mathbf{v} \cdot dt + \frac{1}{2}\mathbf{a} \cdot dt^2 $$

$$ \mathbf{v}_{next} = \mathbf{v} \cdot \alpha^{dt} + \frac{\mathbf{a} + \mathbf{a}_{next}}{2} \cdot dt $$

**代码实现 / Code Implementation:**
```cuda
// 位置更新 / Position update
for (int j=0; j<3; j++)
    p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;

// 速度更新 / Velocity update
for (int j=0; j<3; j++)
    v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
```

#### 1.2.2 加速度计算 / Acceleration Calculation

总加速度由多个力组成：

Total acceleration is composed of multiple forces:

$$ \mathbf{a}_{next} = \mathbf{a}_{filtered} + \mathbf{d}_g - \mathbf{F}_{drag} + \mathbf{F}_{airmode} $$

**组成部分 / Components:**

1. **控制加速度 (Control Acceleration)** $\mathbf{a}_{filtered}$
   - 经过低通滤波的期望加速度
   - Low-pass filtered desired acceleration

2. **扰动加速度 (Disturbance Acceleration)** $\mathbf{d}_g$
   - 模拟风力和扰动
   - Simulates wind and disturbances

3. **空气阻力 (Air Drag)** $\mathbf{F}_{drag}$
   - 速度的二次函数
   - Quadratic function of velocity

4. **角速度加速度 (Angular Velocity Acceleration)** $\mathbf{F}_{airmode}$
   - 模拟角速度变化的影响
   - Simulates the effect of angular velocity changes

**代码实现 / Code Implementation:**
```cuda
// 计算相对风的速度 / Calculate velocity relative to wind
scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];

// 投影到机体坐标系 / Project to body frame
scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];
scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];
scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];

// 计算速度的绝对值 / Calculate absolute values
scalar_t v_up_2 = v_up_s * abs(v_up_s);
scalar_t v_fwd_2 = v_fwd_s * abs(v_fwd_s);
scalar_t v_left_2 = v_left_s * abs(v_left_s);

// 计算二次和线性阻力 / Calculate quadratic and linear drag
scalar_t a_drag_2[3], a_drag_1[3];
for (int j=0; j<3; j++){
    a_drag_2[j] = v_up_2 * R[i][j][2] * z_drag_coef[i][0]
                 + v_left_2 * R[i][j][1]
                 + v_fwd_2 * R[i][j][0];
    a_drag_1[j] = v_up_s * R[i][j][2] * z_drag_coef[i][0]
                 + v_left_s * R[i][j][1]
                 + v_fwd_s * R[i][j][0];
}

// 计算角速度加速度 / Calculate angular velocity acceleration
scalar_t dot = act[i][0] * act_next[i][0] + act[i][1] * act_next[i][1]
               + (act[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
scalar_t n1 = act[i][0] * act[i][0] + act[i][1] * act[i][1] + (act[i][2] + 9.80665) * (act[i][2] + 9.80665);
scalar_t n2 = act_next[i][0] * act_next[i][0] + act_next[i][1] * act_next[i][1] + (act_next[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
scalar_t av = acos(max(-1., min(1., dot / max(1e-8, sqrt(n1) * sqrt(n2))))) / ctl_dt;

scalar_t ax = act[i][0];
scalar_t ay = act[i][1];
scalar_t az = act[i][2] + 9.80665;
scalar_t thrust = sqrt(ax*ax+ay*ay+az*az);
scalar_t airmode_a[3] = {
    ax / thrust * av * airmode_av2a,
    ay / thrust * av * airmode_av2a,
    az / thrust * av * airmode_av2a
};

// 计算总加速度 / Calculate total acceleration
for (int j=0; j<3; j++)
    a_next[i][j] = act_next[i][j] + dg[i][j]
                   - a_drag_2[j] * drag_2[i][0]
                   - a_drag_1[j] * drag_2[i][1]
                   + airmode_a[j];
```

### 1.3 姿态更新 / Attitude Update

#### 1.3.1 姿态控制原理 / Attitude Control Principle

DiffDrone使用**期望推力向量 (Desired Thrust Vector)** 来更新姿态：

DiffDrone uses **Desired Thrust Vector** to update attitude:

1. 从期望加速度提取上向量：$\mathbf{u} = \frac{\mathbf{a}_{thrust}}{||\mathbf{a}_{thrust}||}$
2. 更新前向量：$\mathbf{f}_{new} = \alpha \mathbf{f}_{old} + (1-\alpha) \text{normalize}(\mathbf{v}_{pred} + \text{inertia} \cdot \mathbf{f}_{old})$
3. 计算左向量：$\mathbf{l} = \mathbf{u} \times \mathbf{f}_{new}$
4. 构建新旋转矩阵：$R_{new} = [\mathbf{f}_{new}, \mathbf{l}, \mathbf{u}]$

**代码实现 / Code Implementation:**
```cuda
// 提取上向量 / Extract up vector from thrust
scalar_t ax = a_thr[b][0];
scalar_t ay = a_thr[b][1];
scalar_t az = a_thr[b][2] + 9.80665;  // 加上重力 / Add gravity
scalar_t thrust = sqrt(ax*ax+ay*ay+az*az);

scalar_t ux = ax / thrust;  // 上向量x / Up vector x
scalar_t uy = ay / thrust;  // 上向量y / Up vector y
scalar_t uz = az / thrust;  // 上向量z / Up vector z

// 更新前向量 / Update forward vector
scalar_t fx = R[b][0][0] * yaw_inertia + v_pred[b][0];
scalar_t fy = R[b][1][0] * yaw_inertia + v_pred[b][1];
scalar_t fz = R[b][2][0] * yaw_inertia + v_pred[b][2];

// 归一化并与旧向量混合 / Normalize and blend with old vector
scalar_t t = sqrt(fx * fx + fy * fy + fz * fz);
fx = (1 - alpha[b][0]) * (fx / t) + alpha[b][0] * R[b][0][0];
fy = (1 - alpha[b][0]) * (fy / t) + alpha[b][0] * R[b][1][0];
fz = (1 - alpha[b][0]) * (fz / t) + alpha[b][0] * R[b][2][0];

// 确保前向量与上向量正交 / Ensure forward vector is orthogonal to up vector
fz = (fx * ux + fy * uy) / -uz;

// 重新归一化 / Renormalize
t = sqrt(fx * fx + fy * fy + fz * fz);
fx /= t;
fy /= t;
fz /= t;

// 计算左向量 / Calculate left vector (cross product)
scalar_t lx = uy * fz - uz * fy;
scalar_t ly = uz * fx - ux * fz;
scalar_t lz = ux * fy - uy * fx;

// 构建新旋转矩阵 / Build new rotation matrix
R_new[b][0][0] = fx;
R_new[b][0][1] = lx;
R_new[b][0][2] = ux;
R_new[b][1][0] = fy;
R_new[b][1][1] = ly;
R_new[b][1][2] = uy;
R_new[b][2][0] = fz;
R_new[b][2][1] = lz;
R_new[b][2][2] = uz;
```

#### 1.3.2 控制延迟建模 / Control Delay Modeling

DiffDrone使用**一阶低通滤波器**模拟控制延迟：

DiffDrone uses **First-Order Low-Pass Filter** to simulate control delay:

$$ \alpha = \exp(-\tau \cdot dt) $$

$$ \mathbf{a}_{next} = \mathbf{a}_{pred} \cdot (1-\alpha) + \mathbf{a}_{old} \cdot \alpha $$

其中 $\tau$ 是时间常数，$\alpha$ 是滤波系数。

Where $\tau$ is time constant, $\alpha$ is filter coefficient.

**代码实现 / Code Implementation:**
```cuda
// 计算滤波系数 / Calculate filter coefficient
scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);

// 应用低通滤波 / Apply low-pass filter
for (int j=0; j<3; j++)
    act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
```

---

## 2. 空气阻力模型 / Air Drag Model

### 2.1 阻力方程 / Drag Equation

DiffDrone使用二次阻力模型：

DiffDrone uses quadratic drag model:

$$ \mathbf{F}_{drag} = C_d \cdot ||\mathbf{v}_{rel}|| \cdot \mathbf{v}_{rel} + C_{d1} \cdot \mathbf{v}_{rel} $$

其中 $\mathbf{v}_{rel} = \mathbf{v} - \mathbf{v}_{wind}$ 是相对于风的速度。

Where $\mathbf{v}_{rel} = \mathbf{v} - \mathbf{v}_{wind}$ is velocity relative to wind.

### 2.2 方向相关阻力 / Direction-Dependent Drag

在机体坐标系中，不同方向的阻力系数不同：

In the body frame, drag coefficients differ in different directions:

- $z\_drag\_coef$: 垂直方向的阻力系数（受下洗气流影响）
- $drag\_2[0]$: 二次阻力系数
- $drag\_2[1]$: 线性阻力系数

**代码实现 / Code Implementation:**
```python
# 随机化阻力系数 / Randomize drag coefficients
self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
self.drag_2[:, 0] = 0  # 设置为0，使用二次阻力 / Set to 0, use quadratic drag
self.z_drag_coef = torch.ones((B, 1), device=device)  # 垂直阻力系数 / Vertical drag coefficient
```

**在CUDA中计算 / Calculate in CUDA:**
```cuda
// 计算机体坐标系下的速度分量 / Calculate velocity components in body frame
scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];
scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];
scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];

// 计算阻力加速度 / Calculate drag acceleration
for (int j=0; j<3; j++){
    a_drag_2[j] = v_up_s * v_up_s * R[i][j][2] * z_drag_coef[i][0]
                 + v_left_s * v_left_s * R[i][j][1]
                 + v_fwd_s * v_fwd_s * R[i][j][0];
    a_drag_1[j] = v_up_s * R[i][j][2] * z_drag_coef[i][0]
                 + v_left_s * R[i][j][1]
                 + v_fwd_s * R[i][j][0];
}
```

---

## 3. 碰撞检测 / Collision Detection

### 3.1 几何体相交测试 / Geometry Intersection Tests

#### 3.1.1 射线-球体相交 / Ray-Sphere Intersection

给定射线 $\mathbf{r}(t) = \mathbf{o} + t \cdot \mathbf{d}$ 和球体 $(\mathbf{c}, r)$，相交条件为：

Given ray $\mathbf{r}(t) = \mathbf{o} + t \cdot \mathbf{d}$ and sphere $(\mathbf{c}, r)$, intersection condition:

$$ ||\mathbf{o} + t\mathbf{d} - \mathbf{c}||^2 = r^2 $$

展开为二次方程：
Expand to quadratic equation:

$$ at^2 + bt + c = 0 $$

其中 / Where:
- $a = \mathbf{d} \cdot \mathbf{d}$
- $b = 2\mathbf{d} \cdot (\mathbf{o} - \mathbf{c})$
- $c = (\mathbf{o} - \mathbf{c}) \cdot (\mathbf{o} - \mathbf{c}) - r^2$

判别式 / Discriminant:

$$ \Delta = b^2 - 4ac $$

如果 $\Delta \geq 0$，则相交，最近的交点为：

If $\Delta \geq 0$, intersection exists, closest intersection is:

$$ t = \frac{-b - \sqrt{\Delta}}{2a} \quad (\text{取负号取最近的点}) $$

**代码实现 / Code Implementation:**
```cuda
scalar_t a = dx * dx + dy * dy + dz * dz;
scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
scalar_t d = b * b - 4 * a * c;
if (d >= 0) {
    r = (-b - sqrt(d)) / (2 * a);
    if (r > 1e-5) {
        min_dist = min(min_dist, r);
    } else {
        r = (-b + sqrt(d)) / (2 * a);
        if (r > 1e-5) min_dist = min(min_dist, r);
    }
}
```

#### 3.1.2 射线-圆柱体相交 / Ray-Cylinder Intersection

垂直圆柱体：只考虑X-Y平面的相交

Vertical cylinder: only consider intersection in X-Y plane

$$ (\mathbf{o}_x + t\mathbf{d}_x - c_x)^2 + (\mathbf{o}_y + t\mathbf{d}_y - c_y)^2 = r^2 $$

水平圆柱体：只考虑X-Z平面的相交

Horizontal cylinder: only consider intersection in X-Z plane

$$ (\mathbf{o}_x + t\mathbf{d}_x - c_x)^2 + (\mathbf{o}_z + t\mathbf{d}_z - c_z)^2 = r^2 $$

**代码实现 / Code Implementation:**
```cuda
// 垂直圆柱体 / Vertical cylinder
scalar_t a = dx * dx + dy * dy;
scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
scalar_t d = b * b - 4 * a * c;
if (d >= 0) {
    r = (-b - sqrt(d)) / (2 * a);
    if (r > 1e-5) {
        min_dist = min(min_dist, r);
    }
}

// 水平圆柱体 / Horizontal cylinder
scalar_t a = dx * dx + dz * dz;
scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
// ... 相同的求解过程 / Same solving process
```

#### 3.1.3 射线-长方体相交 / Ray-Box Intersection

使用** slab method** 检测射线与轴对齐长方体的相交：

Use **slab method** to detect intersection between ray and axis-aligned box:

对于每个轴，计算进入和离开的t值：

For each axis, calculate entry and exit t values:

$$ t_{x1} = \frac{c_x - r_x - o_x}{d_x}, \quad t_{x2} = \frac{c_x + r_x - o_x}{d_x} $$
$$ t_{y1} = \frac{c_y - r_y - o_y}{d_y}, \quad t_{y2} = \frac{c_y + r_y - o_y}{d_y} $$
$$ t_{z1} = \frac{c_z - r_z - o_z}{d_z}, \quad t_{z2} = \frac{c_z + r_z - o_z}{d_z} $$

最近的进入点 / Nearest entry:

$$ t_{in} = \max(\min(t_{x1}, t_{x2}), \min(t_{y1}, t_{y2}), \min(t_{z1}, t_{z2})) $$

最远的离开点 / Furthest exit:

$$ t_{out} = \min(\max(t_{x1}, t_{x2}), \max(t_{y1}, t_{y2}), \max(t_{z1}, t_{z2})) $$

相交条件 / Intersection condition:

$$ t_{in} < t_{out} \quad \text{且} \quad t_{in} > 0 $$

**代码实现 / Code Implementation:**
```cuda
scalar_t tx1 = (cx - rx - ox) / dx;
scalar_t tx2 = (cx + rx - ox) / dx;
scalar_t tx_min = min(tx1, tx2);
scalar_t tx_max = max(tx1, tx2);

scalar_t ty1 = (cy - ry - oy) / dy;
scalar_t ty2 = (cy + ry - oy) / dy;
scalar_t ty_min = min(ty1, ty2);
scalar_t ty_max = max(ty1, ty2);

scalar_t tz1 = (cz - rz - oz) / dz;
scalar_t tz2 = (cz + rz - oz) / dz;
scalar_t tz_min = min(tz1, tz2);
scalar_t tz_max = max(tz1, tz2);

scalar_t t_min = max(max(tx_min, ty_min), tz_min);
scalar_t t_max = min(min(tx_max, ty_max), tz_max);

if (t_min < min_dist && t_min < t_max && t_min > 0)
    min_dist = t_min;
```

### 3.2 最近点计算 / Nearest Point Calculation

对于避障，需要计算从点到障碍物表面的最近点：

For obstacle avoidance, need to calculate the nearest point from point to obstacle surface:

#### 3.2.1 点到球体最近点 / Point to Sphere Nearest Point

$$ \mathbf{p}_{nearest} = \mathbf{c} + \frac{\mathbf{p} - \mathbf{c}}{||\mathbf{p} - \mathbf{c}||} \cdot r $$

距离 / Distance:

$$ d = ||\mathbf{p} - \mathbf{c}|| - r $$

#### 3.2.2 点到长方体最近点 / Point to Box Nearest Point

使用**clamping** 计算最近点：

Use **clamping** to calculate nearest point:

$$ \mathbf{p}_{nearest} = \mathbf{c} + \text{clamp}(\mathbf{p} - \mathbf{c}, -\mathbf{r}, \mathbf{r}) $$

其中 clamp 函数将向量限制在 $[-\mathbf{r}, \mathbf{r}]$ 范围内。

Where clamp function clamps vector to $[-\mathbf{r}, \mathbf{r}]$ range.

**代码实现 / Code Implementation:**
```cuda
// 计算点到长方体表面的最近点 / Calculate nearest point from point to box surface
scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3;
scalar_t rx = min(max_r, voxels[batch_base][i][3]);
scalar_t ry = min(max_r, voxels[batch_base][i][4]);
scalar_t rz = min(max_r, voxels[batch_base][i][5]);

scalar_t ptx = cx + max(-rx, min(rx, ox - cx));
scalar_t pty = cy + max(-ry, min(ry, oy - cy));
scalar_t ptz = cz + max(-rz, min(rz, oz - cz));

scalar_t dist = (ptx - ox) * (ptx - ox) + (pty - oy) * (pty - oy) + (ptz - oz) * (ptz - oz);
dist = sqrt(dist);
```

---

## 4. 避障策略 / Obstacle Avoidance Strategy

### 4.1 避障损失函数 / Obstacle Avoidance Loss Function

#### 4.1.1 势能函数 / Potential Function

使用**势能函数**鼓励四轴飞行器远离障碍物：

Use **potential function** to encourage quadrotor to stay away from obstacles:

$$ L_{avoid} = \sum_{t=1}^{T} \sum_{i=1}^{N} v_{to\_pt}(t,i) \cdot \max(0, 1 - d(t,i))^2 $$

其中 / Where:
- $d(t,i)$: 时间步 $t$ 时四轴飞行器 $i$ 到最近障碍物的距离
- $v_{to\_pt}(t,i)$: 四轴飞行器朝向障碍物的速度分量

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

#### 4.1.2 碰撞损失 / Collision Loss

当距离小于安全距离时，施加指数增长的碰撞损失：

When distance is below safety margin, apply exponentially growing collision loss:

$$ L_{collide} = \frac{1}{B \cdot T} \sum_{t=1}^{T} \sum_{i=1}^{B} \text{softplus}(-32 \cdot d(t,i)) \cdot v_{to\_pt}(t,i) $$

**代码实现 / Code Implementation:**
```python
loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()
```

### 4.2 安全距离 / Safety Margin

每个四轴飞行器都有一个安全距离裕度：

Each quadrotor has a safety margin:

$$ d_{safe} = d_{measured} - \text{margin} $$

其中 margin 是随机化的，在 $[0.1, 0.3]$ 范围内。

Where margin is randomized in range $[0.1, 0.3]$.

**代码实现 / Code Implementation:**
```python
self.margin = torch.rand((B,), device=device) * 0.2 + 0.1
```

---

## 5. 渲染算法 / Rendering Algorithm

### 5.1 深度图像生成 / Depth Image Generation

#### 5.1.1 光线发射 / Ray Emission

对于每个像素 $(u, v)$，计算光线方向：

For each pixel $(u, v)$, calculate ray direction:

$$ \mathbf{d} = \mathbf{R} \cdot \mathbf{f}(u,v) $$

其中 $\mathbf{f}(u,v)$ 是相机坐标系下的方向向量：

Where $\mathbf{f}(u,v)$ is direction vector in camera frame:

$$ f_x = R_{0,0} - f_u \cdot R_{0,2} - f_v \cdot R_{0,1} $$
$$ f_y = R_{1,0} - f_u \cdot R_{1,2} - f_v \cdot R_{1,1} $$
$$ f_z = R_{2,0} - f_u \cdot R_{2,2} - f_v \cdot R_{2,1} $$

归一化坐标 / Normalized coordinates:

$$ f_u = (2(u+0.5)/H - 1) \cdot \tan(\theta_{y}/2) $$
$$ f_v = (2(v+0.5)/W - 1) \cdot \tan(\theta_{x}/2) $$

**代码实现 / Code Implementation:**
```cuda
const scalar_t fov_y_half_tan = fov_x_half_tan / W * H;
const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_half_tan - 1e-5;
const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov_x_half_tan - 1e-5;

scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];
```

#### 5.1.2 深度值计算 / Depth Value Calculation

对于每条光线，计算与所有障碍物的最近交点：

For each ray, calculate closest intersection with all obstacles:

$$ \text{depth}(u,v) = \min_{i \in \text{obstacles}} t_i $$

其中 $t_i$ 是光线与障碍物 $i$ 的交点距离。

Where $t_i$ is intersection distance between ray and obstacle $i$.

**代码实现 / Code Implementation:**
```cuda
scalar_t min_dist = 100;

// 检查地面 / Check ground
scalar_t t = (-1 - oz) / dz;
if (t > 0) min_dist = t;

// 检查其他智能体 / Check other agents
for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
    if (i == b || i >= B) continue;
    // 射线-椭球体相交测试 / Ray-ellipsoid intersection test
    // ...
}

// 检查球体 / Check spheres
for (int i = 0; i < balls.size(1); i++) {
    // 射线-球体相交测试 / Ray-sphere intersection test
    // ...
}

// 检查圆柱体 / Check cylinders
// ...

// 检查长方体 / Check boxes
// ...

canvas[b][u][v] = min_dist;  // 保存深度值 / Save depth value
```

### 5.2 渲染优化 / Rendering Optimization

#### 5.2.1 早期终止 / Early Termination

一旦找到小于当前最小距离的交点，立即跳过后续障碍物检查：

Once intersection smaller than current minimum is found, skip remaining obstacle checks:

```cuda
if (r > 1e-5) {
    min_dist = min(min_dist, r);
} else {
    // ... 继续检查 / continue checking
}
```

#### 5.2.2 并行计算 / Parallel Computation

每个像素由独立线程处理，充分利用GPU并行能力：

Each pixel processed by independent thread, fully utilizing GPU parallelism:

```cuda
const int idx = blockIdx.x * blockDim.x + threadIdx.x;
const int B = canvas.size(0);
const int H = canvas.size(1);
const int W = canvas.size(2);

if (idx >= B * H * W) return;

const int b = idx / (H * W);  // 批索引 / Batch index
const int u = (idx % (H * W)) / W;  // 高度索引 / Height index
const int v = idx % W;  // 宽度索引 / Width index
```

---

## 6. 总结 / Summary

DiffDrone的物理模型核心特点：

Key characteristics of DiffDrone physics model:

1. **真实物理建模**: 包含空气阻力、控制延迟、风力扰动等真实物理效果
2. **可微分设计**: 所有物理计算支持梯度反向传播
3. **高效碰撞检测**: 使用光线追踪和几何相交测试实现快速碰撞检测
4. **灵活的障碍物**: 支持球体、圆柱体、长方体等多种几何体
5. **多智能体支持**: 支持集群控制和动态碰撞检测

通过这些精心设计的物理模型，DiffDrone实现了高效、准确的可微分物理模拟。

Through these carefully designed physics models, DiffDrone achieves efficient and accurate differentiable physics simulation.
