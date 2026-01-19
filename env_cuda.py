# DiffDrone Differentiable Environment Simulator
# DiffDrone 可微分环境模拟器
#
# This module implements a differentiable quadrotor physics simulator.
# It supports:
# - Differentiable physics (gradients flow through entire simulation)
# - Depth image rendering using CUDA ray casting
# - Multiple obstacle types (spheres, cylinders, boxes)
# - Multi-agent simulation with collision detection
# - Realistic physical effects (drag, delay, wind)
#
# 该模块实现了可微分的四轴飞行器物理模拟器。
# 它支持：
# - 可微分物理（梯度通过整个模拟传播）
# - 使用CUDA光线追踪的深度图像渲染
# - 多种障碍物类型（球体、圆柱体、长方体）
# - 多智能体模拟，带碰撞检测
# - 真实的物理效果（阻力、延迟、风力）

import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda

# Gradient Decay Autograd Function / 梯度衰减自动微分函数
# Implements custom autograd for gradient decay through time
# 实现时间梯度衰减的自定义自动微分
class GDecay(torch.autograd.Function):
    # Custom autograd function for gradient decay
    # 用于梯度衰减的自定义自动微分函数
    @staticmethod
    def forward(ctx, x, alpha):
        # Forward: save decay factor, return x unchanged
        # 前向传播：保存衰减因子，直接返回输入
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: multiply gradient by decay factor
        # 反向传播：将梯度乘以衰减因子
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply

# Physics Simulation Autograd Function / 物理模拟自动微分函数
# Wraps CUDA kernels for differentiable physics simulation
# 封装CUDA内核以实现可微分物理模拟
class RunFunction(torch.autograd.Function):
    # Custom autograd function for quadrotor dynamics simulation
    # 用于四轴飞行器动力学模拟的自定义自动微分函数
    @staticmethod
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        # Forward: call CUDA for dynamics simulation
        # 前向传播：调用CUDA进行动力学模拟
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        # Save intermediate variables for backward pass
        # 保存反向传播所需的中间变量
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        # Backward: compute gradients
        # 反向传播：计算梯度
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None

run = RunFunction.apply

# Quadrotor Environment / 四轴飞行器环境
class Env:
    # Quadrotor simulation environment with differentiable physics
    # 四轴飞行器模拟环境，包含可微分物理
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10) -> None:
        # Initialize environment parameters
        # 初始化环境参数
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay
        
        # Obstacle parameter ranges [x, y, z, radius/size] for balls/voxels
        # 障碍物参数范围 [x, y, z, 半径/尺寸] 用于球体/体素
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device)  # Ball max values / 球体最大值
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)  # Ball min values / 球体最小值
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device)  # Voxel max / 体素最大值
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)  # Voxel min / 体素最小值
        self.ground_voxel_w = torch.tensor([8., 18, 0, 2.9, 2.9, 1.9], device=device)  # Ground voxel max
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)  # Ground voxel min
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)  # Vertical cylinder max / 垂直圆柱最大值
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)  # Vertical cylinder min
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)  # Horizontal cylinder max
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)  # Horizontal cylinder min
        self.gate_w = torch.tensor([2., 2, 1.0, 0.5], device=device)  # Gate max / 门最大值
        self.gate_b = torch.tensor([3., -1, 0.0, 0.5], device=device)  # Gate min / 门最小值
        self.v_wind_w = torch.tensor([1, 1, 0.2], device=device)  # Wind speed max / 风速最大值
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)  # Standard gravity / 标准重力
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)  # Roof offset / 屋顶偏移
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)  # Time subdivision / 时间细分
        
        # Start and end positions for navigation task
        # 导航任务的起始位置和目标位置
        self.p_init = torch.as_tensor([
            [-1.5, -3., 1],
            [ 9.5, -3., 1],
            [-0.5,  1., 1],
            [ 8.5,  1., 1],
            [ 0.0,  3., 1],
            [ 8.0,  3., 1],
            [-1.0, -1., 1],
            [ 9.0, -1., 1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.p_end = torch.as_tensor([
            [8.,  3., 1],
            [0.,  3., 1],
            [8., -1., 1],
            [0., -1., 1],
            [8., -3., 1],
            [0., -3., 1],
            [8.,  1., 1],
            [0.,  1., 1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.flow = torch.empty((batch_size, 0, height, width), device=device)
        
        # Environment options / 环境选项
        self.single = single
        self.gate = gate
        self.ground_voxels = ground_voxels
        self.scaffold = scaffold
        self.speed_mtp = speed_mtp
        self.random_rotation = random_rotation
        self.cam_angle = cam_angle
        self.fov_x_half_tan = fov_x_half_tan
        self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device)
 
    def reset(self):
        # Reset environment to initial state
        # 重置环境到初始状态
        B = self.batch_size
        device = self.device
 
        # Camera rotation matrix
        # 相机角度旋转矩阵
        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)
 
        # Randomly generate obstacles
        # 随机生成障碍物
        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b  # Spherical obstacles / 球体障碍物
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b  # Voxel obstacles / 体素障碍物
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b  # Vertical cylinders / 垂直圆柱障碍物
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b  # Horizontal cylinders / 水平圆柱障碍物
 
        # Randomize environment parameters for domain randomization
        # 随机化环境参数以进行域随机化
        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1
 
        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
        scale = (self.max_speed - 0.5).clamp_min(1)
 
        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01  # Thrust estimation error / 推力估计误差
 
        # Generate roof obstacles
        # 生成屋顶障碍物
        roof = torch.rand((B,)) < 0.5
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2]
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2]
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4]
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl_h[..., 2])
        self.voxels[roof, 0, 2] = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200
 
        # Generate ground obstacles
        # 生成地面障碍物
        if self.ground_voxels:
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1
            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)
 
        # Scale obstacles based on max speed
        # 根据最大速度缩放障碍物
        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1] *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1] *= (self.max_speed + 4) / scale
 
        # Generate gate obstacles
        # 生成门障碍物
        if self.gate:
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b
            p = gate[None, :, :3]
            nearest_pt = torch.empty_like(p)
            quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
            ones = torch.ones_like(gate_x)
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),
            ], 1)
            self.voxels = torch.cat([self.voxels, gate], 1)
        self.voxels[..., 0] *= scale
        self.balls[..., 0] *= scale
        self.cyl[..., 0] *= scale
        self.cyl_h[..., 0] *= scale
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)
 
        # Initialize quadrotor states
        # 初始化四轴飞行器状态
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)  # Pitch control delay / 俯仰控制延迟
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)  # Yaw control delay / 偏航控制延迟
 
        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        scale = torch.cat([
            scale,
            rd + 0.5,
            torch.rand_like(scale) - 0.5], -1)
        self.p = self.p_init * scale + torch.randn_like(scale) * 0.1  # Position / 位置
        self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1  # Target position / 目标位置
 
        # Random rotation
        # 随机旋转
        if self.random_rotation:
            yaw_bias = torch.rand(B//self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            R = torch.stack([c,-s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
            self.p = torch.squeeze(R @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            self.voxels[..., :3] = (R @ self.voxels[..., :3].transpose(1, 2)).transpose(1, 2)
            self.balls[..., :3] = (R @ self.balls[..., :3].transpose(1, 2)).transpose(1, 2)
            self.cyl[..., :3] = (R @ self.cyl[..., :3].transpose(1, 2)).transpose(1, 2)
 
        # Generate scaffold obstacles
        # 生成脚手架障碍物
        if self.scaffold and random.random() < 0.5:
            x = torch.arange(1, 6, dtype=torch.float, device=device)
            y = torch.arange(-3, 4, dtype=torch.float, device=device)
            z = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed),
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed) * 0.1,
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)
 
        # Initialize velocity, acceleration, etc.
        # 初始化速度、加速度等状态
        self.v = torch.randn((B, 3), device=device) * 0.2  # Velocity / 速度
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w  # Wind velocity / 风速
        self.act = torch.randn_like(self.v) * 0.1  # Action / 动作
        self.a = self.act  # Acceleration / 加速度
        self.dg = torch.randn((B, 3), device=device) * 0.2  # Gravity disturbance / 重力扰动
 
        # Initialize rotation matrix (attitude)
        # 初始化旋转矩阵（姿态）
        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1  # Safety margin / 安全边距
 
        # Drag coefficients
        # 阻力系数
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)
 
    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        # Update rotation matrix (attitude)
        # 更新旋转矩阵（姿态）
        self_forward_vec = R[..., 0]
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr = a_thr - g_std
        thrust = torch.norm(a_thr, 2, -1, True)
        self_up_vec = a_thr / thrust
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[:, 2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1)
 
    def render(self, ctl_dt):
        # Render depth images using CUDA ray casting
        # 使用CUDA光线追踪渲染深度图像
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None
 
    def find_vec_to_nearest_pt(self):
        # Find vector to nearest obstacle point
        # 找到最近障碍物点的向量
        p = self.p + self.v * self.sub_div
        nearest_pt = torch.empty_like(p)
        quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, self.n_drones_per_group)
        return nearest_pt - p
 
    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        # Execute physics simulation step
        # 执行物理模拟步骤
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)  # Update gravity disturbance / 更新重力扰动
        self.p_old = self.p
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        # Update attitude / 更新姿态
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)
 
    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        # Backup dynamics simulation implementation (pure PyTorch)
        # 备用的动力学模拟实现（纯PyTorch）
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        a_next = self.act + self.dg - z_drag - drag
        self.p_old = self.p
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next
 
        # Update attitude / 更新姿态
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)
 
