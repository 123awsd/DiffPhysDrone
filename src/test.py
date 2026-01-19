import math
import torch
import quadsim_cuda


class GDecay(torch.autograd.Function):
    # 梯度衰减的自定义自动微分函数
    @staticmethod
    def forward(ctx, x, alpha):
        # 前向传播:保存衰减因子alpha,直接返回输入x
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播:将梯度乘以衰减因子alpha
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply

# 初始化测试用的随机张量
R = torch.randn((64, 3, 3), dtype=torch.double, device='cuda')  # 旋转矩阵
dg = torch.randn((64, 3), dtype=torch.double, device='cuda')  # 重力扰动
z_drag_coef = torch.randn((64, 1), dtype=torch.double, device='cuda')  # Z轴阻力系数
drag_2 = torch.randn((64, 2), dtype=torch.double, device='cuda')  # 阻力系数
pitch_ctl_delay = torch.randn((64, 1), dtype=torch.double, device='cuda')  # 俯仰控制延迟
g_std = torch.tensor([[0, 0, -9.80665]], dtype=torch.double, device='cuda')  # 标准重力加速度
act_pred = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)  # 预测的动作/加速度
act = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)  # 实际动作
p = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)  # 位置
v = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)  # 速度
v_wind = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)  # 风速
a = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)  # 加速度

# 超参数
grad_decay = 0.4  # 梯度衰减系数
ctl_dt = 1/15  # 控制时间步长

def run_forward_pytorch(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt):
    # 使用PyTorch实现的前向传播函数,用于验证CUDA实现的正确性
    alpha = torch.exp(-pitch_ctl_delay * ctl_dt)  # 计算延迟系数
    act_next = act_pred * (1 - alpha) + act * alpha  # 混合预测动作和当前动作
    # dg = dg * math.sqrt(1 - ctl_dt) + torch.randn_like(dg) * 0.2 * math.sqrt(ctl_dt)
    v_fwd_s, v_left_s, v_up_s = (v.add(-v_wind)[:, None] @ R).unbind(-1)  # 将速度转换到机体坐标系
    # 0.047 = (4*rotor_drag_coefficient*motor_velocity_real) / sqrt(9.8)
    drag = drag_2[:, :1] * (v_fwd_s.abs() * v_fwd_s * R[..., 0] + v_left_s.abs() * v_left_s * R[..., 1] + v_up_s.abs() * v_up_s * R[..., 2] * z_drag_coef)
    drag += drag_2[:, 1:] * (v_fwd_s * R[..., 0] + v_left_s * R[..., 1] + v_up_s * R[..., 2] * z_drag_coef)  # 计算空气阻力
    a_next = act_next + dg - drag  # 计算下一时刻加速度
    p_next = g_decay(p, grad_decay ** ctl_dt) + v * ctl_dt + 0.5 * a * ctl_dt**2  # 计算下一时刻位置
    v_next = g_decay(v, grad_decay ** ctl_dt) + (a + a_next) / 2 * ctl_dt  # 计算下一时刻速度
    return act_next, p_next, v_next, a_next

# 使用CUDA实现的前向传播
act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, 0)

# 使用PyTorch实现的前向传播,用于验证
_act_next, _p_next, _v_next, _a_next = run_forward_pytorch(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt)

# 断言:比较CUDA和PyTorch实现的结果是否一致
assert torch.allclose(act_next, _act_next)
assert torch.allclose(a_next, _a_next)
assert torch.allclose(p_next, _p_next)
assert torch.allclose(v_next, _v_next)

# 生成随机梯度,用于反向传播测试
d_act_next = torch.randn_like(act_next)
d_p_next = torch.randn_like(p_next)
d_v_next = torch.randn_like(v_next)
d_a_next = torch.randn_like(a_next)

# 使用PyTorch自动微分进行反向传播
torch.autograd.backward(
    (_act_next, _p_next, _v_next, _a_next),
    (d_act_next, d_p_next, d_v_next, d_a_next),
)

# 使用CUDA实现的反向传播
d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next, grad_decay, ctl_dt)

# 断言:比较CUDA和PyTorch自动微分的梯度是否一致
assert torch.allclose(d_act_pred, act_pred.grad)
assert torch.allclose(d_act, act.grad)
assert torch.allclose(d_p, p.grad)
assert torch.allclose(d_v, v.grad)
assert torch.allclose(d_a, a.grad)
