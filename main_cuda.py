from collections import defaultdict
import math
from random import normalvariate
from matplotlib import pyplot as plt
from env_cuda import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
from model import Model


# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)  # 恢复训练的checkpoint路径
parser.add_argument('--batch_size', type=int, default=64)  # 批量大小
parser.add_argument('--num_iters', type=int, default=50000)  # 训练迭代次数
parser.add_argument('--coef_v', type=float, default=1.0, help='smooth l1 of norm(v_set - v_real)')  # 速度匹配损失系数
parser.add_argument('--coef_speed', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_v_pred', type=float, default=2.0, help='mse loss for velocity estimation (no odom)')  # 速度预测损失系数
parser.add_argument('--coef_collide', type=float, default=2.0, help='softplus loss for collision (large if close to obstacle, zero otherwise)')  # 碰撞损失系数
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='quadratic clearance loss')  # 避障损失系数
parser.add_argument('--coef_d_acc', type=float, default=0.01, help='control acceleration regularization')  # 加速度正则化系数
parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='control jerk regularizatinon')  # 加加速度正则化系数
parser.add_argument('--coef_d_snap', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_ground_affinity', type=float, default=0., help='legacy')
parser.add_argument('--coef_bias', type=float, default=0.0, help='legacy')
parser.add_argument('--lr', type=float, default=1e-3)  # 学习率
parser.add_argument('--grad_decay', type=float, default=0.4)  # 梯度衰减系数
parser.add_argument('--speed_mtp', type=float, default=1.0)  # 速度倍数
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)  # 视场角的一半的正切
parser.add_argument('--timesteps', type=int, default=150)  # 每个episode的时间步数
parser.add_argument('--cam_angle', type=int, default=10)  # 相机角度
parser.add_argument('--single', default=False, action='store_true')  # 单无人机模式
parser.add_argument('--gate', default=False, action='store_true')  # 使用门障碍物
parser.add_argument('--ground_voxels', default=False, action='store_true')  # 使用地面障碍物
parser.add_argument('--scaffold', default=False, action='store_true')  # 使用脚手架障碍物
parser.add_argument('--random_rotation', default=False, action='store_true')  # 随机旋转
parser.add_argument('--yaw_drift', default=False, action='store_true')  # 偏航漂移
parser.add_argument('--no_odom', default=False, action='store_true')  # 无里程计
args = parser.parse_args()
writer = SummaryWriter()  # TensorBoard日志记录器
print(args)

device = torch.device('cuda')

# 初始化环境和模型
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle)
if args.no_odom:
    model = Model(7, 6)  # 无里程计:输入7维(目标速度3+up向量1+边距1+航向2),输出6维
else:
    model = Model(7+3, 6)  # 有里程计:额外3维速度
model = model.to(device)

# 恢复训练
if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)
optim = AdamW(model.parameters(), args.lr)  # 优化器
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)  # 学习率调度器

ctl_dt = 1 / 15  # 控制时间步长

scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    # 平滑字典中的值用于TensorBoard日志
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))
def barrier(x: torch.Tensor, v_to_pt):
    # 避障障碍函数
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

def is_save_iter(i):
    # 判断是否需要保存记录的迭代
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

pbar = tqdm(range(args.num_iters), ncols=80)  # 进度条
# depths = []
# states = []
B = args.batch_size
for i in pbar:
    # 重置环境和模型
    env.reset()
    model.reset()
    p_history = []
    v_history = []
    target_v_history = []
    vec_to_pt_history = []
    act_diff_history = []
    v_preds = []
    vid = []
    v_net_feats = []
    h = None  # GRU隐藏状态

    # 动作缓冲区,用于模拟控制延迟
    act_lag = 1#往后延迟一个时间步
    act_buffer = [env.act] * (act_lag + 1)
    target_v_raw = env.p_target - env.p
    if args.yaw_drift:#偏航角扰动
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)#5度的偏航角漂移
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([#偏航角漂移矩阵 Z轴的2D旋转矩阵。
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)


    for t in range(args.timesteps):  # 时间步循环
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)  # 随机时间步长
        depth, flow = env.render(ctl_dt)  # 渲染深度图
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())  # 查找最近障碍物

        if is_save_iter(i):
            vid.append(depth[4])  # 保存深度图用于可视化

        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()
        env.run(act_buffer[t], ctl_dt, target_v_raw)  # 运行动力学模拟

        R = env.R
        fwd = env.R[:, :, 0].clone()#旋转矩阵第0列x轴朝向
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0#压缩到2为平面
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)  # 重构旋转矩阵(水平朝向)

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)#模
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)  # 限制目标速度
        state = [
            torch.squeeze(target_v[:, None] @ R, 1),#R:世界系到机体旋转矩阵，机体系的速度
            env.R[:, 2],#up向量在世界系的方向
            env.margin[:, None]]    #安全半径
        local_v = torch.squeeze(env.v[:, None] @ R, 1)#R:世界系到机体旋转矩阵
        if not args.no_odom:
            state.insert(0, local_v)  # 添加局部速度
        state = torch.cat(state, -1)  # 拼接状态向量

        # 深度图归一化和下采样
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02#视差公式，突出近处细节，扰动
        x = F.max_pool2d(x[:, None], 4, 4)
        act, values, h = model(x, state, h)  # 模型推理value是空的 act[B:6]

        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)  # 解码动作
        v_preds.append(v_pred)
        #加速度（类似于动力）减速度（类似于阻尼的作用），引入推力误差
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std  # 考虑推力估计误差
        act_buffer.append(act)
        v_net_feats.append(torch.cat([act, local_v, h], -1))

        v_history.append(env.v)
        target_v_history.append(target_v)

    # 计算损失
    p_history = torch.stack(p_history)
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()  # 地面亲和力损失
    act_buffer = torch.stack(act_buffer)

    v_history = torch.stack(v_history)
    v_history_cum = v_history.cumsum(0)
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30  # 30步平均速度
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))  # 速度匹配损失

    v_preds = torch.stack(v_preds)
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())  # 速度预测损失

    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3  # 偏差损失

    jerk_history = act_buffer.diff(1, 0).mul(15)  # 加加速度
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)  # 加加速度的变化率
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()  # 加速度正则化
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()  # 加加速度正则化
    loss_d_snap = snap_history.pow(2).sum(-1).mean()  # 加加速度变化率正则化

    vec_to_pt_history = torch.stack(vec_to_pt_history)
    distance = torch.norm(vec_to_pt_history, 2, -1)
    distance = distance - env.margin
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)  # 接近速度
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)  # 避障损失
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()  # 碰撞损失

    speed_history = v_history.norm(2, -1)
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)  # 速度损失

    # 总损失
    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_ground_affinity + loss_ground_affinity

    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    pbar.set_description_str(f'loss: {loss:.3f}')
    optim.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optim.step()  # 更新参数
    sched.step()  # 更新学习率


    with torch.no_grad():
        # 计算统计指标
        avg_speed = speed_history.mean(0)
        success = torch.all(distance.flatten(0, 1) > 0, 0)
        _success = success.sum() / B
        smooth_dict({
            'loss': loss,
            'loss_v': loss_v,
            'loss_v_pred': loss_v_pred,
            'loss_obj_avoidance': loss_obj_avoidance,
            'loss_d_acc': loss_d_acc,
            'loss_d_jerk': loss_d_jerk,
            'loss_d_snap': loss_d_snap,
            'loss_bias': loss_bias,
            'loss_speed': loss_speed,
            'loss_collide': loss_collide,
            'loss_ground_affinity': loss_ground_affinity,
            'success': _success,
            'max_speed': speed_history.max(0).values.mean(),
            'avg_speed': avg_speed.mean(),
            'ar': (success * avg_speed).mean()})
        log_dict = {}
        if is_save_iter(i):
            # 保存可视化图表
            # vid = torch.stack(vid).cpu().div(10).clamp(0, 1)[None, :, None]
            fig_p, ax = plt.subplots()
            p_history = p_history[:, 4].cpu()
            ax.plot(p_history[:, 0], label='x')
            ax.plot(p_history[:, 1], label='y')
            ax.plot(p_history[:, 2], label='z')
            ax.legend()
            fig_v, ax = plt.subplots()
            v_history = v_history[:, 4].cpu()
            ax.plot(v_history[:, 0], label='x')
            ax.plot(v_history[:, 1], label='y')
            ax.plot(v_history[:, 2], label='z')
            ax.legend()
            fig_a, ax = plt.subplots()
            act_buffer = act_buffer[:, 4].cpu()
            ax.plot(act_buffer[:, 0], label='x')
            ax.plot(act_buffer[:, 1], label='y')
            ax.plot(act_buffer[:, 2], label='z')
            ax.legend()
            # writer.add_video('demo', vid, i + 1, 15)
            writer.add_figure('p_history', fig_p, i + 1)
            writer.add_figure('v_history', fig_v, i + 1)
            writer.add_figure('a_reals', fig_a, i + 1)
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), f'checkpoint{i//10000:04d}.pth')  # 保存checkpoint
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)  # 记录到TensorBoard
            scaler_q.clear()
