#include <torch/extension.h>

#include <vector>

// CUDA函数的前向声明

void render_cuda(  // 渲染深度图
    torch::Tensor canvas,  // 输出:渲染画布
    torch::Tensor flow,  // 输出:光流
    torch::Tensor balls,  // 输入:球体障碍物
    torch::Tensor cylinders,  // 输入:垂直圆柱障碍物
    torch::Tensor cylinders_h,  // 输入:水平圆柱障碍物
    torch::Tensor voxels,  // 输入:体素障碍物
    torch::Tensor R,  // 输入:当前旋转矩阵
    torch::Tensor R_old,  // 输入:上一帧旋转矩阵
    torch::Tensor pos,  // 输入:当前位置
    torch::Tensor pos_old,  // 输入:上一帧位置
    float drone_radius,  // 输入:无人机半径
    int n_drones_per_group,  // 输入:每组的无人机数量
    float fov_x_half_tan);  // 输入:视场角的一半的正切值

void rerender_backward_cuda(  // 渲染反向传播
    torch::Tensor depth,  // 输入:深度图
    torch::Tensor dddp,  // 输出:位置梯度
    float fov_x_half_tan);  // 输入:视场角的一半的正切值

void find_nearest_pt_cuda(  // 查找最近障碍物点
    torch::Tensor nearest_pt,  // 输出:最近的障碍物点
    torch::Tensor balls,  // 输入:球体障碍物
    torch::Tensor cylinders,  // 输入:垂直圆柱障碍物
    torch::Tensor cylinders_h,  // 输入:水平圆柱障碍物
    torch::Tensor voxels,  // 输入:体素障碍物
    torch::Tensor pos,  // 输入:位置
    float drone_radius,  // 输入:无人机半径
    int n_drones_per_group);  // 输入:每组的无人机数量

torch::Tensor update_state_vec_cuda(  // 更新状态向量(旋转矩阵)
    torch::Tensor R,  // 输入/输出:旋转矩阵
    torch::Tensor a_thr,  // 输入:推力加速度
    torch::Tensor v_pred,  // 输入:预测速度
    torch::Tensor alpha,  // 输入:混合系数
    float yaw_inertia);  // 输入:偏航惯性

std::vector<torch::Tensor> run_forward_cuda(  // 前向传播
    torch::Tensor R,  // 输入:旋转矩阵
    torch::Tensor dg,  // 输入:重力扰动
    torch::Tensor z_drag_coef,  // 输入:Z轴阻力系数
    torch::Tensor drag_2,  // 输入:阻力系数
    torch::Tensor pitch_ctl_delay,  // 输入:俯仰控制延迟
    torch::Tensor act_pred,  // 输入:预测动作
    torch::Tensor act,  // 输入:当前动作
    torch::Tensor p,  // 输入:位置
    torch::Tensor v,  // 输入:速度
    torch::Tensor v_wind,  // 输入:风速
    torch::Tensor a,  // 输入:加速度
    float ctl_dt,  // 输入:控制时间步长
    float airmode_av2a);  // 输入:空模加速度参数

std::vector<torch::Tensor> run_backward_cuda(  // 反向传播
    torch::Tensor R,  // 输入:旋转矩阵
    torch::Tensor dg,  // 输入:重力扰动
    torch::Tensor z_drag_coef,  // 输入:Z轴阻力系数
    torch::Tensor drag_2,  // 输入:阻力系数
    torch::Tensor pitch_ctl_delay,  // 输入:俯仰控制延迟
    torch::Tensor v,  // 输入:速度
    torch::Tensor v_wind,  // 输入:风速
    torch::Tensor act_next,  // 输入:下一时刻动作
    torch::Tensor _d_act_next,  // 输入:下一时刻动作的梯度
    torch::Tensor d_p_next,  // 输入:下一时刻位置的梯度
    torch::Tensor d_v_next,  // 输入:下一时刻速度的梯度
    torch::Tensor _d_a_next,  // 输入:下一时刻加速度的梯度
    float grad_decay,  // 输入:梯度衰减系数
    float ctl_dt);  // 输入:控制时间步长

// C++ Python绑定接口

// // NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
// #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// void render(
//     torch::Tensor canvas,
//     torch::Tensor nearest_pt,
//     torch::Tensor balls,
//     torch::Tensor cylinders,
//     torch::Tensor voxels,
//     torch::Tensor Rt) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(bias);
//   CHECK_INPUT(old_h);
//   CHECK_INPUT(old_cell);

//   return render_cuda(input, weights, bias, old_h, old_cell);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 使用pybind11导出CUDA函数到Python
  m.def("render", &render_cuda, "render (CUDA)");  // 渲染深度图
  m.def("find_nearest_pt", &find_nearest_pt_cuda, "find_nearest_pt (CUDA)");  // 查找最近障碍物点
  m.def("update_state_vec", &update_state_vec_cuda, "update_state_vec (CUDA)");  // 更新旋转矩阵
  m.def("run_forward", &run_forward_cuda, "run_forward_cuda (CUDA)");  // 前向传播
  m.def("run_backward", &run_backward_cuda, "run_backward_cuda (CUDA)");  // 反向传播
  m.def("rerender_backward", &rerender_backward_cuda, "rerender_backward_cuda (CUDA)");  // 渲染反向传播
}
