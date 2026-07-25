#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void nesterovUpdateCudaLauncher(
    const T* v_k, const T* g_k, const T* u_k, T alpha_k,
    T coefficient, T* u_kp1, T* v_kp1, int numel);

template <typename T>
void nesterovUpdateWithBoundaryCudaLauncher(
    const T* v_k, const T* g_k, const T* u_k, T alpha_k,
    T coefficient, const T* node_size_x, const T* node_size_y, T xl, T yl,
    T xh, T yh, int num_movable_nodes, int num_filler_nodes, T* u_kp1,
    T* v_kp1, T* delta_squared, int numel);

template <typename T>
void squaredDifferenceCudaLauncher(
    const T* lhs, const T* rhs, T* output, int numel);

template <typename T>
void stepLengthCudaLauncher(const T* numerator, T* denominator);

void check_nesterov_tensors(
    const at::Tensor& v_k, const at::Tensor& g_k, const at::Tensor& u_k,
    const at::Tensor& u_kp1, const at::Tensor& v_kp1) {
  CHECK_FLAT_CUDA(v_k);
  CHECK_FLAT_CUDA(g_k);
  CHECK_FLAT_CUDA(u_k);
  CHECK_FLAT_CUDA(u_kp1);
  CHECK_FLAT_CUDA(v_kp1);
  CHECK_CONTIGUOUS(v_k);
  CHECK_CONTIGUOUS(g_k);
  CHECK_CONTIGUOUS(u_k);
  CHECK_CONTIGUOUS(u_kp1);
  CHECK_CONTIGUOUS(v_kp1);
  AT_ASSERTM(
      g_k.numel() == v_k.numel() && u_k.numel() == v_k.numel() &&
          u_kp1.numel() == v_k.numel() && v_kp1.numel() == v_k.numel(),
      "all Nesterov state tensors must have the same length");
  AT_ASSERTM(
      g_k.scalar_type() == v_k.scalar_type() &&
          u_k.scalar_type() == v_k.scalar_type() &&
          u_kp1.scalar_type() == v_k.scalar_type() &&
          v_kp1.scalar_type() == v_k.scalar_type(),
      "all Nesterov state tensors must have the same dtype");
}

void nesterov_update_forward(
    at::Tensor v_k, at::Tensor g_k, at::Tensor u_k, double alpha_k,
    double coefficient, at::Tensor u_kp1, at::Tensor v_kp1) {
  check_nesterov_tensors(v_k, g_k, u_k, u_kp1, v_kp1);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(v_k, "nesterovUpdateCudaLauncher", [&] {
    nesterovUpdateCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(v_k, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(g_k, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(u_k, scalar_t),
        static_cast<scalar_t>(alpha_k),
        static_cast<scalar_t>(coefficient),
        DREAMPLACE_TENSOR_DATA_PTR(u_kp1, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(v_kp1, scalar_t), v_k.numel());
  });
}

void nesterov_update_with_boundary_forward(
    at::Tensor v_k, at::Tensor g_k, at::Tensor u_k, double alpha_k,
    double coefficient, at::Tensor node_size_x, at::Tensor node_size_y,
    double xl, double yl, double xh, double yh, int num_movable_nodes,
    int num_filler_nodes, at::Tensor u_kp1, at::Tensor v_kp1,
    at::Tensor delta_squared) {
  check_nesterov_tensors(v_k, g_k, u_k, u_kp1, v_kp1);
  CHECK_FLAT_CUDA(node_size_x);
  CHECK_FLAT_CUDA(node_size_y);
  CHECK_FLAT_CUDA(delta_squared);
  CHECK_CONTIGUOUS(node_size_x);
  CHECK_CONTIGUOUS(node_size_y);
  CHECK_CONTIGUOUS(delta_squared);
  CHECK_EVEN(v_k);
  int num_nodes = v_k.numel() / 2;
  AT_ASSERTM(
      node_size_x.numel() == num_nodes && node_size_y.numel() == num_nodes,
      "node sizes must match half the position tensor length");
  AT_ASSERTM(
      node_size_x.scalar_type() == v_k.scalar_type() &&
          node_size_y.scalar_type() == v_k.scalar_type(),
      "node sizes and Nesterov state tensors must have the same dtype");
  AT_ASSERTM(
      delta_squared.numel() == v_k.numel() &&
          delta_squared.scalar_type() == v_k.scalar_type(),
      "delta_squared must match the Nesterov state tensors");

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      v_k, "nesterovUpdateWithBoundaryCudaLauncher", [&] {
        nesterovUpdateWithBoundaryCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(v_k, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(g_k, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(u_k, scalar_t),
            static_cast<scalar_t>(alpha_k),
            static_cast<scalar_t>(coefficient),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
            static_cast<scalar_t>(xl), static_cast<scalar_t>(yl),
            static_cast<scalar_t>(xh), static_cast<scalar_t>(yh),
            num_movable_nodes, num_filler_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(u_kp1, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(v_kp1, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(delta_squared, scalar_t),
            v_k.numel());
      });
}

void squared_difference_forward(
    at::Tensor lhs, at::Tensor rhs, at::Tensor output) {
  CHECK_FLAT_CUDA(lhs);
  CHECK_FLAT_CUDA(rhs);
  CHECK_FLAT_CUDA(output);
  CHECK_CONTIGUOUS(lhs);
  CHECK_CONTIGUOUS(rhs);
  CHECK_CONTIGUOUS(output);
  AT_ASSERTM(
      rhs.numel() == lhs.numel() && output.numel() == lhs.numel(),
      "squared-difference tensors must have the same length");
  AT_ASSERTM(
      rhs.scalar_type() == lhs.scalar_type() &&
          output.scalar_type() == lhs.scalar_type(),
      "squared-difference tensors must have the same dtype");

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      lhs, "squaredDifferenceCudaLauncher", [&] {
        squaredDifferenceCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(lhs, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(rhs, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(output, scalar_t), lhs.numel());
      });
}

void step_length_forward(at::Tensor numerator, at::Tensor denominator) {
  CHECK_CUDA(numerator);
  CHECK_CUDA(denominator);
  CHECK_CONTIGUOUS(numerator);
  CHECK_CONTIGUOUS(denominator);
  AT_ASSERTM(
      numerator.numel() == 1 && denominator.numel() == 1,
      "step-length inputs must contain one element");
  AT_ASSERTM(
      numerator.scalar_type() == denominator.scalar_type(),
      "step-length inputs must have the same dtype");

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      numerator, "stepLengthCudaLauncher", [&] {
        stepLengthCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(numerator, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(denominator, scalar_t));
      });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::nesterov_update_forward,
        "Nesterov state update (CUDA)");
  m.def(
      "forward_with_boundary",
      &DREAMPLACE_NAMESPACE::nesterov_update_with_boundary_forward,
      "Nesterov state update with boundary projection (CUDA)");
  m.def(
      "squared_difference",
      &DREAMPLACE_NAMESPACE::squared_difference_forward,
      "Elementwise squared difference (CUDA)");
  m.def(
      "step_length",
      &DREAMPLACE_NAMESPACE::step_length_forward,
      "Compute a Nesterov step length in place (CUDA)");
}
