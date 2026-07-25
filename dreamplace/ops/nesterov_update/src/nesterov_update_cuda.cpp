#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void nesterovUpdateCudaLauncher(
    const T* v_k, const T* g_k, const T* u_k, const T* alpha_k,
    T coefficient, T* u_kp1, T* v_kp1, int numel);

void nesterov_update_forward(
    at::Tensor v_k, at::Tensor g_k, at::Tensor u_k, at::Tensor alpha_k,
    double coefficient, at::Tensor u_kp1, at::Tensor v_kp1) {
  CHECK_FLAT_CUDA(v_k);
  CHECK_FLAT_CUDA(g_k);
  CHECK_FLAT_CUDA(u_k);
  CHECK_CUDA(alpha_k);
  CHECK_FLAT_CUDA(u_kp1);
  CHECK_FLAT_CUDA(v_kp1);
  CHECK_CONTIGUOUS(v_k);
  CHECK_CONTIGUOUS(g_k);
  CHECK_CONTIGUOUS(u_k);
  CHECK_CONTIGUOUS(alpha_k);
  CHECK_CONTIGUOUS(u_kp1);
  CHECK_CONTIGUOUS(v_kp1);
  AT_ASSERTM(alpha_k.numel() == 1, "alpha_k must contain one element");
  AT_ASSERTM(
      g_k.numel() == v_k.numel() && u_k.numel() == v_k.numel() &&
          u_kp1.numel() == v_k.numel() && v_kp1.numel() == v_k.numel(),
      "all Nesterov state tensors must have the same length");
  AT_ASSERTM(
      g_k.scalar_type() == v_k.scalar_type() &&
          u_k.scalar_type() == v_k.scalar_type() &&
          alpha_k.scalar_type() == v_k.scalar_type() &&
          u_kp1.scalar_type() == v_k.scalar_type() &&
          v_kp1.scalar_type() == v_k.scalar_type(),
      "all Nesterov state tensors must have the same dtype");

  DREAMPLACE_DISPATCH_FLOATING_TYPES(v_k, "nesterovUpdateCudaLauncher", [&] {
    nesterovUpdateCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(v_k, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(g_k, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(u_k, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(alpha_k, scalar_t),
        static_cast<scalar_t>(coefficient),
        DREAMPLACE_TENSOR_DATA_PTR(u_kp1, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(v_kp1, scalar_t), v_k.numel());
  });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::nesterov_update_forward,
        "Nesterov state update (CUDA)");
}
