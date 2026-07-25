#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computePWSCudaLauncher(const T* net_weights, const int* flat_nodepin,
                           const int* nodepin_start, const int* pin2net_map,
                           int num_physical_nodes, T* node_weights);

template <typename T>
void preconditionCudaLauncher(
    T* grad, const T* node_weights, const T* node_areas,
    const T* density_weight, T alpha, int num_nodes);

/// @brief Compute node weights: the sum of all related net weights.
at::Tensor pws_forward(at::Tensor net_weights, at::Tensor flat_nodepin,
                       at::Tensor nodepin_start, at::Tensor pin2net_map,
                       int num_nodes) {
  CHECK_FLAT_CUDA(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CUDA(flat_nodepin);
  CHECK_CONTIGUOUS(flat_nodepin);
  CHECK_FLAT_CUDA(nodepin_start);
  CHECK_CONTIGUOUS(nodepin_start);
  CHECK_FLAT_CUDA(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);

  int num_physical_nodes = nodepin_start.numel() - 1;
  at::Tensor node_weights = at::zeros(num_nodes, net_weights.options());
  DREAMPLACE_DISPATCH_FLOATING_TYPES(net_weights, "computePWSCudaLauncher", [&] {
    computePWSCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(flat_nodepin, int),
        DREAMPLACE_TENSOR_DATA_PTR(nodepin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
        num_physical_nodes,
        DREAMPLACE_TENSOR_DATA_PTR(node_weights, scalar_t));
  });
  return node_weights;
}

void precondition_forward(
    at::Tensor grad, at::Tensor node_weights, at::Tensor node_areas,
    at::Tensor density_weight, double alpha) {
  CHECK_FLAT_CUDA(grad);
  CHECK_FLAT_CUDA(node_weights);
  CHECK_FLAT_CUDA(node_areas);
  CHECK_CUDA(density_weight);
  CHECK_CONTIGUOUS(grad);
  CHECK_CONTIGUOUS(node_weights);
  CHECK_CONTIGUOUS(node_areas);
  CHECK_CONTIGUOUS(density_weight);
  int num_nodes = node_weights.numel();
  AT_ASSERTM(
      grad.numel() == num_nodes * 2 && node_areas.numel() == num_nodes,
      "gradient and node arrays must have compatible lengths");
  AT_ASSERTM(
      density_weight.numel() == 1,
      "density_weight must contain one element");
  AT_ASSERTM(
      node_weights.scalar_type() == grad.scalar_type() &&
          node_areas.scalar_type() == grad.scalar_type() &&
          density_weight.scalar_type() == grad.scalar_type(),
      "all preconditioner tensors must have the same dtype");
  AT_ASSERTM(
      node_weights.get_device() == grad.get_device() &&
          node_areas.get_device() == grad.get_device() &&
          density_weight.get_device() == grad.get_device(),
      "all preconditioner tensors must be on the same CUDA device");

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      grad, "preconditionCudaLauncher", [&] {
        preconditionCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(grad, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_weights, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_areas, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(density_weight, scalar_t),
            static_cast<scalar_t>(alpha), num_nodes);
      });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pws_forward, "PWS forward (CUDA)");
  m.def(
      "precondition",
      &DREAMPLACE_NAMESPACE::precondition_forward,
      "Apply the placement gradient preconditioner (CUDA)");
}
