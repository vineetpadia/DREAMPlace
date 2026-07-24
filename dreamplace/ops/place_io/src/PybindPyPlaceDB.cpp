/**
 * @file   PybindPyPlaceDB.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Python binding for PyPlaceDB 
 */

#include "PyPlaceDB.h"

namespace _pybind {

template<typename T>
void sumPinWeightsLauncher(
    const DREAMPLACE_NAMESPACE::PyPlaceDB& db,
    const T* net_weights, T* node_weights) {
  // We assume that weights array has enough memory space.
  for (auto i = 0u; i < db.num_nodes; ++i) {
    node_weights[i] = 0;
    const auto& pins = db.node2pin_map[i];
    for (const auto& pin : pins)
      node_weights[i] += net_weights[db.pin2net_map[pin.cast<size_t>()]];
  }
}

/// \brief sum up pin weights inside a node.
/// \param db the placement database interface.
/// \param weights result array.
void sum_pin_weights(
    const DREAMPLACE_NAMESPACE::PyPlaceDB& db,
    at::Tensor net_weights,
    at::Tensor node_weights) {
  // Check torch tensors.
  CHECK_FLAT_CPU(net_weights);
  CHECK_FLAT_CPU(node_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_CONTIGUOUS(node_weights);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
    net_weights, "sumPinWeightsLauncher",
    [&] {
      sumPinWeightsLauncher<scalar_t>(db,
        DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(node_weights, scalar_t));
    });
}

/// \brief Expose a contiguous vector as a numpy array without copying.
///
/// The large flat arrays used to be pybind11::list, which meant building one Python
/// object per element in C++ and then walking that list again in np.array() on the
/// Python side -- roughly 4.5M object creations per read for a design like adaptec1.
/// Handing back a view costs nothing here; PlaceDB.py immediately copies it into an
/// array of its own dtype. `base` keeps the owning PyPlaceDB alive for the view.
template <typename T>
pybind11::array_t<T> as_numpy(const std::vector<T>& v, pybind11::object base) {
  return pybind11::array_t<T>({(pybind11::ssize_t)v.size()},
                              {(pybind11::ssize_t)sizeof(T)}, v.data(), base);
}

} // namespace _pybind

// Expanded outside namespace _pybind, so qualify the helper.
#define DREAMPLACE_VECTOR_PROPERTY(name)                              \
  def_property_readonly(#name, [](pybind11::object self) {            \
    return _pybind::as_numpy(                                         \
        self.cast<DREAMPLACE_NAMESPACE::PyPlaceDB&>().name, self);    \
  })

void bind_PyPlaceDB(pybind11::module& m)
{
    pybind11::class_<DREAMPLACE_NAMESPACE::PyPlaceDB>(m, "PyPlaceDB")
        .def(pybind11::init<>())
        .def_readwrite("num_nodes", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_nodes)
        .def_readwrite("num_terminals", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_terminals)
        .def_readwrite("num_terminal_NIs", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_terminal_NIs)
        .def_readwrite("node_name2id_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_name2id_map)
        .def_readwrite("node_names", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_names)
        .DREAMPLACE_VECTOR_PROPERTY(node_x)
        .DREAMPLACE_VECTOR_PROPERTY(node_y)
        .def_readwrite("node_orient", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_orient)
        .DREAMPLACE_VECTOR_PROPERTY(node_size_x)
        .DREAMPLACE_VECTOR_PROPERTY(node_size_y)
        .DREAMPLACE_VECTOR_PROPERTY(node2orig_node_map)
        .def_readwrite("pin_direct", &DREAMPLACE_NAMESPACE::PyPlaceDB::pin_direct)
        .DREAMPLACE_VECTOR_PROPERTY(pin_offset_x)
        .DREAMPLACE_VECTOR_PROPERTY(pin_offset_y)
        .def_readwrite("pin_names", &DREAMPLACE_NAMESPACE::PyPlaceDB::pin_names)
        .def_readwrite("net_name2id_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::net_name2id_map)
        .def_readwrite("pin_name2id_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::pin_name2id_map)
        .def_readwrite("net_names", &DREAMPLACE_NAMESPACE::PyPlaceDB::net_names)
        .def_readwrite("net2pin_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::net2pin_map)
        .DREAMPLACE_VECTOR_PROPERTY(flat_net2pin_map)
        .DREAMPLACE_VECTOR_PROPERTY(flat_net2pin_start_map)
        .DREAMPLACE_VECTOR_PROPERTY(net_weights)
        .DREAMPLACE_VECTOR_PROPERTY(net_weight_deltas)
        .DREAMPLACE_VECTOR_PROPERTY(net_criticality)
        .DREAMPLACE_VECTOR_PROPERTY(net_criticality_deltas)
        .def_readwrite("node2pin_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::node2pin_map)
        .DREAMPLACE_VECTOR_PROPERTY(flat_node2pin_map)
        .DREAMPLACE_VECTOR_PROPERTY(flat_node2pin_start_map)
        .def_readwrite("regions", &DREAMPLACE_NAMESPACE::PyPlaceDB::regions)
        .def_readwrite("flat_region_boxes", &DREAMPLACE_NAMESPACE::PyPlaceDB::flat_region_boxes)
        .def_readwrite("flat_region_boxes_start", &DREAMPLACE_NAMESPACE::PyPlaceDB::flat_region_boxes_start)
        .def_readwrite("node2fence_region_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::node2fence_region_map)
        .DREAMPLACE_VECTOR_PROPERTY(pin2node_map)
        .DREAMPLACE_VECTOR_PROPERTY(pin2net_map)
        .def_readwrite("rows", &DREAMPLACE_NAMESPACE::PyPlaceDB::rows)
        .def_readwrite("xl", &DREAMPLACE_NAMESPACE::PyPlaceDB::xl)
        .def_readwrite("yl", &DREAMPLACE_NAMESPACE::PyPlaceDB::yl)
        .def_readwrite("xh", &DREAMPLACE_NAMESPACE::PyPlaceDB::xh)
        .def_readwrite("yh", &DREAMPLACE_NAMESPACE::PyPlaceDB::yh)
        .def_readwrite("row_height", &DREAMPLACE_NAMESPACE::PyPlaceDB::row_height)
        .def_readwrite("site_width", &DREAMPLACE_NAMESPACE::PyPlaceDB::site_width)
        .def_readwrite("total_space_area", &DREAMPLACE_NAMESPACE::PyPlaceDB::total_space_area)
        .def_readwrite("num_movable_pins", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_movable_pins)
        .def_readwrite("num_routing_grids_x", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_routing_grids_x)
        .def_readwrite("num_routing_grids_y", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_routing_grids_y)
        .def_readwrite("routing_grid_xl", &DREAMPLACE_NAMESPACE::PyPlaceDB::routing_grid_xl)
        .def_readwrite("routing_grid_yl", &DREAMPLACE_NAMESPACE::PyPlaceDB::routing_grid_yl)
        .def_readwrite("routing_grid_xh", &DREAMPLACE_NAMESPACE::PyPlaceDB::routing_grid_xh)
        .def_readwrite("routing_grid_yh", &DREAMPLACE_NAMESPACE::PyPlaceDB::routing_grid_yh)
        .def_readwrite("unit_horizontal_capacities", &DREAMPLACE_NAMESPACE::PyPlaceDB::unit_horizontal_capacities)
        .def_readwrite("unit_vertical_capacities", &DREAMPLACE_NAMESPACE::PyPlaceDB::unit_vertical_capacities)
        .def_readwrite("initial_horizontal_demand_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::initial_horizontal_demand_map)
        .def_readwrite("initial_vertical_demand_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::initial_vertical_demand_map)
        .def("sum_pin_weights", &_pybind::sum_pin_weights)
        ;

}
