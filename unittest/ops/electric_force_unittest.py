import unittest

import torch

import dreamplace.configure as configure
from dreamplace.ops.electric_potential import electric_potential_cpp

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from dreamplace.ops.electric_potential import electric_potential_cuda


CUDA_AVAILABLE = (
    configure.compile_configurations["CUDA_FOUND"] == "TRUE"
    and torch.cuda.device_count()
)


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required")
class ElectricForceTest(unittest.TestCase):
    def test_deterministic_force_handles_fixed_nodes_and_fillers(self):
        for dtype in (torch.float32, torch.float64):
            num_bins_x = 8
            num_bins_y = 8
            num_nodes = 8
            num_movable_nodes = 3
            num_filler_nodes = 2
            num_physical_nodes = num_nodes - num_filler_nodes

            x = torch.tensor(
                [2, 15, 23, 35, 45, 55, 65, 72], dtype=dtype
            )
            y = torch.tensor(
                [4, 12, 28, 33, 47, 58, 63, 71], dtype=dtype
            )
            pos = torch.cat((x, y))
            node_size_x = torch.tensor(
                [5, 7, 6, 4, 8, 3, 5, 6], dtype=dtype
            )
            node_size_y = torch.tensor(
                [6, 5, 8, 7, 4, 6, 5, 7], dtype=dtype
            )
            offset_x = torch.tensor(
                [0, 0.5, 0, 1, 0, 0.5, 0, 1], dtype=dtype
            )
            offset_y = torch.tensor(
                [0.5, 0, 1, 0, 0.5, 0, 1, 0], dtype=dtype
            )
            ratio = torch.tensor(
                [1, 0.75, 1.25, 1, 0.5, 1, 0.8, 1.2], dtype=dtype
            )
            field_x = torch.linspace(
                -0.5, 1.25, num_bins_x * num_bins_y, dtype=dtype
            )
            field_y = torch.linspace(
                1.5, -0.75, num_bins_x * num_bins_y, dtype=dtype
            )
            bin_center_x = torch.arange(num_bins_x, dtype=dtype) * 10 + 5
            bin_center_y = torch.arange(num_bins_y, dtype=dtype) * 10 + 5
            grad_pos = torch.tensor(0.75, dtype=dtype)

            common = (
                grad_pos,
                num_bins_x,
                num_bins_y,
                2,
                2,
                2,
                2,
                field_x,
                field_y,
                pos,
                node_size_x,
                node_size_y,
                offset_x,
                offset_y,
                ratio,
                bin_center_x,
                bin_center_y,
                0.0,
                0.0,
                80.0,
                80.0,
                10.0,
                10.0,
                num_movable_nodes,
                num_filler_nodes,
            )
            expected = electric_potential_cpp.electric_force(*common)

            cuda_common = tuple(
                value.cuda() if torch.is_tensor(value) else value
                for value in common
            )
            sorted_node_map = torch.arange(
                num_movable_nodes, dtype=torch.int32, device="cuda"
            )
            actual = electric_potential_cuda.electric_force(
                *cuda_common, 1, sorted_node_map
            ).cpu()
            actual_negative = electric_potential_cuda.electric_force_negative(
                *cuda_common, 1, sorted_node_map
            ).cpu()

            torch.testing.assert_close(actual, expected)
            expected_negative = -expected
            self.assertTrue(
                torch.equal(
                    actual_negative.view(
                        torch.int32 if dtype == torch.float32 else torch.int64
                    ),
                    expected_negative.view(
                        torch.int32 if dtype == torch.float32 else torch.int64
                    ),
                )
            )
            fixed_x = actual[num_movable_nodes:num_physical_nodes]
            fixed_y = actual[
                num_nodes + num_movable_nodes:
                num_nodes + num_physical_nodes
            ]
            expected_fixed = torch.zeros(
                num_physical_nodes - num_movable_nodes, dtype=dtype
            )
            self.assertTrue(
                torch.equal(fixed_x, expected_fixed)
            )
            self.assertTrue(
                torch.equal(fixed_y, expected_fixed)
            )


if __name__ == "__main__":
    unittest.main()
