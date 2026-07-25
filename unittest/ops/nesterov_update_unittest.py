import unittest

import numpy as np
import torch

import dreamplace.configure as configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from dreamplace.ops.move_boundary import move_boundary_cuda
    from dreamplace.ops.nesterov_update import nesterov_update_cuda


CUDA_AVAILABLE = (
    configure.compile_configurations["CUDA_FOUND"] == "TRUE"
    and torch.cuda.device_count()
)


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required")
class NesterovUpdateTest(unittest.TestCase):
    def test_matches_unfused_update_bitwise(self):
        for dtype, coefficient in (
            (torch.float32, float(np.float32(0.61803395))),
            (torch.float64, 0.6180339887498948),
        ):
            torch.manual_seed(9)
            numel = 100003
            v_k = torch.randn(numel, dtype=dtype, device="cuda") * 1000
            g_k = torch.randn(numel, dtype=dtype, device="cuda") * 0.01
            u_k = torch.randn(numel, dtype=dtype, device="cuda") * 1000
            alpha_k = torch.tensor(3.1415927, dtype=dtype, device="cuda")

            expected_u = torch.empty_like(v_k)
            expected_v = torch.empty_like(v_k)
            torch.mul(alpha_k, g_k, out=expected_u)
            torch.sub(v_k, expected_u, out=expected_u)
            torch.sub(expected_u, u_k, out=expected_v)
            expected_v.mul_(coefficient)
            torch.add(expected_u, expected_v, out=expected_v)

            actual_u = torch.empty_like(v_k)
            actual_v = torch.empty_like(v_k)
            nesterov_update_cuda.forward(
                v_k,
                g_k,
                u_k,
                alpha_k,
                coefficient,
                actual_u,
                actual_v,
            )

            self.assertTrue(torch.equal(expected_u, actual_u))
            self.assertTrue(torch.equal(expected_v, actual_v))

    def test_matches_separate_boundary_projection_bitwise(self):
        torch.manual_seed(17)
        num_nodes = 50003
        num_movable_nodes = 31007
        num_filler_nodes = 11003
        dtype = torch.float32
        v_k = torch.randn(num_nodes * 2, dtype=dtype, device="cuda") * 1500
        g_k = torch.randn_like(v_k) * 0.01
        u_k = torch.randn_like(v_k) * 1500
        alpha_k = torch.tensor(2.75, dtype=dtype, device="cuda")
        coefficient = float(np.float32(0.53125))
        node_size_x = torch.rand(num_nodes, dtype=dtype, device="cuda") * 20
        node_size_y = torch.rand(num_nodes, dtype=dtype, device="cuda") * 30
        bounds = (0.0, -10.0, 1000.0, 1200.0)

        expected_u = torch.empty_like(v_k)
        expected_v = torch.empty_like(v_k)
        nesterov_update_cuda.forward(
            v_k,
            g_k,
            u_k,
            alpha_k,
            coefficient,
            expected_u,
            expected_v,
        )
        move_boundary_cuda.forward(
            expected_v,
            node_size_x,
            node_size_y,
            *bounds,
            num_movable_nodes,
            num_filler_nodes,
        )

        actual_u = torch.empty_like(v_k)
        actual_v = torch.empty_like(v_k)
        nesterov_update_cuda.forward_with_boundary(
            v_k,
            g_k,
            u_k,
            alpha_k,
            coefficient,
            node_size_x,
            node_size_y,
            *bounds,
            num_movable_nodes,
            num_filler_nodes,
            actual_u,
            actual_v,
        )

        self.assertTrue(torch.equal(expected_u, actual_u))
        self.assertTrue(torch.equal(expected_v, actual_v))


if __name__ == "__main__":
    unittest.main()
