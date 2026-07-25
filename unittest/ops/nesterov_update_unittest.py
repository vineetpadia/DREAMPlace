import unittest

import numpy as np
import torch

import dreamplace.configure as configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from dreamplace.ops.nesterov_update import nesterov_update_cuda


class NesterovUpdateTest(unittest.TestCase):
    @unittest.skipUnless(
        configure.compile_configurations["CUDA_FOUND"] == "TRUE"
        and torch.cuda.device_count(),
        "CUDA is required",
    )
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


if __name__ == "__main__":
    unittest.main()
