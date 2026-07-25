import unittest

import torch

import dreamplace.configure as configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from dreamplace.ops.pin_weight_sum import pws_cuda


CUDA_AVAILABLE = (
    configure.compile_configurations["CUDA_FOUND"] == "TRUE"
    and torch.cuda.device_count()
)


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required")
class PreconditionTest(unittest.TestCase):
    def test_matches_separate_pytorch_operations_bitwise(self):
        for dtype in (torch.float32, torch.float64):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                torch.manual_seed(31)
                num_nodes = 100003
                gradient = torch.randn(
                    num_nodes * 2, dtype=dtype, device="cuda"
                )
                node_weights = torch.rand(
                    num_nodes, dtype=dtype, device="cuda"
                ) * 20
                node_areas = torch.rand(
                    num_nodes, dtype=dtype, device="cuda"
                ) * 100
                density_weight = torch.tensor(
                    [0.03125], dtype=dtype, device="cuda"
                )
                alpha = 8.0

                expected = gradient.clone()
                denominator = (
                    node_weights + alpha * density_weight * node_areas
                )
                denominator.clamp_(min=1.0)
                expected[:num_nodes].div_(denominator)
                expected[num_nodes:].div_(denominator)

                actual = gradient.clone()
                pws_cuda.precondition(
                    actual,
                    node_weights,
                    node_areas,
                    density_weight,
                    alpha,
                )
            stream.synchronize()
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
