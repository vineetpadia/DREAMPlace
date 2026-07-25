import unittest

import torch

from dreamplace.NesterovAcceleratedGradientOptimizer import (
    NesterovAcceleratedGradientOptimizer,
)


class NesterovOptimizerTest(unittest.TestCase):
    @staticmethod
    def _make_optimizer(clone_gradient):
        position = torch.tensor(
            [3.0, -2.0, 0.5, 7.0],
            dtype=torch.float32,
            requires_grad=True,
        )

        def objective_and_gradient(candidate):
            if candidate.grad is not None:
                candidate.grad.zero_()
            weights = candidate.new_tensor([1.0, 2.0, 3.0, 5.0])
            objective = (candidate.square() * weights * 0.5).sum()
            objective.backward()
            gradient = (
                candidate.grad.clone() if clone_gradient else candidate.grad
            )
            return objective, gradient

        objective_and_gradient(position)
        optimizer = NesterovAcceleratedGradientOptimizer(
            [position],
            lr=0.1,
            obj_and_grad_fn=objective_and_gradient,
            constraint_fn=lambda candidate: candidate,
            use_bb=False,
        )
        return position, optimizer

    def test_buffer_rotation_matches_gradient_copy_fallback(self):
        rotated_position, rotated_optimizer = self._make_optimizer(False)
        copied_position, copied_optimizer = self._make_optimizer(True)
        original_position_ptr = rotated_position.data_ptr()

        for step in range(12):
            rotated_optimizer.step()
            copied_optimizer.step()
            self.assertTrue(
                torch.equal(rotated_position, copied_position),
                f"position mismatch at step {step}",
            )
            rotated_gradient = rotated_optimizer.param_groups[0]["g_k"][0]
            copied_gradient = copied_optimizer.param_groups[0]["g_k"][0]
            self.assertTrue(
                torch.equal(rotated_gradient, copied_gradient),
                f"gradient mismatch at step {step}",
            )

            scratch = rotated_optimizer.param_groups[0]["v_kp1"][0]
            if step % 2 == 0:
                self.assertEqual(scratch.data_ptr(), original_position_ptr)
            else:
                self.assertEqual(
                    rotated_position.data_ptr(), original_position_ptr
                )


if __name__ == "__main__":
    unittest.main()
