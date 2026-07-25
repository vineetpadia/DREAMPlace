##
# @file   NesterovAcceleratedGradientOptimizer.py
# @author Yibo Lin
# @date   Aug 2018
# @brief  Nesterov's accelerated gradient method proposed by e-place.
#

import os
import sys
import time
import pickle
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn
import pdb
import dreamplace.configure as configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.nesterov_update.nesterov_update_cuda as nesterov_update_cuda
else:
    nesterov_update_cuda = None


def _nesterov_acceleration_float32(a_k):
    """Match the CUDA float32 recurrence without launching scalar kernels."""
    a_k_squared = np.float32(a_k * a_k)
    root = np.float32(
        np.sqrt(np.float32(
            np.float32(4.0) * a_k_squared + np.float32(1.0)
        ))
    )
    a_kp1 = np.float32(
        np.float32(np.float32(1.0) + root) / np.float32(2.0)
    )
    coef = np.float32(np.float32(a_k - np.float32(1.0)) / a_kp1)
    return a_kp1, float(coef)


class NesterovAcceleratedGradientOptimizer(Optimizer):
    """
    @brief Follow the Nesterov's implementation of e-place algorithm 2
    http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf
    """
    def __init__(self, params, lr=required, obj_and_grad_fn=required, constraint_fn=None, use_bb=True):
        """
        @brief initialization
        @param params variable to optimize
        @param lr learning rate
        @param obj_and_grad_fn a callable function to get objective and gradient
        @param constraint_fn a callable function to force variables to satisfy all the constraints
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # u_k is major solution
        # v_k is reference solution
        # obj_k is the objective at v_k
        # a_k is optimization parameter
        # alpha_k is the step size
        # v_k_1 is previous reference solution
        # g_k_1 is gradient to v_k_1
        # obj_k_1 is the objective at v_k_1
        defaults = dict(lr=lr,
                u_k=[], v_k=[], g_k=[], obj_k=[], a_k=[], alpha_k=[],
                v_k_1=[], g_k_1=[], obj_k_1=[],
                u_kp1 = [None],
                v_kp1 = [None],
                obj_eval_count=0)
        super(NesterovAcceleratedGradientOptimizer, self).__init__(params, defaults)
        self.obj_and_grad_fn = obj_and_grad_fn
        self.constraint_fn = constraint_fn
        self.use_bb = use_bb
        self._can_fuse_boundary = all(
            hasattr(constraint_fn, name)
            for name in (
                "node_size_x", "node_size_y", "xl", "yl", "xh", "yh",
                "num_movable_nodes", "num_filler_nodes",
            )
        )

        # I do not know how to get generator's length
        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with single tensor is supported")

    def __setstate__(self, state):
        super(NesterovAcceleratedGradientOptimizer, self).__setstate__(state)

    def add_param_group(self, param_group):
        # DREAMPlace runs this optimizer eagerly. Bypass PyTorch's lazy
        # torch._dynamo-disabling wrapper, which otherwise imports the entire
        # compiler stack during construction.
        add_param_group = getattr(
            Optimizer.add_param_group, "__wrapped__", Optimizer.add_param_group
        )
        return add_param_group(self, param_group)

    def zero_grad(self, set_to_none=True):
        # Keep the base implementation and semantics without paying the same
        # lazy compiler import on the first optimization step.
        zero_grad = getattr(Optimizer.zero_grad, "__wrapped__", Optimizer.zero_grad)
        return zero_grad(self, set_to_none=set_to_none)

    def step(self, closure=None):
        if self.use_bb:
            self.step_bb(closure)
        else:
            self.step_nobb(closure)

    def step_nobb(self, closure=None):
        """
        @brief Performs a single optimization step.
        @param closure A callable closure function that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    # directly use p as v_k to save memory
                    #group['v_k'].append(torch.autograd.Variable(p.data, requires_grad=True))
                    group['v_k'].append(p)
                    obj, grad = obj_and_grad_fn(group['v_k'][i])
                    group['g_k'].append(grad.data.clone()) # must clone
                    group['obj_k'].append(obj.data.clone())
                u_k = group['u_k'][i]
                v_k = group['v_k'][i]
                g_k = group['g_k'][i]
                obj_k = group['obj_k'][i]
                if not group['a_k']:
                    if g_k.is_cuda and g_k.dtype == torch.float32:
                        group['a_k'].append(np.float32(1.0))
                    else:
                        group['a_k'].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group['v_k_1'].append(torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i].data-group['lr']*g_k)
                    obj, grad = obj_and_grad_fn(group['v_k_1'][i])
                    group['g_k_1'].append(grad.data)
                    group['obj_k_1'].append(obj.data.clone())
                a_k = group['a_k'][i]
                v_k_1 = group['v_k_1'][i]
                g_k_1 = group['g_k_1'][i]
                obj_k_1 = group['obj_k_1'][i]
                if not group['alpha_k']:
                    group['alpha_k'].append((v_k.data-v_k_1.data).norm(p=2) / (g_k-g_k_1).norm(p=2))
                alpha_k = group['alpha_k'][i]

                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)
                v_kp1 = group['v_kp1'][i]
                u_kp1_state = group.setdefault('u_kp1', [None])
                if u_kp1_state[i] is None:
                    u_kp1_state[i] = torch.zeros_like(v_k)
                u_kp1 = u_kp1_state[i]

                # line search with alpha_k as hint
                if isinstance(a_k, np.float32):
                    a_kp1, coef = _nesterov_acceleration_float32(a_k)
                else:
                    a_kp1 = (1 + (4*a_k.pow(2)+1).sqrt()) / 2
                    coef = (a_k-1) / a_kp1
                alpha_kp1 = 0
                backtrack_cnt = 0
                max_backtrack_cnt = 10

                #ttt = time.time()
                while True:
                    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    # Optimizer state updates are not part of the objective's
                    # autograd graph. Using the value view avoids constructing
                    # and immediately discarding a graph every iteration.
                    if (nesterov_update_cuda is not None and u_kp1.is_cuda
                            and isinstance(a_k, np.float32)):
                        can_fuse_boundary = self._can_fuse_boundary
                        if can_fuse_boundary:
                            nesterov_update_cuda.forward_with_boundary(
                                v_k.data, g_k, u_k, alpha_k, coef,
                                constraint_fn.node_size_x,
                                constraint_fn.node_size_y,
                                constraint_fn.xl, constraint_fn.yl,
                                constraint_fn.xh, constraint_fn.yh,
                                constraint_fn.num_movable_nodes,
                                constraint_fn.num_filler_nodes,
                                u_kp1, v_kp1.data)
                        else:
                            nesterov_update_cuda.forward(
                                v_k.data, g_k, u_k, alpha_k, coef,
                                u_kp1, v_kp1.data)
                    else:
                        can_fuse_boundary = False
                        torch.mul(alpha_k, g_k, out=u_kp1)
                        torch.sub(v_k.data, u_kp1, out=u_kp1)
                        #constraint_fn(u_kp1)
                        torch.sub(u_kp1, u_k, out=v_kp1.data)
                        v_kp1.data.mul_(coef)
                        torch.add(u_kp1, v_kp1.data, out=v_kp1.data)
                    # make sure v_kp1 subjects to constraints
                    # g_kp1 must correspond to v_kp1
                    if not can_fuse_boundary:
                        constraint_fn(v_kp1)

                    f_kp1, g_kp1 = obj_and_grad_fn(v_kp1)

                    #tt = time.time()
                    alpha_kp1 = torch.sqrt(torch.sum((v_kp1.data-v_k.data)**2) / torch.sum((g_kp1.data-g_k.data)**2))
                    # alpha_kp1 = torch.dist(v_kp1.data, v_k.data, p=2) / torch.dist(g_kp1.data, g_k.data, p=2)
                    backtrack_cnt += 1
                    group['obj_eval_count'] += 1
                    #logging.debug("\t\talpha_kp1 %.3f ms" % ((time.time()-tt)*1000))
                    #torch.cuda.synchronize()
                    #logging.debug(prof)

                    #logging.debug("alpha_kp1 = %g, line_search_count = %d, obj_eval_count = %d" % (alpha_kp1, backtrack_cnt, group['obj_eval_count']))
                    #logging.debug("|g_k| = %.6E, |g_kp1| = %.6E" % (g_k.norm(p=2), g_kp1.norm(p=2)))
                    if alpha_kp1 > 0.95*alpha_k or backtrack_cnt >= max_backtrack_cnt:
                        alpha_k.data.copy_(alpha_kp1.data)
                        break
                    else:
                        alpha_k.data.copy_(alpha_kp1.data)
                #if v_k.is_cuda:
                #    torch.cuda.synchronize()
                #logging.debug("\tline search %.3f ms" % ((time.time()-ttt)*1000))

                v_k_1.data.copy_(v_k.data)
                g_k_1.data.copy_(g_k.data)
                obj_k_1.data.copy_(obj_k.data)

                group['u_k'][i], group['u_kp1'][i] = u_kp1, u_k
                v_k.data.copy_(v_kp1.data)
                g_k.data.copy_(g_kp1.data)
                obj_k.data.copy_(f_kp1.data)
                if isinstance(a_k, np.float32):
                    group['a_k'][i] = a_kp1
                else:
                    a_k.data.copy_(a_kp1.data)

                # although the solution should be u_k
                # we need the gradient of v_k
                # the update of density weight also requires v_k
                # I do not know how to copy u_k back to p when exit yet
                #p.data.copy_(v_k.data)

        return loss

    def step_bb(self, closure=None):
        """
        @brief Performs a single optimization step.
        @param closure A callable closure function that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    group['v_k'].append(p)
                u_k = group['u_k'][i]
                v_k = group['v_k'][i]
                obj_k, g_k = obj_and_grad_fn(v_k)
                if not group['obj_k']:
                    group['obj_k'].append(None)
                group['obj_k'][i] = obj_k.data.clone()
                if not group['a_k']:
                    group['a_k'].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group['v_k_1'].append(torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i]-group['lr']*g_k)
                a_k = group['a_k'][i]
                v_k_1 = group['v_k_1'][i]
                obj_k_1, g_k_1 = obj_and_grad_fn(v_k_1)
                if not group['obj_k_1']:
                    group['obj_k_1'].append(None)
                group['obj_k_1'][i] = obj_k_1.data.clone()
                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)
                v_kp1 = group['v_kp1'][i]
                if not group['alpha_k']:
                    group['alpha_k'].append((v_k-v_k_1).norm(p=2) / (g_k-g_k_1).norm(p=2))
                alpha_k = group['alpha_k'][i]
                # line search with alpha_k as hint
                a_kp1 = (1 + (4*a_k.pow(2)+1).sqrt()) / 2
                coef = (a_k-1) / a_kp1
                with torch.no_grad():
                    s_k = (v_k - v_k_1)
                    y_k = (g_k - g_k_1)
                    bb_long_step_size = (s_k.dot(s_k) / torch.sum(s_k * y_k)).data
                    bb_short_step_size = (s_k.dot(y_k) / y_k.dot(y_k)).data
                    lip_step_size = (s_k.norm(p=2) / y_k.norm(p=2)).data
                    step_size = bb_short_step_size if bb_short_step_size > 0 else min(lip_step_size, alpha_k)
                
                # one step
                u_kp1 = v_k - step_size*g_k
                v_kp1.data.copy_(u_kp1 + coef*(u_kp1-u_k))
                constraint_fn(v_kp1)
                group['obj_eval_count'] += 1

                v_k_1.data.copy_(v_k.data)
                #g_k_1.data.copy_(g_k.data)
                #obj_k_1.data.copy_(obj_k.data)
                alpha_k.data.copy_(step_size.data)
                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                #g_k.data.copy_(g_kp1.data)
                #obj_k.data.copy_(f_kp1.data)
                a_k.data.copy_(a_kp1.data)

                # although the solution should be u_k
                # we need the gradient of v_k
                # the update of density weight also requires v_k
                # I do not know how to copy u_k back to p when exit yet
                #p.data.copy_(v_k.data)
        return loss
