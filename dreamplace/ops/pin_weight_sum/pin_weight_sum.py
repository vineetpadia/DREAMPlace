import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import pdb

import dreamplace.ops.pin_weight_sum.pws_cpp as pws_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.pin_weight_sum.pws_cuda as pws_cuda

class PinWeightSumFunction(Function):
    """accumulate pin weights of a node.
    @param net_weights weight of nets
    @param flat_nodepin flat nodepin map, length of #pins 
    @param nodepin_start starting index in nodepin map for each net, length of #nodes+1,
            the last entry is #pins  
    @param pin2net_map pin2net map, second set of options
    @param num_nodes the total number of nodes including fillers
    """
    @staticmethod
    def forward(ctx, net_weights, flat_nodepin, nodepin_start, pin2net_map, num_nodes):
        if net_weights.is_cuda:
            func = pws_cuda.forward
        else:
            func = pws_cpp.forward
        output = func(net_weights, flat_nodepin, nodepin_start, pin2net_map, num_nodes)
        return output

class PinWeightSum(nn.Module):
    """ 
    @brief Accumulate pin weights of a node. 
    Support one algorithm: node-by-node (TODO: atomic)
    Different parameters are required for different algorithms. 
    """
    def __init__(self,
                 flat_nodepin=None,
                 nodepin_start=None,
                 pin2net_map=None,
                 num_nodes=None,
                 algorithm='node-by-node'):
        """
        @brief initialization 
        @param flat_nodepin flat nodepin map, length of #pins 
        @param nodepin_start starting index in nodepin map for each net, length of #nodes+1,
                the last entry is #pins  
        @param pin2net_map pin2net map, second set of options 
        @param algorithm must be node-by-node
        """
        super(PinWeightSum, self).__init__()
        if algorithm == 'node-by-node':
            assert flat_nodepin is not None and nodepin_start is not None, \
                "flat_nodepin, nodepin_start are requried parameters for algorithm node-by-node"
        self.flat_nodepin = flat_nodepin
        self.nodepin_start = nodepin_start
        self.pin2net_map = pin2net_map
        self.num_nodes = num_nodes
        self.algorithm = algorithm
        self._cache_key = None
        self._cache_value = None

    def invalidate(self):
        """Drop the memoized sum.

        Required whenever net weights are mutated through a path that does not bump the
        tensor's autograd version counter -- notably the HeteroSTA timer, which writes
        net_weights in place on the GPU via raw pointers (see NonLinearPlace: "net_weights
        are modified in-place on GPU - no copy needed"). The OpenTimer path uses .copy_(),
        which does bump the version and is caught by the key below, but do not rely on that.
        """
        self._cache_key = None
        self._cache_value = None

    def precondition(
        self, grad, node_weights, node_areas, density_weight, alpha
    ):
        """Apply the common single-density gradient preconditioner."""
        pws_cuda.precondition(
            grad, node_weights, node_areas, density_weight, alpha
        )
        return grad

    def forward(self, net_weights):
        if self.algorithm == 'node-by-node':
            # Net weights are constant for non-timing-driven placement, yet this sum is
            # recomputed on every gradient pass (see the note in PlaceObj.PreconditionOp).
            # Memoize on identity + version so ordinary runs traverse the netlist once.
            key = (net_weights.data_ptr(), net_weights._version, net_weights.shape)
            if self._cache_key == key and self._cache_value is not None:
                return self._cache_value
            value = PinWeightSumFunction.apply(
                net_weights,
                self.flat_nodepin, self.nodepin_start,
                self.pin2net_map, self.num_nodes)
            self._cache_key = key
            self._cache_value = value
            return value
