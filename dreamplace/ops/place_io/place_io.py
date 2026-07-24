##
# @file   place_io.py
# @author Yibo Lin
# @date   Aug 2018
#

from torch.autograd import Function

import dreamplace.ops.place_io.place_io_cpp as place_io_cpp
from dreamplace.ops.place_io.place_io_cpp import SolutionFileFormat, Direction1DType, Direction2DType, OrientEnum, PlaceStatusEnum, MultiRowAttrEnum, SignalDirectEnum, PlanarDirectEnum, RegionTypeEnum


class PlaceIOFunction(Function):
    @staticmethod
    def build_args(params):
        """
        @brief build place_io command-line arguments
        """
        args = ["DREAMPlace"]
        if "aux_input" in params.__dict__ and params.aux_input:
            args.extend(["--bookshelf_aux_input", str(params.aux_input)])
        if "lef_input" in params.__dict__ and params.lef_input:
            if isinstance(params.lef_input, list):
                for lef in params.lef_input:
                    args.extend(["--lef_input", str(lef)])
            else:
                args.extend(["--lef_input", str(params.lef_input)])
        if "def_input" in params.__dict__ and params.def_input:
            args.extend(["--def_input", str(params.def_input)])
        if "verilog_input" in params.__dict__ and params.verilog_input:
            args.extend(["--verilog_input", str(params.verilog_input)])
        if "sort_nets_by_degree" in params.__dict__:
            args.extend(
                ["--sort_nets_by_degree", str(params.sort_nets_by_degree)]
            )
        return args

    @staticmethod
    def read_from_args(args):
        """
        @brief read design from prebuilt arguments and store it in the database
        """
        return place_io_cpp.forward(args)

    @staticmethod
    def read(params):
        """
        @brief read design and store in placement database
        """
        return PlaceIOFunction.read_from_args(PlaceIOFunction.build_args(params))

    @staticmethod
    def pydb(raw_db):
        """
        @brief convert to python database 
        @param raw_db original placement database 
        """
        return place_io_cpp.pydb(raw_db)

    @staticmethod
    def write(raw_db, filename, sol_file_format, node_x, node_y):
        """
        @brief write solution in specific format 
        @param raw_db original placement database 
        @param filename output file 
        @param sol_file_format solution file format, DEF|DEFSIMPLE|BOOKSHELF|BOOKSHELFALL
        @param node_x x coordinates of cells, only need movable cells; if none, use original position 
        @param node_y y coordinates of cells, only need movable cells; if none, use original position
        """
        return place_io_cpp.write(raw_db, filename, sol_file_format, node_x,
                                  node_y)

    @staticmethod
    def apply(raw_db, node_x, node_y):
        """
        @brief apply solution 
        @param raw_db original placement database 
        @param node_x x coordinates of cells, only need movable cells
        @param node_y y coordinates of cells, only need movable cells
        """
        return place_io_cpp.apply(raw_db, node_x, node_y)
