import dace
from dace.graph.graph import SubgraphView, Graph
from dace.graph.nodes import CodeNode, LibraryNode
from dace.properties import Property
import dace.symbolic as sym
import dace.dtypes as types

from typing import Any, Dict, List, Union, Type

from dace.perf.arith_counter import count_arithmetic_ops, \
                          count_arithmetic_ops_state, \
                          count_arithmetic_ops_subgraph

from dace.perf.movement_counter import count_moved_data, \
                             count_moved_data_state, \
                             count_moved_data_subgraph

import os, sys, platform
import subprocess

import matplotlib.pyplot as plot

import math
import sympy

class PerformanceSpec:
    ''' PerformanceSpec Struct
        contains hardware information for Roofline model
    '''
    def __init__(self,
                 peak_bandwidth,
                 peak_performance,
                 data_type: Type,
                 debug = True
                 ):
        self.peak_bandwidth = peak_bandwidth
        self.peak_performance = peak_performance
        self._data_type = data_type
        self.debug = debug

        # infer no of bytes used per unit
        self._infer_bytes(self._data_type)

    def _infer_bytes(self, dtype):
        # infer no of bytes used per unit
        successful = False
        try:
            # dace dtype
            self.bytes_per_datapoint = types._BYTES[self.data_type.type]
            successful = True
        except Exception:
            pass

        try:
            # np dtype
            self.bytes_per_datapoint = types._BYTES[dtype]
            successful = True
        except Exception:
            pass

        if not successful:
            print("WARNING: Could not infer data size from data_type")
            print("Assuming 64 bit precision for data")
            self.bytes_per_datapoint = 8
        else:
            if self.debug:
                print(f"Parsed data size of {self.bytes_per_datapoint} from input")

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, dtype):
        self._data_type = dtype
        self._infer_bytes(dtype)

    @staticmethod
    def from_json(filename):
        with open(filename) as json_file:
            data = json.load(json_file)
            spec = PerformanceSpec(data["peak_bandwidth"],
                                   data["peak_performance,"],
                                   data["data_type"])
        return spec




class Roofline:
    ''' class Roofline
        for OI calculation and Roofline plots
    '''
    def __init__(self,
                 specs: PerformanceSpec,
                 symbols: Union[List[dace.symbol], Dict[dace.symbol, int]],
                 debug: bool = True,
                 name: str = "roofline"
):

        self.specs = specs
        self.data = {}
        self.data_symbolic = {}
        self.debug = debug
        self.name = name

        if type(symbols) is list:
            self.symbols = {symbol: symbol.get() for symbol in symbols}
        else:
            self.symbols = symbols


    def evaluate(self, name: str,
                   graph: Graph,
                   symbols_replacement: Dict[str, Any] = None):

        if isinstance(graph, SubgraphView):
            name = f"Subgraph_{name}"
            print(f"Performance counter on Subgraph {name}")
        if isinstance(graph, dace.sdfg.SDFG):
            name = f"SDFG_{name}"
            print(f"Performance counter on SDFG {name}")
        if isinstance(graph, dace.sdfg.SDFGState):
            name = f"State_{name}"
            print(f"Performance counter on State {name}")

        if name in self.data:
            index = 1
            while(True):
                if not name+str(index) in self.data:
                    name = name+str(index)
                    break
                index += 1

        self.data[name] = None
        self.data_symbolic[name] = None
        memory_count = 0
        flop_count = 0

        symbols_replacement = symbols_replacement or {}

        if isinstance(graph, SubgraphView):
            memory_count = count_moved_data_subgraph(graph._graph, graph, symbols_replacement)
            flop_count = count_arithmetic_ops_subgraph(graph._graph, graph, symbols_replacement)
        if isinstance(graph, dace.sdfg.SDFG):
            memory_count = count_moved_data(graph, symbols_replacement)
            flop_count = count_arithmetic_ops(graph, symbols_replacement)
        if isinstance(graph, dace.sdfg.SDFGState):
            memory_count = count_moved_data_state(graph, symbols_replacement)
            flot_count = count_arithmetic_ops_state(graph, symbols_replacement)

        operational_intensity = flop_count / (memory_count * self.specs.bytes_per_datapoint)
        # evaluate remaining sym functions
        x, y = sympy.symbols('x y')
        sym_locals = {sympy.Function('int_floor') : sympy.Lambda((x,y), sympy.functions.elementary.integers.floor  (x/y)),
                      sympy.Function('int_ceil')  : sympy.Lambda((x,y), sympy.functions.elementary.integers.ceiling(x/y)),
                      sympy.Function('floor')     : sympy.Lambda((x),   sympy.functions.elementary.integers.floor(x)),
                      sympy.Function('ceiling')   : sympy.Lambda((x),   sympy.functions.elementary.integers.ceiling(x)),
                      }
        for fun, lam in sym_locals.items():
            operational_intensity.replace(fun, lam)


        self.data_symbolic[name] = operational_intensity
        try:
            self.data[name] = sym.evaluate(operational_intensity, self.symbols)
            self.data[name] = float(self.data[name])
        except TypeError:
            print("Not all the variables are defined in Symbols")
            print("Data after tried evaluation:")
            print(self.data[name])

        if self.debug:
            print(f"Determined OI {operational_intensity} on {graph}")
            print(f"Evaluated to {self.data[name]}")

    def plot(self, save_path = None, save_name = None, groups = None, show = False):
        x_ridge = self.specs.peak_performance / self.specs.peak_bandwidth
        y_ridge = self.specs.peak_performance
        ridge = [x_ridge, y_ridge]

        base_x = 2
        base_y = 10

        x_0 = 0
        x_2 = max(list(self.data.values()) + [10**3])
        y_0 = 0
        y_2 = self.specs.peak_performance

        plot.loglog([x_0,x_ridge,x_2],[y_0,y_ridge,y_2],
                     basex = base_x, basey = base_y, linewidth = 3.0)

        x_min = min(list(self.data.values()) + [0.5])*base_x**(-1.0)
        x_max = max(list(self.data.values()) + [20]) *base_x**(+2.0)
        y_min = 1       * base_y**(-1.5)
        y_max = y_ridge *(base_y**(+1.5))

        for (dname, oi) in self.data.items():
            plot.loglog([oi, oi], [y_0,y_max],label=dname, basex = base_x, basey = base_y)

        plot.title(f"{self.name}[{self.symbols}]")
        plot.xlabel(f"Operational Intensity (FLOP/byte)")
        plot.ylabel(f"GFLOP/s")
        plot.legend()

        # set axis

        plot.xlim(x_min, x_max)
        plot.ylim(y_min, y_max)

        plot.grid()

        if save_path is not None and save_path != '' and save_path[-1] != '/':
            save_path += '/'

        if save_name is None:
            save_name = self.name

        if save_path is not None:
            plot.savefig(fname = save_path+save_name)

        if show is True:
            plot.show()
            plot.close()

        plot.clf()


    def sdfv(self):
        pass

    @staticmethod
    def gather_system_specs():
        # TODO: Auto system specs inferrence
        self.gather_cpu_specs()
        self.gather_gpu_specs()

    @staticmethod
    def gather_cpu_specs():
        # TODO
        pass

    @staticmethod
    def gather_gpu_specs():
        # TODO
        pass



if __name__ == '__main__':
    # some tests

    # My Crapbook:
    # ark.intel.com
    # Bandwidth =  1867 Mhz   *   64 bits    *       2       /  8
    #            memory speed    channel size   dual channel   byte/bit
    # Peak Perf = 2.7 GhZ     *     4        *      2        *  4
    #            Compute speed    VEX/SIMD         FMA         core (incl. hyperthreading?)

    peak_bandwidth = 1.867 * 64 * 2 / 8
    peak_performance = 2.7 * 4 * 2 * 4

    spec = PerformanceSpec(peak_bandwidth, peak_performance, dace.float64)
    roofline = Roofline(spec, "test")
    roofline.data["test1"] = 1
    roofline.data["test2"] = 0.5
    roofline.data["test3"] = 0.25
    roofline.data["test4"] = 0.125
    roofline.data["test5"] = 2.5
    roofline.data["test6"] = 300
    roofline.plot(show = True, save_path = '')
