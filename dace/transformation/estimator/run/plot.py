import matplotlib.pyplot as plt
import numpy as np
import dace

from dace.sdfg.graph import SubgraphView
from dace.transformation.estimator import ExecutionScore, MemletScore
from dace.transformation.estimator import ConnectedEnumerator
from dace.transformation.estimator.programs import factory
from dace.transformation.subgraph.composite import CompositeFusion
from dace.properties import make_properties, Property
from dace.transformation.estimator.run.tester import test_scorer, get_sdfg, list_top

from typing import List, Type, Callable

@make_properties
class Plotter:
    
    ignore_failed_experiments = Property(desc = "Ignore failed experiments "
                                                "when plotting",
                                         dtype = bool,
                                         default = True)
    @staticmethod
    def obtain_subgraphs(program_name: str,
                        score1: Type,
                        score2: Type,
                        gpu: bool = False,
                        enumerator_type: Type = ConnectedEnumerator,
                        **kwargs):
        ''' Run and Score Subgraphs with two
            different Scoring Functions
        '''
        sdfg, graph = get_sdfg(program_name, gpu)
        io = factory.get_args(program_name)

        subgraphs = []
        for score in (score1, score2):
            r = test_scorer(sdfg = sdfg,
                            graph = graph,
                            io = io,
                            scoring_type = score,
                            enumerator_type = enumerator_type,
                            gpu = gpu,
                            **kwargs)
            subgraphs.append(r)

        return tuple(subgraphs)
    
    @staticmethod
    def correlation_plot(program_name: str,
                        score1: Type,
                        score2: Type,
                        enumerator_type: Type = ConnectedEnumerator,
                        gpu: bool = False,
                        nruns: int = 30,
                        transformation_function: Type = CompositeFusion,
                        condition_function: Callable = CompositeFusion.can_be_applied,
                        save: bool = True,
                        view: bool = False,
                        save_base_name: str = 'correlation',
                        plot_origin = True,
                        **kwargs):
        '''
        Creates a Correlation Plot of two ScoringFunctions
        on a graph of choice
        '''
        # get subgraphs
        r = Plotter.obtain_subgraphs(program_name = program_name,
                                     score1 = score1,
                                     score2 = score2,
                                     enumerator_type = enumerator_type,
                                     gpu = gpu,
                                     nruns = nruns,
                                     transformation_function = transformation_function,
                                     condition_function = condition_function,
                                     **kwargs)

        datapoints = list()
        # subgraphs to analyze
        (s1, s2) = r
        for (subgraph1, sc1) in s1:
            if Plotter.ignore_failed_experiments and sc1 in [0,-1]:
                continue 
            subgraph1 = set(subgraph1)
            for (subgraph2, sc2) in s2:
                if Plotter.ignore_failed_experiments and sc2 in [0,-1]:
                    continue
                if all([n in subgraph1 for n in subgraph2]) and \
                len(subgraph1) == len(subgraph2):
                    datapoints.append((sc1, sc2))

        x = list(d[0] for d in datapoints)
        y = list(d[1] for d in datapoints)

        plt.plot(x, y, 'o')
        if plot_origin:
            plt.plot([1.0],[1.0],'ro')
            x.append(1.0)
            y.append(1.0)
        m,b = np.polyfit(x,y,1)
        corr = np.corrcoef(x,y)
        plt.plot(x, [m*e + b for e in x])
    
        plt.xlabel(score1.name())
        plt.ylabel(score2.name())
        plt.annotate(f"m={m}, c={corr}", (0,0))

        if view:
            plt.show()
        if save:
            save_base_name += ('_' + program_name)
            save_base_name += '.pdf'
            plt.savefig(save_base_name)
        
        print(score1)
        list_top(s1)
        print(score2)
        list_top(s2)
        


if __name__ == '__main__':
    Plotter.correlation_plot('hdiff',
                             ExecutionScore,
                             MemletScore,
                             gpu = True,
                             transient_allocation = dace.dtypes.StorageType.Register,
                             schedule_innermaps = dace.dtypes.ScheduleType.Sequential,
                             stencil_unroll_loops = True,
                             deduplicate = True)
