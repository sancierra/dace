import matplotlib as plt
import dace

from dace.sdfg.graph import SubgraphView

from .tester import test_executor, test_memlet

def obtain_subgraphs(score1, score2, **kwargs):
    # DUMMY:
    subgraphs = []
    for score in (score1, score2):
        if isinstance(score, ExecutionScore):
            r = test_executor(**kwargs)
        elif isinstance(score, MemletScore):
            r = test_executor(**kwargs)
        else:
            raise NotImplementedError("Wrong Score")
        subgraphs.append(r)

    return tuple(subgraphs)

def correlation_plot(program_name
                     score1: Type,
                     score2: Type,
                     enumerator_type = ConnectedEnumerator,
                     gpu = False,
                     nruns: int = None,
                     transformation_function = CompositeFusion,
                     condition_function = CompositeFusion.can_be_applied,
                     **kwargs):
    # get subgraphs
    r = obtain_subgraph(score1, score2,
                        enumerator_type = enumerator_type,
                        gpu = gpu,
                        nruns = nruns,
                        transformation_function = transformation_function,
                        condition_function = condition_function,
                        **kwargs)

    pass
