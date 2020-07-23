import dace
from dace.sdfg import SDFG

from dace.transformation.heterogeneous.pipeline import fusion, expand_maps, expand_reduce
from dace.transformation.interstate import StateFusion

vadv_unfused = SDFG.from_file('vadv-input.sdfg')
# apply state fusion exhaustively
vadv_unfused.apply_transformations_repeated(StateFusion)
vadv_unfused.view()

vadv_fused_partial = SDFG.from_file('vadv-2part.sdfg')
vadv_fused_partial.view()

vadv_fused_full = SDFG.from_file('vadv-fused.sdfg')
vadv_fused_full.view()
