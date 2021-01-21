import dace   


sdfg = dace.sdfg.SDFG.from_file('encoder.sdfg')
sdfg.expand_library_nodes()
sdfg.apply_strict_transformations()
sdfg.save('encoder_expanded.sdfg')