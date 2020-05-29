import dace
import diode

import copy
import os
import re
import time

from dace.config import Config
from dace.sdfg import propagation
from dace.sdfg.graph import SubgraphView
from dace.transformation import pattern_matching

# register all the transformations
from dace.transformation import dataflow, interstate #, heterogeneous
from dace.transformation.optimizer import SDFGOptimizer

# perf stuff
from dace.perf.roofline import Roofline
from dace.perf.sdfv_roofline import view


class SDFGRooflineOptimizer(SDFGOptimizer):
    ''' SDFGOptimizer - style command line optimizer
        with integrated Roofline plotter
    '''

    def __init__(self, sdfg, roofline, inplace=False, roofline_save = None):
        super().__init__(sdfg, inplace=inplace)
        #self.sdfg.fill_scope_connectors()
        #if self.sdfg._propagate:
        #    dace.sdfg.propagate_labels_sdfg(self.sdfg)

        self.roofline = roofline
        self.roofline_save = roofline_save


    def optimize(self):
        """ Adapted from transformation.optimizer
            A command-line UI for applying patterns on the SDFG.
            :return: An optimized SDFG object
        """

        sdfg_file = self.sdfg.name + '.sdfg'
        if os.path.isfile(sdfg_file):
            ui_input = input('An SDFG with the filename "%s" was found. '
                             'Would you like to use it instead? [Y/n] ' %
                             sdfg_file)
            if len(ui_input) == 0 or ui_input[0] not in ['n', 'N']:
                return dace.SDFG.from_file(sdfg_file)

        # Visualize SDFGs during optimization process
        VISUALIZE_SDFV = Config.get_bool('optimizer', 'visualize_sdfv')
        SAVE_INTERMEDIATE = Config.get_bool('optimizer', 'save_intermediate')

        self.roofline.evaluate('baseline', self.sdfg)

        if SAVE_INTERMEDIATE:
            self.sdfg.save(os.path.join('_dacegraphs', 'before.sdfg'))
        if VISUALIZE_SDFV:
            view(self.sdfg, self.roofline)

        cumulated_pattern_name = ''
        # Optimize until there is not pattern matching or user stops the process.
        pattern_counter = 0
        while True:
            # Print in the UI all the pattern matching options.
            ui_options = sorted(self.get_pattern_matches())
            ui_options_idx = 0
            for pattern_match in ui_options:
                sdfg = self.sdfg.sdfg_list[pattern_match.sdfg_id]
                print('%d. Transformation %s' %
                      (ui_options_idx, pattern_match.print_match(sdfg)))
                ui_options_idx += 1

            # If no pattern matchings were found, quit.
            if ui_options_idx == 0:
                print('No viable transformations found')
                break

            ui_input = input(
                'Select the pattern to apply (0 - %d or name$id): ' %
                (ui_options_idx - 1))

            pattern_name, occurrence, param_dict = _parse_cli_input(ui_input)

            pattern_match = None
            if (pattern_name is None and occurrence >= 0
                    and occurrence < ui_options_idx):
                pattern_match = ui_options[occurrence]
            elif pattern_name is not None:
                counter = 0
                for match in ui_options:
                    if type(match).__name__ == pattern_name:
                        if occurrence == counter:
                            pattern_match = match
                            break
                        counter = counter + 1

            if pattern_match is None:
                print(
                    'You did not select a valid option. Quitting optimization ...'
                )
                if self.roofline_save is not None:
                    self.roofline.plot(save_path=self.roofline_save)
                break

            match_id = (str(occurrence) if pattern_name is None else '%s$%d' %
                        (pattern_name, occurrence))
            sdfg = self.sdfg.sdfg_list[pattern_match.sdfg_id]
            print('You selected (%s) pattern %s with parameters %s' %
                  (match_id, pattern_match.print_match(sdfg), str(param_dict)))

            # update name pattern
            cumulated_pattern_name += '|' if cumulated_pattern_name != '' else ''
            cumulated_pattern_name += pattern_match.print_match(sdfg)[0:4]

            # Set each parameter of the parameter dictionary separately
            for k, v in param_dict.items():
                setattr(pattern_match, k, v)

            pattern_match.apply(sdfg)
            self.applied_patterns.add(type(pattern_match))


            if SAVE_INTERMEDIATE:
                filename = 'after_%d_%s_b4lprop' % (
                    pattern_counter + 1, type(pattern_match).__name__)
                self.sdfg.save(os.path.join('_dacegraphs', filename + '.sdfg'))

            if not pattern_match.annotates_memlets():
                propagation.propagate_memlets_sdfg(self.sdfg)

            self.roofline.evaluate(cumulated_pattern_name, self.sdfg)


            pattern_counter += 1
            if SAVE_INTERMEDIATE:
                filename = 'after_%d_%s' % (pattern_counter,
                                                type(pattern_match).__name__)
                self.sdfg.save(
                    os.path.join('_dacegraphs', filename + '.sdfg'))

                print("Saving Graph")


            if VISUALIZE_SDFV:
                view(self.sdfg, self.roofline)

        return self.sdfg


    def save_tmp(self):
        basepath = os.path.dirname(os.pathrealpath(diode.__file__))
        save_folder = basepath + '/cache/'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_name = save_folder + 'tmp'
        self.roofline.plot(save_path = save_name)

def _parse_cli_input(line):
    """ Parses a command line input, which may include a transformation name
        (optional), its occurrence ID, and its parameters (optional).
        Syntax Examples:
            * 5                  - Chooses the fifth transformation
            * MapReduceFusion$0  - First occurrence of MapReduceFusion
            * 4(array='A')       - Transformation number 4 with one parameter
            * StripMining$1(param='i', tile_size=64) - Strip mining #2 with
                                                       parameters
        :param line: Input line string
        :return: A tuple with (transformation name or None if not given,
                                      occurrence or -1 if not given,
                                      parameter dictionary or {} if not given)
    """
    # First try matching explicit all-inclusive string "A$num(values)"
    match = re.findall(r'(.*)\$(\d+)\((.*)\)', line)
    if len(match) == 1:
        trans_name, occurrence, param_dict = match[0]
    else:
        # Then, try to match "num(values)"
        match = re.findall(r'(\d+)\((.*)\)', line)
        if len(match) == 1:
            trans_name = None
            occurrence, param_dict = match[0]
        else:
            # After that, try to match "A$num"
            match = re.findall(r'(.*)\$(\d+)', line)
            if len(match) == 1:
                trans_name, occurrence = match[0]
                param_dict = {}
            else:
                # Finally, try to match "num"
                match = re.findall(r'(\d+)', line)
                if len(match) == 1:
                    trans_name = None
                    occurrence = match[0]
                    param_dict = {}
                else:
                    return (None, -1, {})

    # Try to parse the results
    try:
        occurrence = int(occurrence)
    except ValueError:
        occurrence = -1
    try:
        if isinstance(param_dict, str):
            param_dict = eval('dict(' + param_dict + ')')
    except:  # Here we have to catch ANY exception since literally anything
        # can happen
        param_dict = {}

    return trans_name, occurrence, param_dict
