import copy
import sys
import time
from abc import ABC

from skmine.base import BaseMiner
from skmine.graph.graphmdl.label_codes import LabelCodes
from skmine.graph.graphmdl.code_table import CodeTable
from skmine.graph.graphmdl.code_table_row import CodeTableRow
from skmine.graph.graphmdl import utils


def _order_pruning_rows(row):
    return row.pattern_usage()


class GraphMDl(BaseMiner):
    def __init__(self, data=None, debug=False, timeout=0):
        self._data = data
        self._label_codes = None
        self._code_table = None
        self._rewritten_graph = None
        self.description_length = 0.0
        self._patterns = set()
        self._already_test = []
        self._pruning_rows = []
        self._debug = debug
        self._old_usage = dict()
        self._timeout = timeout

    def _init_graph_mdl(self):
        """ Initialize the algorithm elements """
        self._already_test = []
        self._label_codes = LabelCodes(self._data)  # label codes creation
        # CT0 creation
        self._code_table = CodeTable(self._label_codes, self._data)
        # CT0 cover
        self._code_table.cover(debug=self._debug)
        self._rewritten_graph = self._code_table.rewritten_graph()
        self.description_length = self._code_table.compute_description_length()
        if self._debug:
            print("\n initial CT ", self._code_table)
            print("Initial DL ", self.description_length)
        print("GraphMDL+ run ...")

    def fit(self, D, timeout=0, debug=False):
        if debug:
            self._debug = debug
        if timeout != 0:
            self._timeout = timeout
        if D is None and self._data is None:
            raise ValueError("You should give a graph")
        else:
            self._data = D
            self._init_graph_mdl()
            """if 'iterations' in kwargs.keys() and 'timeout' in kwargs.keys():
                pass
            elif 'iterations' in kwargs.keys():
                self._init_graph_mdl()
                self._fit(iterations=kwargs['iterations'])"""
            if self._timeout != 0:
                self._anytime_graph_mdl_with_timeout(self._timeout)
                return self
            else:
                self._graph_mdl()
                pass

            # self._graph_mdl_end()

    """def _fit(self, iterations=None):
        if iterations is not None:
            self._anytime_graph_mdl_with_iterations(iterations)
        else:
            self._graph_mdl()"""

    def _graph_mdl(self):
        """ Non-anytime graphmdl+ algorithm"""
        stop = False
        while not stop:
            if self._debug:
                print("Candidate generation and sort start .....")
            b = time.time()
            candidates = utils.generate_candidates(self._rewritten_graph, self._code_table)
            candidates.sort(reverse=True, key=self._order_candidates)  # sort the candidates list
            if self._debug:
                print(f"Candidate generation and sort end .....time ={time.time() - b}")
                print("candidates number", len(candidates))
                print("GraphMDl best Ct search start .....")
            if len(candidates) != 0:
                b = time.time()
                for candidate in candidates:
                    if candidate not in self._already_test:
                        # Add a candidate to a ct, cover and compute description length
                        temp_ct = copy.deepcopy(self._code_table)
                        self._compute_old_usage()
                        row = CodeTableRow(candidate.final_pattern())
                        temp_ct.add_row(row)
                        temp_ct.cover(debug=self._debug)
                        temp_code_length = temp_ct.compute_description_length()
                        self._already_test.append(candidate)
                        # if the new ct is better than the old, break and generate new candidates
                        # with the new ct
                        if temp_code_length < self.description_length:
                            # self._code_table = temp_ct
                            self._rewritten_graph = temp_ct.rewritten_graph()
                            self.description_length = temp_code_length
                            self._code_table = temp_ct
                            # print("\n New CT", self._code_table)
                            if self._debug:
                                print("GraphMDl new ct found .....")
                                print("New DL ", self.description_length)
                                print("new pattern added: ", utils.display_graph(row.pattern()))
                            self._compute_pruning_candidates()
                            self._pruning()
                            if self._debug:
                                print(f"search time = {time.time() - b}")
                            del candidates
                            break
                        elif temp_code_length > self.description_length and candidates.index(candidate) == len(candidates) - 1:
                            if self._debug:
                                print(f"search time = {time.time() - b}")
                                print("None best code table found so stop")
                            del candidates
                            stop = self._graph_mdl_end()
                            break
                        else:
                            # if the candidate not improve the result, remove it to the code table
                            temp_ct.remove_row(row)
                    else:
                        if self._debug:
                            print("Already test")
                # print("\n None best code table found")
                # self.anytime_graph_mdl(1)
            else:
                stop = self._graph_mdl_end()

        return self

    """def _anytime_graph_mdl_with_iterations(self, iterations):
            # Anytime graph mdl with iterations number
        i = 0
        while i <= iterations - 1:
            print("Candidate generation and sort start .....")
            b = time.time()
            candidates = utils.get_candidates(self._rewritten_graph, self._code_table)
            candidates.sort(reverse=True, key=self._order_candidates)  # sort the candidates list
            print(f"Candidate generation and sort end ..... time ={time.time() - b} ")
            print("GraphMDl best Ct search start .....")
            if len(candidates) != 0:
                b = time.time()
                for candidate in candidates:
                    if candidate not in self._already_test:
                        # Add a candidate to a ct, cover and compute description length
                        temp_ct = copy.deepcopy(self._code_table)
                        self._compute_old_usage()
                        row = CodeTableRow(candidate.final_pattern())
                        temp_ct.add_row(row)
                        temp_ct.cover()
                        temp_code_length = temp_ct.compute_description_length()
                        self._already_test.append(candidate)
                        # if the new ct is better than the old, break and generate new candidates
                        # with the new ct
                        if temp_code_length < self.description_length:
                            self._rewritten_graph = temp_ct.rewritten_graph()
                            self.description_length = temp_code_length
                            self._code_table = temp_ct
                            # print("\n New CT", self._code_table)
                            print("New DL ", self.description_length)
                            print("new pattern added: ", utils.display_graph(row.pattern()))
                            self._compute_pruning_candidates()
                            self._pruning()
                            print(f"search time = {time.time() - b}")
                            break
                        elif temp_code_length > self.description_length \
                                and candidates.index(candidate) == len(candidates) - 1:
                            self._graph_mdl_end()
                            i = iterations - 1
                            break
                        else:
                            # if the candidate not improve the result, remove it to the code table
                            temp_ct.remove_row(row)
            else:
                self._graph_mdl_end()
            i += 1
        # self._graph_mdl_end()"""

    def _anytime_graph_mdl_with_timeout(self, timeout):
        """ Anytime graph mdl with timeout"""
        begin = time.time()
        current = 0
        while current < timeout:
            if self._debug:
                print("Candidate generation and sort start .....")
            b = time.time()
            candidates = utils.generate_candidates(self._rewritten_graph, self._code_table)
            candidates.sort(reverse=True, key=self._order_candidates)
            if self._debug:
                print(f"Candidate generation and sort end ..........time ={time.time() - b}")
                print("candidates number", len(candidates))
                print("GraphMDl best Ct search start .....")
            current = time.time() - begin
            if self._stop_by_time(current, timeout):
                break
            if len(candidates) != 0:
                b = time.time()
                for candidate in candidates:
                    if candidate not in self._already_test:
                        # Add a candidate to a ct, cover and compute description length
                        current = time.time() - begin
                        if self._stop_by_time(current, timeout):
                            break
                        temp_ct = copy.deepcopy(self._code_table)
                        self._compute_old_usage()
                        row = CodeTableRow(candidate.final_pattern())
                        temp_ct.add_row(row)
                        temp_ct.cover()
                        temp_code_length = temp_ct.compute_description_length()
                        self._already_test.append(candidate)
                        # if the new ct is better than the old, break and generate new candidates
                        # with the new ct
                        if temp_code_length < self.description_length:
                            # self._code_table = temp_ct
                            self._rewritten_graph = temp_ct.rewritten_graph()
                            self.description_length = temp_code_length
                            self._code_table = temp_ct
                            if self._debug:
                                print("New DL ", self.description_length)
                                print("new pattern added: ", utils.display_graph(row.pattern()))
                                print(f"search time = {time.time() - b}")

                            self._compute_pruning_candidates()
                            self._pruning()
                            break
                        elif temp_code_length > self.description_length and candidates.index(candidate) == len(candidates) - 1:
                            if self._debug:
                                print("None best code table found")
                                print(f"search time = {time.time() - b}")
                            self._graph_mdl_end()
                            current = timeout
                            break
                        else:
                            # if the candidate not improve the result, remove it to the code table
                            temp_ct.remove_row(row)
                    else:
                        if self._debug:
                            print("Already test")
            else:
                self._graph_mdl_end()
                current = timeout
        return self

    def _stop_by_time(self, passed_time, timeout):
        if passed_time >= timeout:
            return self._graph_mdl_end()
        else:
            return False

    def _graph_mdl_end(self):
        """ End of the graph mdl algorithm, cover the code table and stop"""
        print("GraphMDL end .....")
        self._code_table.cover(debug=self._debug)
        self._rewritten_graph = self._code_table.rewritten_graph()
        self.description_length = self._code_table.compute_description_length()
        return True

    def _order_candidates(self, candidate):
        """Provide the candidate elements to order candidates
        Parameters
        ----------
        candidate
        Returns
        -------
        list
        """
        return [candidate.usage, candidate.exclusive_port_number,
                -self._label_codes.encode(candidate.final_pattern())]

    def _compute_old_usage(self):
        """ Store pattern usage """
        self._old_usage = dict()
        for r in self._code_table.rows():
            self._old_usage[r.pattern()] = r.pattern_usage()

    def _compute_pruning_candidates(self):
        """ Find the row where their usage has lowered since the last usage"""
        for r in self._code_table.rows():
            if r.pattern() in self._old_usage.keys():
                if r.pattern_usage() < self._old_usage[r.pattern()]:
                    self._pruning_rows.append(r)

    def _pruning(self):
        """ Make the code table pruning as krimp pruning"""
        if self._debug:
            print("pruning start ...")
        self._compute_old_usage()  # compute old pattern usage
        temp_ct = copy.deepcopy(self._code_table)
        self._pruning_rows.sort(key=_order_pruning_rows)  # sort the pruning rows
        for r in self._pruning_rows:
            temp_ct.remove_row(r)
            temp_ct.cover()
            if temp_ct.compute_description_length() < self.description_length:
                self._code_table = temp_ct
                self._rewritten_graph = temp_ct.rewritten_graph()
                self.description_length = temp_ct.compute_description_length()
                self._compute_pruning_candidates()  # recompute the pruning candidates
                self._pruning()
            else:
                temp_ct.add_row(r)
        if self._debug:
            print("pruning end")

    def patterns(self):
        """ Provide found patterns
        Returns
        -------
        list
        """
        for r in self._code_table.rows():
            if r.code_length() != 0:
                self._patterns.add(r.pattern())

        return self._patterns

    def discover(self, *args, **kwargs):
        print(self._code_table)
        print("final description length : ", self.description_length)
        print("Non singleton patterns found : ")
        for p in self.patterns():
            print('\n', utils.display_graph(p))
        if len(self._code_table.singleton_code_length()) != 0:
            print("Singleton patterns found : ")
            for s in self._code_table.singleton_code_length().keys():
                print("\n", s)
