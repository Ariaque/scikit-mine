import copy
from abc import ABC

from skmine.base import BaseMiner
from skmine.graph.graphmdl.label_codes import LabelCodes
from skmine.graph.graphmdl.code_table import CodeTable
from skmine.graph.graphmdl.code_table_row import CodeTableRow
from skmine.graph.graphmdl import utils


def _order_pruning_rows(row):
    return row.pattern_usage()


class GraphMDl(BaseMiner):
    def __init__(self, data=None):
        self._data = data
        self._label_codes = None
        self._code_table = None
        self._rewritten_graph = None
        self.description_length = 0.0
        self._patterns = []
        self._already_test = []
        self._pruning_rows = []
        self._old_usage = dict()

    def _init_graph_mdl(self):
        """ Initialize the algorithm elements """
        self._already_test = []
        self._label_codes = LabelCodes(self._data)  # label codes creation
        # CT0 creation
        self._code_table = CodeTable(self._label_codes, self._data)
        # CT0 cover
        self._code_table.cover()
        self._rewritten_graph = self._code_table.rewritten_graph()
        self.description_length = self._code_table.compute_description_length()
        print("\n initial CT ", self._code_table)
        print("Initial DL ", self.description_length)

    def fit(self, D, y=None):
        # iterations = 10
        #  i = 0
        if D is None and self._data is None:
            raise ValueError("You should give a graph")
        else:
            self._data = D
            self._init_graph_mdl()
            # while i < iterations:
            self._graph_mdl()
            #  i += 1
            return self

    def _graph_mdl(self):
        """ Non-anytime graphmdl+ algorithm"""
        candidates = utils.get_candidates(self._rewritten_graph, self._code_table)
        candidates.sort(reverse=True, key=self._order_candidates)  # sort the candidates list
        self._search_best_code_table(candidates)

    def _search_best_code_table(self, candidates):
        """ search if one candidate improve the actual code table
        Parameters
        ----------
        candidates
        Returns
        --------
        GraphMDl
        """
        if len(candidates) != 0:
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
                        # self._code_table = temp_ct
                        self._rewritten_graph = temp_ct.rewritten_graph()
                        self.description_length = temp_code_length
                        self._code_table = temp_ct
                        # print("\n New CT", self._code_table)
                        print("New DL ", self.description_length)
                        print("new pattern added: ", utils.display_graph(row.pattern()))
                        self._compute_pruning_candidates()
                        self._pruning()
                        self._graph_mdl()
                    else:
                        # if the candidate not improve the result, remove it to the code table
                        temp_ct.remove_row(row)
            print("\n None best code table found")
            return self
        else:
            return self

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

    def summary(self):
        print(self._code_table)
        print("description length : ", self.description_length)
        print("patterns_number : ", len(self._patterns))

    def patterns(self):
        """ Provide found patterns
        Returns
        -------
        list
        """
        for r in self._code_table.rows():
            if r.code_length() != 0:
                self._patterns.append(r.pattern())

        return self._patterns

    def discover(self, *args, **kwargs):
        print(self._code_table)
        print("description length : ", self.description_length)
        print("patterns_number : ", len(self._patterns))
