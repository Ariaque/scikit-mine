from skmine.graph.graphmdl.label_codes import LabelCodes
from skmine.graph.graphmdl.code_table import CodeTable
from skmine.graph.graphmdl.code_table_row import CodeTableRow
from skmine.graph.graphmdl import utils


class GraphMDl:
    def __init__(self, data=None):
        self._data = data
        self._label_codes = None
        self._code_table = None
        self._rewritten_graph = None
        self.description_length = 0.0
        self._patterns = []

    def _init_graph_mdl(self):
        """ Initialize the algorithm elements """
        self._label_codes = LabelCodes(self._data)  # label codes creation
        # CT0 creation
        self._code_table = CodeTable(self._label_codes, self._data)
        # CT0 cover
        self._code_table.cover()
        self._rewritten_graph = self._code_table.rewritten_graph()
        self.description_length = self._code_table.compute_description_length()
        print("\n initial CT ", self._code_table)
        print("Initial DL ", self.description_length)

    def fit(self, data=None):
        # iterations = 10
        #  i = 0
        if data is None and self._data is None:
            raise ValueError("You should give a graph")
        else:
            self._data = data
            self._init_graph_mdl()
            # while i < iterations:
            self._graph_mdl()
            #  i += 1
            return self

    def _graph_mdl(self):
        """ Non-anytime graphmdl+ algorithm"""
        candidates = utils.get_candidates(self._rewritten_graph, self._code_table)
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
                # Add a candidate to a ct, cover and compute description length
                temp_ct = self._code_table
                row = CodeTableRow(candidate.final_pattern)
                temp_ct.add_row(row)
                temp_ct.cover()
                temp_code_length = temp_ct.compute_description_length()
                # if the new ct is better than the old, break and generate new candidates
                # with the new ct
                if temp_code_length < self.description_length:
                    # self._code_table = temp_ct
                    self._rewritten_graph = temp_ct.rewritten_graph()
                    self.description_length = temp_code_length
                    print("\n New CT", self._code_table)
                    print("New DL ", self.description_length)
                    self._graph_mdl()
                elif temp_code_length > self.description_length and candidates.index(candidate) == len(
                        candidates) - 1:
                    # if the last candidates doesn't improve the code table,
                    # then stop the algorithm
                    print("\n None best code table found")
                    temp_ct.remove_row(row)
                    return self
                else:
                    # if the candidate not improve the result, remove it to the code table
                    temp_ct.remove_row(row)
        else:
            return self

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

        return self.patterns
