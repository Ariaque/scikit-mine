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
        self.patterns = []

    def fit(self, data=None):
        iterations = 10
        i = 0
        if data is None and self._data is None:
            raise ValueError("You should give a graph")
        else:
            self._data = data
            self._label_codes = LabelCodes(self._data)  # label codes creation

            # CT0 creation
            self._code_table = CodeTable(self._label_codes, self._data)
            # CT0 cover
            self._code_table.cover()
            self._rewritten_graph = self._code_table.rewritten_graph()
            self.description_length = self._code_table.compute_description_length()
            print("\n initial CT ", self._code_table)
            print("\n initial LD ", self.description_length)
            while i < iterations:
                candidates = utils.get_candidates(self._rewritten_graph, self._code_table)

                if len(candidates) == 0:  # if it doesn't have candidates stop the algorithm
                    break

                for candidate in candidates:
                    # Add a candidate to a ct, cover and compute description length
                    temp_ct = self._code_table
                    row = CodeTableRow(candidate.final_pattern)
                    temp_ct.add_row(row)
                    temp_ct.cover()
                    temp_code_length = temp_ct.compute_description_length()
                    # if candidates.index(candidate) != len(candidates) - 1:
                    # if the new ct is better than the old, break and generate new candidates
                    # with the new ct
                    if temp_code_length < self.description_length:
                        self._code_table = temp_ct
                        self._rewritten_graph = temp_ct.rewritten_graph()
                        self.description_length = temp_code_length
                        print("\n New ct", self._code_table)
                        print("\n New LD ", self.description_length)
                        break
                    else:
                        # if the candidate not improve the result, remove it to the code table
                        temp_ct.remove_row(row)
                    """ else:
                        # if none candidate improve the current code table,
                        # return the code table and stop the loop
                        if temp_code_length > self.description_length:
                            return self
                        else:
                            self._code_table = temp_ct
                            self._rewritten_graph = temp_ct.rewritten_graph()
                            self.description_length = temp_code_length
                            break """

                i += 1

            # get all ct patterns
            for r in self._code_table.rows():
                if r.code_length() != 0:
                    self.patterns.append(r.pattern())

        return self

    def summary(self):
        print(self._code_table)
        print("description length : ", self.description_length)
        print("patterns_number : ", len(self.patterns))
