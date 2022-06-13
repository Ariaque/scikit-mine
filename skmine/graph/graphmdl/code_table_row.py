from skmine.graph.graphmdl import utils


class CodeTableRow:
    """
        Object to represent a row of the code table
    """

    def __init__(self, pattern, pattern_code=None, pattern_port_code=None):
        self._pattern = pattern
        self._pattern_code = pattern_code
        self._pattern_port_code = pattern_port_code
        self._embeddings = []
        self._code_length = 0.0
        self._port_code_length = dict()
        self._description_length = 0.0

    def pattern(self):
        """ Provide the row pattern
        Returns
        -------
        object
        """
        return self._pattern

    def pattern_code(self):
        """ Provide the row pattern code
        Returns
        -------
        float
        """
        return self._pattern_code

    def set_pattern_code(self, pattern_code):
        """ Set the row pattern code
        Parameters
        ----------
        pattern_code
        """
        self._pattern_code = pattern_code

    def pattern_port_code(self):
        """ Provide the port code of the row pattern
        Returns
        -------
        dict
        """
        return self._pattern_port_code

    def set_pattern_port_code(self, port_code):
        """ set the port code of the row pattern
        Parameters
        ---------
        port_code
        """
        self._pattern_port_code = port_code

    def set_embeddings(self, embeddings):
        """ Set the pattern row embeddings
        Parameters
        ----------
        embeddings
        """
        self._embeddings = embeddings

    def embeddings(self):
        """ Provide the pattern row embeddings
        Returns
        -------
        list
        """
        return self._embeddings

    def compute_code_length(self, rows_usage_sum):
        """ Compute the code length of the row and its ports
        Parameters
        ---------
        rows_usage_sum : total of usage for the code table rows
        """
        if self._pattern_code == 0:
            self._code_length = 0.0
        else:
            self._code_length = utils.log2(self._pattern_code, rows_usage_sum)

        # compute port usage sum
        port_usage_sum = 0.0
        for k in self._pattern_port_code.keys():
            port_usage_sum = port_usage_sum + self._pattern_port_code[k]

        for p in self._pattern_port_code.keys():
            code = utils.log2(self._pattern_port_code[p], port_usage_sum)
            self._port_code_length[p] = code

    def compute_description_length(self, standard_table):
        if self._pattern_code is None:
            self._description_length = 0.0

        if self._pattern_port_code is None or self._port_code_length.__len__() == 0:
            raise ValueError("Row's codes should be compute")

        self._description_length = self._code_length
        for value in self._port_code_length.values():
            self._description_length += value

        if utils.is_edge_singleton(self._pattern):
            # self._description_length += utils.encode(self._pattern, standard_table)
            pass
        elif utils.is_vertex_singleton(self._pattern):
            # self._description_length += utils.encode_singleton(self._pattern, standard_table)
            pass
        else:
            self._description_length += utils.encode(self._pattern, standard_table)
