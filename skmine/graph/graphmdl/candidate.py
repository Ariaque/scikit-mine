from skmine.graph.graphmdl import utils


class Candidate:
    """
        It represents candidate whose format is <P1,P2,{(v1,v1)..}> in the algorithm,
        here, P1 and P2 are patterns and the last parameters, a port list

        Parameters
        -----------
        first_pattern_label : str
            It's the candidate first pattern name
        second_pattern_label : str
            It's the candidate second pattern name
        port: list
            It's the candidate port list
    """

    def __init__(self, first_pattern_label, second_pattern_label, port):
        self.first_pattern_label = first_pattern_label
        self.second_pattern_label = second_pattern_label
        self.first_pattern = None
        self.second_pattern = None
        self.port = port  # list of the candidate port association
        self.data_port = set()  # candidate port number in the rewritten graph
        self.usage = 0  # estimated usage
        self.exclusive_port_number = 0
        self._final_pattern = None  # merge pattern
        self.code_length = 0.0  # merge pattern description length

    def set_usage(self, usage):
        """ Set candidate estimated usage
        Parameters
        ----------
        usage
        """
        self.usage = usage

    def final_pattern(self):
        """ Provide the candidate merge pattern
        Returns
        --------
        object
        """
        if self._final_pattern is None:
            self._final_pattern = utils.merge_candidate(self)
            return self._final_pattern
        else:
            return self._final_pattern

    def _is_ports_equals(self, ports):
        """ Check if the candidate ports list are similar to a given ports list
        Parameters
        ----------
        ports
        Returns
        -------
        bool"""
        if len(ports) != len(self.port):
            return False
        else:
            return not (False in [p in ports for p in self.port])

    def compute_description_length(self, label_codes):
        """ Compute description length from the label codes to the candidate merge pattern
        Parameters
        ----------
        label_codes
        """
        if self.final_pattern is not None and label_codes is not None:
            if utils.is_edge_singleton(self.final_pattern):
                label = self.final_pattern[1][2]['label']
                self.code_length = label_codes.encode_singleton_edge(label)
            elif utils.is_vertex_singleton(self.final_pattern):
                label = self.final_pattern.nodes[1]['label']
                self.code_length = label_codes.encode_singleton_vertex(label)
            else:
                self.code_length = label_codes.encode(self.final_pattern)
        else:
            raise ValueError("You should create the final pattern and set the label codes before computing")

    def inverse(self):
        """ Provide the candidate inverse
        Returns
        -------
        Candidate
        """
        ports = []
        for p in self.port:
            ports.append((p[1], p[0]))
        c = self.first_pattern_label
        self.first_pattern_label = self.second_pattern_label
        self.second_pattern_label = c
        self.port = ports
        del c

        return self

    def __str__(self) -> str:
        return "<{},{},{}>".format(self.first_pattern_label, self.second_pattern_label, self.port)

    def __eq__(self, o: object) -> bool:
        return o.first_pattern_label == self.first_pattern_label \
               and o.second_pattern_label == self.second_pattern_label \
               and self._is_ports_equals(o.port)
