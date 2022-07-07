from networkx import Graph
import skmine.graph.graphmdl.utils as utils


class LabelCodes:
    """
      It is only a storage for label frequency in the initial data graph
    """

    def __init__(self, graph: Graph):
        """
             Parameters
             -----------
               graph:Graph
                 the treated graph
            """
        self._total_label = utils.get_total_label(graph)
        self._vertexLC = dict([(u, utils.log2(v, self._total_label))
                               for u, v in utils.count_vertex_label(graph).items()])  # Vertex label code length
        self._edgeLC = dict([(u, utils.log2(v, self._total_label))
                             for u, v in utils.count_edge_label(graph).items()])    # edge label code length

    def display_vertex_lc(self):
        """ Display vertexLC content

            Returns
            --------
            str
       """
        msg = ""
        for i, j in self._vertexLC.items():
            msg += "{}->{}\n".format(i, j)
        return msg

    def display_edge_lc(self):

        """ Display vertexLC content

                   Returns
                   --------
                   str
        """
        msg = ""
        for i, j in self._edgeLC.items():
            msg += "{}->{}\n".format(i, j)
        return msg

    def total_label(self):
        """
        Provide labels total number
        Returns
        -------
        double
        """
        return self._total_label

    def vertex_lc(self):
        """
        Provide all vertex and their values from the label codes
        Returns
        -------
        dict
        """
        return self._vertexLC

    def edges_lc(self):
        """
        Provide all edges and their values from the label codes
        Returns
        -------

        """
        return self._edgeLC

    def encode(self, pattern: Graph):
        """ Compute description length of a pattern with this label codes
        Parameters
        ----------
        pattern
        Returns
        ---------
        float
        """
        return utils.encode(pattern, self)

    def encode_singleton_vertex(self, vertex_singleton_label):
        """ Compute description length of a vertex singleton pattern with this label codes
        Parameters
        ----------
        vertex_singleton_label
        Returns
        ---------
        float
        """
        return utils.encode_singleton(self, 1, vertex_singleton_label)

    def encode_singleton_edge(self, edge_singleton_label):
        """ Compute description length of an edge singleton pattern with this label codes
        Parameters
        ----------
        edge_singleton_label
        Returns
        ---------
        float
        """
        return utils.encode_singleton(self, 2, edge_singleton_label)

    def __str__(self) -> str:
        return "Edge label\n-----------------\n" + self.display_edge_lc() + "\nVertex label\n----------------------\n" + self.display_vertex_lc()
