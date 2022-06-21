from networkx import Graph
import skmine.graph.graphmdl.utils as utils


class StandardTable:
    """
      StandardTable : Initial code table for the GraphMDL algorithm
      It's different of Krimp standard table
      Here standard table is only a storage for label frequency in the initial data graph
    """

    # Authors : Arnauld Djedjemel
    #           Francesco Bariatti

    def __init__(self, graph: Graph):
        """
             Parameters
             -----------
               graph:Graph
                 the treated graph
            """
        self._total_label = utils.get_total_label(graph)
        self._vertexST = dict([(u, utils.log2(v, self._total_label))
                               for u, v in utils.count_vertex_label(graph).items()])  # Vertex label code length
        self._edgeST = dict([(u, utils.log2(v, self._total_label))
                             for u, v in utils.count_edge_label(graph).items()])    # edge label code length

    def display_vertex_st(self):
        """ Display vertexSt content

            Returns
            --------
            str
       """
        msg = ""
        for i, j in self._vertexST.items():
            msg += "{}->{}\n".format(i, j)
        return msg

    def display_edge_st(self):

        """ Display vertexSt content

                   Returns
                   --------
                   str
        """
        msg = ""
        for i, j in self._edgeST.items():
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

    def vertex_st(self):
        """
        Provide all vertex and their values from the standard table
        Returns
        -------
        dict
        """
        return self._vertexST

    def edges_st(self):
        """
        Provide all edges and their values from the standard table
        Returns
        -------

        """
        return self._edgeST

    def encode(self, pattern: Graph):
        """ Compute description length of a pattern with this standard table
        Parameters
        ----------
        pattern
        Returns
        ---------
        float
        """
        return utils.encode(pattern, self)

    def encode_singleton_vertex(self, vertex_singleton_label):
        """ Compute description length of a vertex singleton pattern with this standard table
        Parameters
        ----------
        vertex_singleton_label
        Returns
        ---------
        float
        """
        return utils.encode_singleton(self, 1, vertex_singleton_label)

    def encode_singleton_edge(self, edge_singleton_label):
        """ Compute description length of an edge singleton pattern with this standard table
        Parameters
        ----------
        edge_singleton_label
        Returns
        ---------
        float
        """
        return utils.encode_singleton(self, 2, edge_singleton_label)

    def __str__(self) -> str:
        return "Edge label\n-----------------\n" + self.display_edge_st() + "\nVertex label\n----------------------\n" + self.display_vertex_st()
