from networkx import Graph
import skmine.graph.graphmdl.utils as utils


class StandardTable:
    """
      StandardTable : Initial code table for the GraphMDL algorithm
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
                               for u, v in utils.count_vertex_label(graph).items()] )
        self._edgeST = dict([(u, utils.log2(v, self._total_label))
                             for u, v in utils.count_edge_label(graph).items()])

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

    def __str__(self) -> str:
        return "Edge label\n-----------------\n" + self.display_edge_st() + "\nVertex label\n----------------------\n" + self.display_vertex_st()
