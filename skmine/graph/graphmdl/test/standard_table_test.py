import networkx as nx
from ..standard_table import StandardTable
import unittest


class StandardTableTest(unittest.TestCase):

    def setUp(self):
        g = nx.DiGraph()
        g.add_nodes_from(range(1, 9))
        g.add_edge(2, 1, label='a')
        g.add_edge(4, 1, label='a')
        g.add_edge(6, 1, label='a')
        g.add_edge(6, 8, label='a')
        g.add_edge(8, 6, label='a')
        g.add_edge(1, 3, label='b')
        g.add_edge(1, 5, label='b')
        g.add_edge(1, 7, label='b')
        g.nodes[1]['label'] = 'y'
        g.nodes[2]['label'] = 'x'
        g.nodes[3]['label'] = 'z'
        g.nodes[4]['label'] = 'x'
        g.nodes[5]['label'] = 'z'
        g.nodes[6]['label'] = 'x'
        g.nodes[7]['label'] = 'z'
        g.nodes[8]['label'] = 'w', 'x'

        self.graph = g
        self.ST = StandardTable(self.graph)

    def test_total_label(self):
        self.assertEqual(self.ST.total_label(), 17.0)

    def test_vertex_st(self):
        self.assertEqual(self.ST.vertex_st()['y'], 4.09)
        self.assertEqual(self.ST.vertex_st()['x'], 2.09)
        self.assertEqual(self.ST.vertex_st()['z'], 2.50)
        self.assertEqual(self.ST.vertex_st()['w'], 4.09)

    def test_edges_st(self):
        self.assertEqual(self.ST.edges_st()['a'], 1.77)
        self.assertEqual(self.ST.edges_st()['b'], 2.50)
