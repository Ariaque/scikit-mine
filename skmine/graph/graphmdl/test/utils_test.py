import unittest
import networkx as nx
import skmine.graph.graphmdl.utils as utils
from skmine.graph.graphmdl.standard_table import StandardTable as ST


class MyTestCase(unittest.TestCase):

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
        self.standard_table = ST(self.graph)

    def test_count_edge_label(self):
        self.assertEqual(len(utils.count_edge_label(self.graph).items()), 2)
        self.assertEqual(utils.count_edge_label(self.graph)['a'], 5)
        self.assertEqual(utils.count_edge_label(self.graph)['b'], 3)

    def test_count_vertex_label(self):
        self.assertEqual(len(utils.count_vertex_label(self.graph).items()), 4)
        self.assertEqual(utils.count_vertex_label(self.graph)['x'], 4)
        self.assertEqual(utils.count_vertex_label(self.graph)['y'], 1)
        self.assertEqual(utils.count_vertex_label(self.graph)['z'], 3)
        self.assertEqual(utils.count_vertex_label(self.graph)['w'], 1)

    def test_get_total_label(self):
        self.assertEqual(utils.get_total_label(self.graph), 17)

    def test_binomial(self):
        with self.assertRaises(ValueError):
            utils.binomial(2, 5)

        self.assertEqual(utils.binomial(2, 0), 1)
        self.assertEqual(utils.binomial(4, 3), 4)
        self.assertEqual(utils.binomial(4, 2), 6)

    def test_universal_integer_encoding(self):
        with self.assertRaises(ValueError):
            utils.universal_integer_encoding(0)

        self.assertEqual(1, utils.universal_integer_encoding(1))

    def test_universal_integer_encoding_with0(self):
        with self.assertRaises(ValueError):
            utils.universal_integer_encoding_with0(-1)

        self.assertEqual(1, utils.universal_integer_encoding_with0(0))

    def test_get_description_length(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from(range(1, 3))
        g1.add_edge(1, 2, label='a')
        g1.nodes[1]['label'] = 'x'
        # print(utils.binomial(5.0, 2))
        print(utils.get_description_length(g1, self.standard_table))
        print(utils.get_description_length(self.graph, self.standard_table))
        self.assertAlmostEqual(21.44, utils.get_description_length(g1, self.standard_table), 2)
        self.assertAlmostEqual(111.76, utils.get_description_length(self.graph, self.standard_table), 2)


if __name__ == '__main__':
    unittest.main()
