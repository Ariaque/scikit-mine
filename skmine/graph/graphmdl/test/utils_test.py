import networkx as nx
import pytest
import skmine.graph.graphmdl.utils as utils
from skmine.graph.graphmdl.standard_table import StandardTable as ST

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
graph = g
standard_table = ST(graph)


def test_count_edge_label():
    assert len(utils.count_edge_label(graph).items()) == 2
    assert utils.count_edge_label(graph)['a'] == 5
    assert utils.count_edge_label(graph)['b'] == 3


def test_count_vertex_label():
    assert len(utils.count_vertex_label(graph).items()) == 4
    assert utils.count_vertex_label(graph)['x'] == 4
    assert utils.count_vertex_label(graph)['y'] == 1
    assert utils.count_vertex_label(graph)['z'] == 3
    assert utils.count_vertex_label(graph)['w'] == 1


def test_get_total_label():
    assert utils.get_total_label(graph) == 17


def test_binomial():
    with pytest.raises(ValueError):
        utils.binomial(2, 5)

    assert utils.binomial(2, 0) == 1
    assert utils.binomial(4, 3) == 4
    assert utils.binomial(4, 2) == 6


def test_universal_integer_encoding():
    with pytest.raises(ValueError):
        utils.universal_integer_encoding(0)

    assert utils.universal_integer_encoding(1) == 1


def test_universal_integer_encoding_with0():
    with pytest.raises(ValueError):
        utils.universal_integer_encoding_with0(-1)

        assert utils.universal_integer_encoding_with0(1) == 1


def test_encode():
    g1 = nx.DiGraph()
    g1.add_nodes_from(range(1, 3))
    g1.add_edge(1, 2, label='a')
    g1.nodes[1]['label'] = 'x'

    assert pytest.approx(utils.encode(g1, standard_table), rel=1e-01) == 21.44
    assert pytest.approx(utils.encode(graph, standard_table), rel=1e-01) == 111.76


def test_encode_vertex_singleton():
    with pytest.raises(ValueError):
        utils.encode_vertex_singleton(standard_table, '')

    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'x'), rel=1e-01) == 12.67
    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'y'), rel=1e-01) == 14.67
    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'z'), rel=1e-01) == 13.09
    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'w'), rel=1e-01) == 14.67


def test_encode_edge_singleton():
    with pytest.raises(ValueError):
        utils.encode_edge_singleton(standard_table, '')

    assert pytest.approx(utils.encode_edge_singleton(standard_table, 'a'), rel=1e-01) == 14.35
    assert pytest.approx(utils.encode_edge_singleton(standard_table, 'b'), rel=1e-01) == 15.09


def test_encode_singleton():
    with pytest.raises(ValueError):
        utils.encode_singleton(standard_table, 0, 'a')

    assert pytest.approx(utils.encode_singleton(standard_table, 2, 'a'), rel=1e-01) == 14.35
    assert pytest.approx(utils.encode_singleton(standard_table, 1, 'x'), rel=1e-01) == 12.67
