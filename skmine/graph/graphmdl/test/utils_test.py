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

    p5 = nx.DiGraph()
    p5.add_node(1, label='x')
    p5.add_node(2, label='y')
    p5.add_edge(1, 2, label='a')
    print(utils.encode(p5, standard_table))

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

    # Test for graph


ng = nx.Graph()
ng.add_nodes_from(range(1, 6))
ng.add_edge(1, 2, label='e')
ng.add_edge(2, 3, label='e')
ng.add_edge(2, 4, label='e')
ng.add_edge(5, 2, label='e')
ng.nodes[1]['label'] = 'A'
ng.nodes[2]['label'] = 'A'
ng.nodes[3]['label'] = 'B'
ng.nodes[4]['label'] = 'B'
ng.nodes[5]['label'] = 'A'

pattern = nx.Graph()
pattern.add_nodes_from(range(1, 3))
pattern.add_edge(1, 2, label='e')
pattern.nodes[1]['label'] = 'A'


def test_get_embeddings():
    ng2 = nx.DiGraph()
    ng2.add_nodes_from(range(1, 6))
    ng2.add_edge(1, 2, label='e')
    ng2.add_edge(2, 3, label='e')
    ng2.add_edge(2, 4, label='e')
    ng2.add_edge(5, 2, label='e')
    ng2.nodes[1]['label'] = 'A'
    ng2.nodes[2]['label'] = 'A'
    ng2.nodes[3]['label'] = 'B'
    ng2.nodes[4]['label'] = 'B'
    ng2.nodes[5]['label'] = 'A'

    ngp = nx.DiGraph()
    ngp.add_nodes_from(range(1, 3))
    ngp.add_edge(1, 2, label='e')
    ngp.nodes[1]['label'] = 'A'
    # first test for digraph
    assert len(utils.get_embeddings(ngp, ng2)) == 4
    assert utils.get_embeddings(ngp, ng2)[0][1] == 1
    assert utils.get_embeddings(ngp, ng2)[0][2] == 2

    assert len(utils.get_embeddings(pattern, ng)) == 6
    assert utils.get_embeddings(pattern, ng)[5][5] == 1
    assert utils.get_embeddings(pattern, ng)[5][2] == 2

    test1 = nx.Graph()
    ptest1 = nx.Graph()
    test1.add_node(1)
    test1.nodes[1]['label'] = 'w', 'x', 'y'
    ptest1.add_node(1)
    ptest1.nodes[1]['label'] = 'w', 'x'

    assert len(utils.get_embeddings(ptest1, test1)) != 0
    assert len(utils.get_embeddings(test1, ptest1)) == 0

    test2 = nx.Graph()
    test2.add_node(1)
    ptest2 = nx.Graph()
    ptest2.add_node(1, label='x')
    assert len(utils.get_embeddings(ptest2, test2)) == 0
    assert len(utils.get_embeddings(test2, ptest2)) != 0


def test_is_vertex_singleton():
    g1 = nx.DiGraph()
    g1.add_node(1, label='a')
    assert utils.is_vertex_singleton(g1) is True

    g2 = nx.Graph()
    g2.add_node(1, label='a')
    assert utils.is_vertex_singleton(g2) is True

    g3 = nx.Graph()
    g3.add_nodes_from(range(1, 3))
    assert utils.is_vertex_singleton(g3) is False

    g4 = nx.DiGraph()
    g4.add_node(1)
    g4.nodes[1]['label'] = 'a', 'b'
    assert utils.is_vertex_singleton(g4) is False


def test_is_edge_singleton():
    g1 = nx.DiGraph()
    g1.add_node(1, label='a')
    assert utils.is_edge_singleton(g1) is False

    g2 = nx.Graph()
    g2.add_node(1)
    g2.add_node(2, label='a')
    assert utils.is_edge_singleton(g2) is False

    g3 = nx.DiGraph()
    g3.add_node(1)
    g3.add_node(2)
    g3.add_edge(1, 2)
    assert utils.is_edge_singleton(g3) is False

    g4 = nx.Graph()
    g4.add_node(1)
    g4.add_node(2)
    g4.add_edge(1, 2)
    g4[1][2]['label'] = 'a', 'b'
    print(bool(utils.count_edge_label(g4)))
    assert utils.is_edge_singleton(g4) is False

    g5 = nx.DiGraph()
    g5.add_node(1)
    g5.add_node(2)
    g5.add_edge(1, 2)
    g5[1][2]['label'] = 'a'
    assert utils.is_edge_singleton(g5) is True


def test_get_support():
    pattern1 = nx.DiGraph()
    pattern1.add_nodes_from(range(1, 4))
    pattern1.nodes[1]['label'] = 'x'
    pattern1.nodes[2]['label'] = 'y'
    pattern1.nodes[3]['label'] = 'z'
    pattern1.add_edge(1, 2, label='a')
    pattern1.add_edge(2, 3, label='b')

    assert utils.get_support(utils.get_embeddings(pattern1, graph)) == 1
    assert utils.get_support(utils.get_embeddings(pattern, ng)) == 2


def test_get_label_index():
    values = ('a', 'b')
    assert utils.get_label_index('a', values) == 0

    with pytest.raises(ValueError):
        utils.get_label_index('c', values)


def test_get_node_label():
    test = nx.Graph()
    test.add_node(1)
    test.nodes[1]['label'] = 'a', 'b'

    with pytest.raises(ValueError):
        utils.get_node_label(1, 6, test)
        utils.get_node_label(2, 0, test)

    assert utils.get_node_label(1, 0, test) == 'a'


def test_get_edge_label():
    test = nx.Graph()
    test.add_nodes_from(range(1, 3))
    test.add_edge(1, 2, label='a')
    assert utils.get_edge_label(1, 2, test) == 'a'

    test.add_edge(2, 3)
    with pytest.raises(ValueError):
        utils.get_edge_label(2, 3, test)
        utils.get_edge_label(2, 4, test)


def test_is_without_edge():
    test = nx.Graph()
    test.add_node(1)
    test.add_node(2)
    assert utils.is_without_edge(test) is True

    test.add_edge(1, 2)
    assert utils.is_without_edge(test) is False


def test_get_edge_in_embedding():
    p7 = nx.DiGraph()
    p7.add_node(1, label='x')
    p7.add_node(2)
    p7.add_node(3)
    p7.add_edge(1, 2, label='a')
    p7.add_edge(1, 3, label='a')

    embeddings = utils.get_embeddings(p7, graph)
    assert list(utils.get_edge_in_embedding(embeddings[0], p7))[0][0] == 1
    assert list(utils.get_edge_in_embedding(embeddings[0], p7))[0][1] == 2
    assert list(utils.get_edge_in_embedding(embeddings[0], p7))[1][0] == 1
    assert list(utils.get_edge_in_embedding(embeddings[0], p7))[1][1] == 3


def test_get_key_from_value():
    p7 = nx.DiGraph()
    p7.add_node(1, label='x')
    p7.add_node(2)
    p7.add_node(3)
    p7.add_edge(1, 2, label='a')
    p7.add_edge(1, 3, label='a')

    embeddings = utils.get_embeddings(p7, graph)
    assert utils.get_key_from_value(embeddings[0], 1) == 6
    assert utils.get_key_from_value(embeddings[0], 2) == 8
    assert utils.get_key_from_value(embeddings[0], 3) == 1
