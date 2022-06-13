import networkx as nx
import pytest
from ..code_table import *
from ..code_table_row import CodeTableRow
from ..standard_table import StandardTable


def test_is_node_marked():
    test = nx.Graph()
    test.add_node(1)
    test.nodes[1]['label'] = 'x', 'w'

    with pytest.raises(ValueError):
        is_node_marked(6, test, 1, 'x')
        is_node_marked(1, test, 1, 'a')

    assert is_node_marked(1, test, 1, 'x') is False
    test.nodes[1]['cover_mark'] = {'x': 1}
    assert is_node_marked(1, test, 1, 'x') is True


def test_is_node_labels_marked():
    test = nx.Graph()
    test.add_node(1)
    test.nodes[1]['label'] = 'x', 'w'

    with pytest.raises(ValueError):
        is_node_labels_marked(6, test, 1, 'x')
        is_node_labels_marked(1, test, 1, 'a')

    test.nodes[1]['cover_mark'] = {'x': 1}
    assert is_node_labels_marked(1, test, 1, ('x', 'w')) is False
    test.nodes[1]['cover_mark'] = {'x': 1, 'w': 1}
    assert is_node_labels_marked(1, test, 1, ('x', 'w')) is True


def test_is_edge_marked():
    test = nx.Graph()
    test.add_node(range(1, 3))
    test.add_edge(1, 2, label='e')

    with pytest.raises(ValueError):
        is_edge_marked(2, 3, test, 1, 'e')
        is_edge_marked(1, 2, test, 1, 'a')

    assert is_edge_marked(1, 2, test, 1, 'e') is False
    test[1][2]['cover_mark'] = {'e': 1}
    assert is_edge_marked(1, 2, test, 1, 'e') is True


def test_mark_node():
    test = nx.Graph()
    test.add_node(1, label='x')

    with pytest.raises(ValueError):
        mark_node(1, test, 1, None)

    mark_node(1, test, 1, 'x')
    assert is_node_marked(1, test, 1, 'x') is True

    test.nodes[1]['label'] = test.nodes[1]['label'], 'w'
    mark_node(1, test, 2, ('x', 'w'))

    assert is_node_marked(1, test, 1, 'x') is False
    assert is_node_marked(1, test, 2, 'x') is True
    assert is_node_marked(1, test, 2, 'w') is True


def test_mark_edge():
    test = nx.Graph()
    test.add_node(1, label='x')
    test.add_node(2)
    test.add_edge(1, 2, label='e')

    with pytest.raises(ValueError):
        mark_edge(1, 2, test, 1, None)
        mark_edge(1, 2, test, 1, 'f')

    mark_edge(1, 2, test, 1, 'e')
    assert is_edge_marked(1, 2, test, 1, 'e') is True

    mark_edge(1, 2, test, 2, 'e')
    assert is_edge_marked(1, 2, test, 1, 'e') is False
    assert is_edge_marked(1, 2, test, 2, 'e') is True


gtest = nx.DiGraph()
gtest.add_nodes_from(range(1, 6))
gtest.add_edge(1, 2, label='e')
gtest.add_edge(2, 3, label='e')
gtest.add_edge(2, 4, label='e')
gtest.add_edge(5, 2, label='e')
gtest.nodes[1]['label'] = 'A'
gtest.nodes[2]['label'] = 'A'
gtest.nodes[3]['label'] = 'B'
gtest.nodes[4]['label'] = 'B'
gtest.nodes[5]['label'] = 'A'

pattern = nx.DiGraph()
pattern.add_nodes_from(range(1, 3))
pattern.add_edge(1, 2, label='e')
pattern.nodes[1]['label'] = 'A'
embeddings = utils.get_embeddings(pattern, gtest)


def test_is_embedding_marked():
    assert is_embedding_marked(embeddings[0], pattern, gtest, 1) is False

    gtest[1][2]['cover_mark'] = {'e': 1}
    assert is_embedding_marked(embeddings[0], pattern, gtest, 1) is True

    mark_edge(1, 2, gtest, 2, 'e')
    assert is_embedding_marked(embeddings[0], pattern, gtest, 1) is False
    assert is_embedding_marked(embeddings[0], pattern, gtest, 2) is True


def test_mark_embedding():
    mark_embedding(embeddings[0], gtest, pattern, 1)

    assert is_edge_marked(1, 2, gtest, 1, 'e') is True
    assert is_node_marked(1, gtest, 1, 'A') is True
    assert is_node_marked(2, gtest, 1, 'A') is False

    mark_embedding(embeddings[0], gtest, pattern, 2)
    assert is_edge_marked(1, 2, gtest, 1, 'e') is False
    assert is_node_marked(1, gtest, 1, 'A') is False

    assert is_edge_marked(1, 2, gtest, 2, 'e') is True
    assert is_node_marked(1, gtest, 2, 'A') is True

    pattern.nodes[2]['label'] = 'A'
    mark_embedding(embeddings[0], gtest, pattern, 1)
    assert is_node_marked(2, gtest, 1, 'A') is True


def test_get_node_label_number():
    test = nx.Graph()
    test.add_node(1)
    # test.nodes[1]['label'] = 'a'

    with pytest.raises(ValueError):
        get_node_label_number(2, test)

    test.nodes[1]['label'] = 'a'
    assert get_node_label_number(1, test) == 1

    test.nodes[1]['label'] = test.nodes[1]['label'], 'b'
    assert get_node_label_number(1, test) == 2

    test.add_node(2)
    assert get_node_label_number(2, test) == 0


def test_search_port():
    test1 = nx.DiGraph()
    test1.add_node(40, label='x')
    test1.add_node(41)
    test1.add_node(42)
    test1.add_node(43, label='y')
    test1.add_edge(40, 41, label='a')
    test1.add_edge(40, 42, label='a')
    test1.add_edge(40, 43, label='a')
    ptest1 = nx.DiGraph()
    ptest1.add_node(1, label='x')
    ptest1.add_node(2)
    ptest1.add_edge(1, 2, label='a')

    embed = utils.get_embeddings(ptest1, test1)
    mark_embedding(embed[0], test1, ptest1, 1)
    assert (1 in search_port(test1, embed[0], 1, dict()).keys()) is True
    mark_embedding(embed[1], test1, ptest1, 1)
    port_usage = search_port(test1, embed[0], 1, dict())
    assert search_port(test1, embed[1], 1, port_usage)[1] == 2


def test_is_node_edges_marked():
    mark_embedding(embeddings[0], gtest, pattern, 1)

    assert is_node_edges_marked(gtest, 1, 1) is True
    assert is_node_edges_marked(gtest, 2, 1) is False


test = nx.Graph()


def test_is_node_all_labels_marked():
    test.add_node(1, label='x')
    test.add_node(3, label='y')
    test.add_node(2)
    test.nodes()[2]['label'] = 'x', 'y'
    test.add_edge(1, 2, label='e')
    test.add_edge(2, 3, label='e')

    mark_node(2, test, 1, 'x')
    mark_node(1, test, 1, 'x')

    assert is_node_all_labels_marked(2, test, 1) is False
    assert is_node_all_labels_marked(1, test, 1) is True


def test_row_cover():
    ptest = nx.Graph()
    ptest.add_node(1, label='x')

    test.add_node(1, label='x')
    test.add_node(3, label='y')
    test.add_node(2)
    test.nodes()[2]['label'] = 'x', 'y'
    test.add_edge(1, 2, label='e')
    test.add_edge(2, 3, label='e')

    row_test = CodeTableRow(ptest)
    row_test.set_embeddings(utils.get_embeddings(ptest, test))
    # with pytest.raises(ValueError):
    row_cover(row_test, test, 1)
    assert is_node_marked(1, test, 1, ptest.nodes(data=True)[1]['label']) is True
    assert is_node_marked(2, test, 1, ptest.nodes(data=True)[1]['label']) is True

    row = CodeTableRow(pattern)
    row.set_embeddings(embeddings)

    row_cover(row, gtest, 1)

    for edge in gtest.edges(data=True):
        assert is_edge_marked(edge[0], edge[1], gtest, 1, gtest[edge[0]][edge[1]]['label']) is True

    for node in gtest.nodes(data=True):
        if node[1]['label'] == 'A':
            assert is_node_marked(node[0], gtest, 1, node[1]['label']) is True
        else:
            assert is_node_marked(node[0], gtest, 1, node[1]['label']) is False


standard_table = StandardTable(gtest)
code_table = CodeTable(standard_table, gtest)


def test_rows():
    assert len(code_table.rows()) == 0


def test_add_row():
    row1 = CodeTableRow(pattern)

    pattern2 = nx.DiGraph()
    pattern2.add_node(1, label='B')

    row2 = CodeTableRow(pattern2)

    code_table.add_row(row1)
    code_table.add_row(row2)

    assert len(code_table.rows()[0].pattern().nodes()) == 2
    assert len(code_table.rows()[1].pattern().nodes()) == 1


graph = nx.DiGraph()
graph.add_nodes_from(range(1, 9))
graph.add_edge(2, 1, label='a')
graph.add_edge(4, 1, label='a')
graph.add_edge(6, 1, label='a')
graph.add_edge(6, 8, label='a')
graph.add_edge(8, 6, label='a')
graph.add_edge(1, 3, label='b')
graph.add_edge(1, 5, label='b')
graph.add_edge(1, 7, label='b')
graph.nodes[1]['label'] = 'y'
graph.nodes[2]['label'] = 'x'
graph.nodes[3]['label'] = 'z'
graph.nodes[4]['label'] = 'x'
graph.nodes[5]['label'] = 'z'
graph.nodes[6]['label'] = 'x'
graph.nodes[7]['label'] = 'z'
graph.nodes[8]['label'] = 'w', 'x'


def test_cover():
    pattern_1 = nx.DiGraph()
    pattern_1.add_node(range(1, 4))
    pattern_1.add_edge(1, 2, label='a')
    pattern_1.add_edge(2, 3, label='b')
    pattern_1.nodes[1]['label'] = 'x'
    pattern_1.nodes[2]['label'] = 'y'
    pattern_1.nodes[3]['label'] = 'z'

    pattern_2 = nx.DiGraph()
    pattern_2.add_node(1, label='w')
