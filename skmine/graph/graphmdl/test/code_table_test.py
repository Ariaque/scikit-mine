import networkx as nx
import pytest
from ..code_table import *
from ..code_table_row import CodeTableRow
from ..label_codes import LabelCodes


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


def init_gtest():
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
    label_codes = LabelCodes(gtest)
    code_table = CodeTable(label_codes, gtest)
    rewritten_graph = nx.DiGraph()
    return gtest, pattern, embeddings, label_codes, code_table, rewritten_graph


def test_is_embedding_marked():
    gtest = init_gtest()[0]
    pattern = init_gtest()[1]
    embeddings = init_gtest()[2]
    assert is_embedding_marked(embeddings[0], pattern, gtest, 1) is False

    gtest[1][2]['cover_mark'] = {'e': 1}
    assert is_embedding_marked(embeddings[0], pattern, gtest, 1) is True

    mark_edge(1, 2, gtest, 2, 'e')
    assert is_embedding_marked(embeddings[0], pattern, gtest, 1) is False
    assert is_embedding_marked(embeddings[0], pattern, gtest, 2) is True


def test_mark_embedding():
    gtest = init_gtest()[0]
    pattern = init_gtest()[1]
    embeddings = init_gtest()[2]
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
    mark_embedding(embed[1], test1, ptest1, 1)
    port_usage = dict()
    ports = search_port(test1, embed[0], 1, ptest1, port_usage)
    assert ports[0][0] == 40
    assert ports[0][1] == 1
    assert (1 in port_usage.keys()) is True
    search_port(test1, embed[1], 1, ptest1, port_usage)
    assert port_usage[1] != 1


def test_is_node_edges_marked():
    gtest = init_gtest()[0]
    pattern = init_gtest()[1]
    embeddings = init_gtest()[2]
    mark_embedding(embeddings[0], gtest, pattern, 1)

    assert is_node_edges_marked(gtest, 1, pattern, 1) is True
    assert is_node_edges_marked(gtest, 2, pattern, 1) is False


def test_is_node_all_labels_marked():
    test = nx.Graph()
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
    gtest = init_gtest()[0]
    pattern = init_gtest()[1]
    embeddings = init_gtest()[2]
    rewritten_graph = init_gtest()[5]

    ptest = nx.Graph()
    ptest.add_node(1, label='x')

    test = nx.Graph()
    test.add_node(1, label='x')
    test.add_node(3, label='y')
    test.add_node(2)
    test.nodes()[2]['label'] = 'x', 'y'
    test.add_edge(1, 2, label='e')
    test.add_edge(2, 3, label='e')

    row_test = CodeTableRow(ptest)
    row_test.set_embeddings(utils.get_embeddings(ptest, test))
    # with pytest.raises(ValueError):
    row_cover(row_test, test, 1, rewritten_graph, 0)
    assert is_node_marked(1, test, 1, ptest.nodes(data=True)[1]['label']) is True
    assert is_node_marked(2, test, 1, ptest.nodes(data=True)[1]['label']) is True

    row = CodeTableRow(pattern)
    row.set_embeddings(embeddings)

    row_cover(row, gtest, 1, rewritten_graph, 1)

    for edge in gtest.edges(data=True):
        assert is_edge_marked(edge[0], edge[1], gtest, 1, gtest[edge[0]][edge[1]]['label']) is True

    for node in gtest.nodes(data=True):
        if node[1]['label'] == 'A':
            assert is_node_marked(node[0], gtest, 1, node[1]['label']) is True
        else:
            assert is_node_marked(node[0], gtest, 1, node[1]['label']) is False


def test_singleton_cover():
    rewritten_graph = init_gtest()[5]

    test = nx.Graph()
    test.add_node(1, label='x')
    test.add_node(3, label='y')
    test.add_node(2)
    test.nodes()[2]['label'] = 'x', 'y'
    test.add_edge(1, 2, label='e')
    test.add_edge(2, 3, label='e')
    res = singleton_cover(test, 1, rewritten_graph)

    assert res[0]['x'] == 2
    assert res[0]['y'] == 2
    assert res[1]['e'] == 2


def test_rows():
    code_table = init_gtest()[4]
    assert len(code_table.rows()) == 0


def test_add_row():
    gtest = init_gtest()[0]
    pattern = init_gtest()[1]
    embeddings = init_gtest()[2]
    rewritten_graph = init_gtest()[5]
    code_table = init_gtest()[4]

    row1 = CodeTableRow(pattern)

    pattern2 = nx.DiGraph()
    pattern2.add_node(1, label='B')

    row2 = CodeTableRow(pattern2)

    code_table.add_row(row1)
    code_table.add_row(row2)

    assert len(code_table.rows()[0].pattern().nodes()) == 2
    assert len(code_table.rows()[1].pattern().nodes()) == 1


def test_create_rewrite_edge():
    rewritten_graph = init_gtest()[5]

    rewritten_graph.add_node(1, label='40')
    rewritten_graph.add_node(2, label='P1')

    create_rewrite_edge(rewritten_graph, 2, 40, pattern_port=1)
    assert ((2, 1) in list(rewritten_graph.edges(2))) is True
    assert rewritten_graph[2][1]['label'] == 'v1'

    create_rewrite_edge(rewritten_graph, 2, 41, pattern_port=2)
    assert (3 in rewritten_graph.nodes()) is True
    assert rewritten_graph.nodes[3]['label'] == '41'
    assert ((2, 3) in list(rewritten_graph.edges(2))) is True
    assert rewritten_graph[2][3]['label'] == 'v2'


def test_create_vertex_singleton_node():
    rewritten_graph = init_gtest()[5]
    rewritten_graph.add_node(1, label='40')
    create_vertex_singleton_node(rewritten_graph, 'x', 40)
    assert (2 in rewritten_graph.nodes()) is True
    assert ('is_Pattern' in rewritten_graph.nodes(data=True)[2]) is True
    assert rewritten_graph.nodes[2]['is_Pattern'] is True
    assert ('is_singleton' in rewritten_graph.nodes(data=True)[2]) is True
    assert rewritten_graph.nodes[2]['is_singleton'] is True
    assert rewritten_graph.nodes[2]['label'] == 'x'
    assert rewritten_graph[2][1]['label'] == 'v1'


def test_create_edge_singleton_node():
    rewritten_graph = init_gtest()[5]
    rewritten_graph.add_node(1, label='40')
    create_edge_singleton_node(rewritten_graph, 'a', 40, 41)
    assert rewritten_graph.nodes[2]['is_Pattern'] is True
    assert rewritten_graph.nodes[2]['is_singleton'] is True
    assert rewritten_graph.nodes[3]['label'] == '41'
    assert rewritten_graph.nodes[2]['label'] == 'a'
    assert rewritten_graph[2][1]['label'] == 'v1'
    assert rewritten_graph[2][3]['label'] == 'v2'


def test_create_pattern_node():
    rewritten_graph = init_gtest()[5]
    rewritten_graph.add_node(1, label='40')
    create_pattern_node(rewritten_graph, 1, [(40, 1)])
    assert rewritten_graph.nodes[2]['is_Pattern'] is True
    assert ('is_singleton' in rewritten_graph.nodes(data=True)[2]) is False
    assert rewritten_graph.nodes[2]['label'] == 'P1'
    assert rewritten_graph[2][1]['label'] == 'v1'


def init_graph():
    res = dict()
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

    res['graph'] = graph

    p1 = nx.DiGraph()
    p1.add_nodes_from(range(1, 4))
    p1.add_edge(1, 2, label='a')
    p1.add_edge(2, 3, label='b')
    p1.nodes[1]['label'] = 'x'
    p1.nodes[2]['label'] = 'y'
    p1.nodes[3]['label'] = 'z'
    row1 = CodeTableRow(p1)
    res['p1'] = p1
    res['row1'] = row1

    p2 = nx.DiGraph()
    p2.add_nodes_from(range(1, 3))
    p2.nodes[1]['label'] = 'x'
    p2.nodes[2]['label'] = 'x'
    p2.add_edge(1, 2, label='a')
    p2.add_edge(2, 1, label='a')
    row2 = CodeTableRow(p2)

    res['p2'] = p2
    res['row2'] = row2

    p3 = nx.DiGraph()
    p3.add_node(1, label='x')
    p3.add_node(2)
    p3.add_edge(1, 2, label='a')
    row3 = CodeTableRow(p3)
    res['p3'] = p3
    res['row3'] = row3

    p4 = nx.DiGraph()
    p4.add_node(1, label='z')
    p4.add_node(2)
    p4.add_edge(2, 1, label='b')
    row4 = CodeTableRow(p4)
    res['p4'] = p4
    res['row4'] = row4

    p5 = nx.DiGraph()
    p5.add_node(1, label='x')
    p5.add_node(2, label='y')
    p5.add_edge(1, 2, label='a')
    row5 = CodeTableRow(p5)
    res['p5'] = p5
    res['row5'] = row5

    p6 = nx.DiGraph()
    p6.add_node(1, label='y')
    p6.add_node(2, label='z')
    p6.add_edge(1, 2, label='b')
    row6 = CodeTableRow(p6)
    res['p6'] = p6
    res['row6'] = row6

    p7 = nx.DiGraph()
    p7.add_node(1, label='y')
    p7.add_node(2)
    p7.add_node(3)
    p7.add_edge(1, 2, label='a')
    p7.add_edge(1, 3, label='a')
    row7 = CodeTableRow(p7)
    res['p7'] = p7
    res['row7'] = row7
    lc = LabelCodes(graph)
    res['lc'] = lc
    ct = CodeTable(lc, graph)
    res['ct'] = ct
    return res


def test_cover():
    res = init_graph()
    ct = res['ct']
    # ct.add_row(row7)
    ct.add_row(res['row3'])
    ct.add_row(res['row4'])
    ct.add_row(res['row5'])
    ct.add_row(res['row6'])
    ct.cover()

    print(ct)


def test_compute_ct_description_length():
    res = init_graph()
    ct = res['ct']
    with pytest.raises(ValueError):
        ct.compute_ct_description_length()

    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 112.75

    ct.add_row(res['row1'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 113.55

    ct.add_row(res['row2'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 103.29

    ct.remove_row(res['row1'])
    ct.remove_row(res['row2'])

    ct.add_row(res['row3'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 102.23

    ct.add_row(res['row4'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 91.02

    ct.add_row(res['row5'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 108.91

    ct.add_row(res['row6'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 118.0


def test_rewritten_graph():
    res = init_graph()
    ct = res['ct']
    ct.add_row(res['row5'])
    p8 = nx.DiGraph()
    p8.add_node(1)
    p8.add_node(2, label='z')
    p8.add_edge(1, 2, label='b')
    row8 = CodeTableRow(p8)
    ct.add_row(row8)
    ct.cover()
    print('\n data_port: ', ct.data_port())
    print('\n port count :', utils.count_port_node(ct.rewritten_graph()))
    # assert utils.count_port_node(ct.rewritten_graph()) == 6
    print('\n pattern infos :', utils.get_pattern_node_infos(ct.rewritten_graph()))
    # assert len(utils.get_pattern_node_infos(ct.rewritten_graph())['P0']) == 5
    print('\n port infos :', utils.get_port_node_infos(ct.rewritten_graph()))
    can = utils.get_candidates(ct.rewritten_graph(), ct)
    print(can)


def test_compute_rewritten_graph_description():
    res = init_graph()
    ct = res['ct']
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 143.51

    ct.add_row(res['row1'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 56.49

    ct.add_row(res['row2'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 38.05

    ct.remove_row(res['row1'])
    ct.remove_row(res['row2'])

    ct.add_row(res['row3'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 108.15

    ct.add_row(res['row4'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 72.84

    ct.add_row(res['row5'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 68.61

    ct.add_row(res['row6'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 68.61


def test_is_ct_edge_singleton():
    res = init_graph()
    ct = res['ct']
    ct.add_row(res['row5'])
    p8 = nx.DiGraph()
    p8.add_node(1)
    p8.add_node(2, label='z')
    p8.add_edge(1, 2, label='b')
    row8 = CodeTableRow(p8)
    ct.add_row(row8)
    ct.cover()
    assert ct.is_ct_edge_singleton('a') is True
    assert ct.is_ct_edge_singleton('w') is False


def test_is_ct_vertex_singleton():
    res = init_graph()
    ct = res['ct']
    ct.add_row(res['row5'])
    p8 = nx.DiGraph()
    p8.add_node(1)
    p8.add_node(2, label='z')
    p8.add_edge(1, 2, label='b')
    row8 = CodeTableRow(p8)
    ct.add_row(row8)
    ct.cover()
    assert ct.is_ct_vertex_singleton('a') is False
    assert ct.is_ct_vertex_singleton('w') is True
