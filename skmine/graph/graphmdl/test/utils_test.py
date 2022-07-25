import networkx as nx
import pytest
import skmine.graph.graphmdl.utils as utils
from skmine.graph.graphmdl.label_codes import LabelCodes as LC
from skmine.graph.graphmdl.code_table import CodeTable
from skmine.graph.graphmdl.code_table_row import CodeTableRow
from skmine.graph.graphmdl.candidate import Candidate


def init_graph():
    res = dict()
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
    label_codes = LC(g)
    res['graph'] = g
    res['st'] = label_codes
    return res


def test_count_edge_label():
    graph = init_graph()['graph']
    assert len(utils.count_edge_label(graph).items()) == 2
    assert utils.count_edge_label(graph)['a'] == 5
    assert utils.count_edge_label(graph)['b'] == 3


def test_count_vertex_label():
    graph = init_graph()['graph']
    assert len(utils.count_vertex_label(graph).items()) == 4
    assert utils.count_vertex_label(graph)['x'] == 4
    assert utils.count_vertex_label(graph)['y'] == 1
    assert utils.count_vertex_label(graph)['z'] == 3
    assert utils.count_vertex_label(graph)['w'] == 1


def test_get_total_label():
    graph = init_graph()['graph']
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
    standard_table = init_graph()['st']
    graph = init_graph()['graph']
    g1 = nx.DiGraph()
    g1.add_nodes_from(range(1, 3))
    g1.add_edge(1, 2, label='a')
    g1.nodes[1]['label'] = 'x'

    p5 = nx.DiGraph()
    p5.add_node(1, label='x')
    p5.add_node(2, label='y')
    p5.add_edge(1, 2, label='a')
    # print(utils.encode(p5, standard_table))

    assert pytest.approx(utils.encode(g1, standard_table), rel=1e-01) == 21.44
    assert pytest.approx(utils.encode(graph, standard_table), rel=1e-01) == 111.76


def test_encode_vertex_singleton():
    standard_table = init_graph()['st']
    with pytest.raises(ValueError):
        utils.encode_vertex_singleton(standard_table, '')

    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'x'), rel=1e-01) == 12.67
    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'y'), rel=1e-01) == 14.67
    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'z'), rel=1e-01) == 13.09
    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'w'), rel=1e-01) == 14.67


def test_encode_edge_singleton():
    standard_table = init_graph()['st']
    with pytest.raises(ValueError):
        utils.encode_edge_singleton(standard_table, '')

    assert pytest.approx(utils.encode_edge_singleton(standard_table, 'a'), rel=1e-01) == 14.35
    assert pytest.approx(utils.encode_edge_singleton(standard_table, 'b'), rel=1e-01) == 15.09


def test_encode_singleton():
    standard_table = init_graph()['st']
    with pytest.raises(ValueError):
        utils.encode_singleton(standard_table, 0, 'a')

    assert pytest.approx(utils.encode_singleton(standard_table, 2, 'a'), rel=1e-01) == 14.35
    assert pytest.approx(utils.encode_singleton(standard_table, 1, 'x'), rel=1e-01) == 12.67

    # Test for graph


def init_ng():
    res = dict()

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
    res['ng'] = ng
    pattern = nx.Graph()
    pattern.add_nodes_from(range(1, 3))
    pattern.add_edge(1, 2, label='e')
    pattern.nodes[1]['label'] = 'A'
    res['pattern'] = pattern
    return res


def test_get_embeddings():
    ng = init_ng()['ng']
    pattern = init_ng()['pattern']
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
    ng = init_ng()['ng']
    pattern = init_ng()['pattern']
    graph = init_graph()['graph']
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
    graph = init_graph()['graph']
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
    graph = init_graph()['graph']
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


def init_graph2():
    res = dict()
    g = nx.DiGraph()
    g.add_nodes_from(range(1, 9))
    g.add_edge(2, 1, label='a')
    g.add_edge(4, 1, label='a')
    g.add_edge(6, 1, label='a')
    g.add_edge(6, 8, label='a')
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
    res['g'] = g
    label_codes = LC(g)
    res['label_codes'] = label_codes
    p1 = nx.DiGraph()
    p1.add_node(1, label='x')
    p1.add_node(2)
    p1.add_edge(1, 2, label='a')
    res['p1'] = p1
    p2 = nx.DiGraph()
    p2.add_node(1, label='y')
    p2.add_node(2)
    p2.add_edge(1, 2, label='b')
    res['p2'] = p2
    ct = CodeTable(label_codes, g)
    ct.add_row(CodeTableRow(res['p1']))
    ct.add_row(CodeTableRow(res['p2']))
    ct.cover()
    res['ct'] = ct
    return res


def test_get_two_nodes_all_port():
    test = nx.DiGraph()
    test.add_node(1, is_Pattern=True)
    test.add_node(2, is_Pattern=True)
    test.add_node(3)
    test.add_node(4)
    test.add_node(5, is_Pattern=True)
    test.add_edge(1, 3)
    test.add_edge(1, 4)
    test.add_edge(2, 3)
    test.add_edge(2, 4)
    test.add_edge(5, 4)

    assert len(utils.get_two_nodes_all_port(1, 2, test)) == 2
    print('\n', utils.get_two_nodes_all_port(1, 2, test))
    assert len(utils.get_two_nodes_all_port(5, 2, test)) == 1
    print('\n', utils.get_two_nodes_all_port(5, 2, test))


def test_generate_candidates():
    res = init_graph2()
    ct = res['ct']

    candidates = utils.generate_candidates_2(ct.rewritten_graph(), ct)
    # print(candidates)


def test_compute_pattern_usage():
    res = init_graph2()
    ct = res['ct']
    """ {2: ['P0v2', 'P0v2', 'P0v2', 'P1v1', 'P1v1', 'P1v1'], 
    5: ['P0v1', 'P0v1'], 6: ['P0v2', 'wv1', 'xv1'], 
    9: ['P1v2', 'zv1'], 11: ['P1v2', 'zv1'], 
    13: ['P1v2', 'zv1']} """
    usage = utils.compute_pattern_usage(ct.rewritten_graph(), ['P0v2'], {2})
    assert usage == 3
    usage = utils.compute_pattern_usage(ct.rewritten_graph(), ['zv1'], {11, 9, 13})
    assert usage == 3

    rwg = nx.DiGraph()
    rwg.add_node(1, is_Pattern=True, label='P1')
    rwg.add_node(2)
    rwg.add_node(3, is_Pattern=True, label='P2')
    rwg.add_node(4, is_Pattern=True, label='P2')
    rwg.add_node(5)
    rwg.add_edge(1, 2, label='v1')
    rwg.add_edge(1, 5, label='v2')
    rwg.add_edge(3, 1, label='v2')
    rwg.add_edge(3, 5, label='v1')
    rwg.add_edge(4, 2, label='v1')
    usage = utils.compute_pattern_usage(rwg, ['P1v1', 'P1v2'], {2, 5})
    assert usage == 1
    # print(usage)


def test_compute_candidate_usage():
    res = init_graph2()
    ct = res['ct']
    candidates = utils.generate_candidates(ct.rewritten_graph(), ct)
    c1 = Candidate('P0', 'P0', [('v2', 'v2')])
    c1.first_pattern = res['p1']
    c1.second_pattern = res['p1']
    c1.data_port = {2}
    utils.compute_candidate_usage(ct.rewritten_graph(), c1, ct, candidates)
    assert c1.usage == 1

    c2 = Candidate('P0', 'P1', [('v2', 'v1')])
    c2.first_pattern = res['p1']
    c2.second_pattern = res['p2']
    c2.data_port = {2}
    utils.compute_candidate_usage(ct.rewritten_graph(), c2, ct, candidates)
    assert c2.usage == 3

    c3 = Candidate('z', 'P1', [('v1', 'v2')])
    c3.second_pattern = res['p2']
    with pytest.raises(ValueError):
        utils.compute_candidate_usage(ct.rewritten_graph(), c3, ct, candidates)

    c4 = Candidate('w', 'x', [('v1', 'v1')])
    c4.data_port = {6}
    utils.compute_candidate_usage(ct.rewritten_graph(), c4, ct, candidates)
    assert c4.usage == 1

    c5 = Candidate('P1', 'z', [('v2', 'v1')])
    c5.first_pattern = res['p2']
    c5.data_port = {11, 9, 13}
    utils.compute_candidate_usage(ct.rewritten_graph(), c5, ct, candidates)
    assert c5.usage == 3


def test_compute_pattern_embeddings():
    res = init_graph2()
    ct = res['ct']

    assert utils.compute_pattern_embeddings(ct.rewritten_graph(), 'P1') == 3
    assert utils.compute_pattern_embeddings(ct.rewritten_graph(), 'w') == 1


def test_is_candidate_port_exclusive():
    res = init_graph2()
    ct = res['ct']
    candidates = utils.generate_candidates(ct.rewritten_graph(), ct)
    c = Candidate('P0', 'P0', [('v1', 'v1')])
    c1 = Candidate('P0', 'P0', [('v2', 'v2')])
    assert utils.is_candidate_port_exclusive(candidates, c, 5) is True
    assert utils.is_candidate_port_exclusive(candidates, c1, 2) is False


def test_get_candidates():
    res = init_graph2()
    ct = res['ct']
    restricted_candidates = utils.get_candidates(ct.rewritten_graph(), ct)
    assert len(restricted_candidates) == 8
    assert restricted_candidates[1].usage == 3
    assert restricted_candidates[1].exclusive_port_number == 0


def test_merge_candidate():
    pa = nx.DiGraph()
    pa.add_node(1, label='A')
    pa.add_node(2, label='B')
    pa.add_node(3, label='C')
    pa.add_edge(1, 2, label='a')
    pa.add_edge(2, 3, label='b')

    pb = nx.DiGraph()
    pb.add_node(1, label='D')
    pb.add_node(2, label='E')
    pb.add_node(3, label='F')
    pb.add_edge(1, 2, label='c')
    pb.add_edge(2, 3, label='d')

    c = Candidate('P0', 'P1', [('v2', 'v1')])
    c.first_pattern = pa
    c.second_pattern = pb

    graph = utils.merge_candidate(c)

    assert len(graph.nodes()) == 5
    assert len(graph.nodes[2]['label']) == 2
    assert graph[2][4]['label'] == 'c'

    c1 = Candidate('P0', 'P0', [('v3', 'v2')])
    c1.first_pattern = pa
    c1.second_pattern = pa
    g1 = utils.merge_candidate(c1)
    assert len(g1.nodes[3]['label']) == 2
    assert g1[4][3] is not None
    assert g1[3][5] is not None

    c2 = Candidate('P0', 'P1', [('v1', 'v1')])
    c2.first_pattern = pa
    c2.second_pattern = pb
    g2 = utils.merge_candidate(c2)
    assert len(g2.nodes[1]['label']) == 2
    assert g2[1][4]['label'] == 'c'

    c3 = Candidate('P0', 'P1', [('v1', 'v1'), ('v3', 'v3')])
    c3.first_pattern = pa
    c3.second_pattern = pb
    g3 = utils.merge_candidate(c3)
    assert len(g3.nodes()) == 4
    assert g3.nodes[1]['label'] == ('D', 'A')
    assert g3.nodes[3]['label'] == ('F', 'C')
    assert g3[4][3] is not None

    p1 = nx.DiGraph()
    p1.add_node(1, label='C')
    p1.add_node(2)
    p1.add_node(3, label='C')
    p1.add_edge(1, 2, label='single')
    p1.add_edge(3, 1, label='single')

    p2 = nx.DiGraph()
    p2.add_node(1, label='x')
    p2.add_node(2)
    p2.add_edge(1, 2, label='a')
    c4 = Candidate('P1', 'P1', [('v2', 'v3'), ('v3', 'v2'), ('v1', 'v1')])
    c4.first_pattern = p1
    c4.second_pattern = p1
    g4 = utils.merge_candidate(c4)
    assert len(g4.nodes()) == 3
