import networkx as nx
from ..standard_table import StandardTable
import pytest


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

    graph = g
    ST = StandardTable(graph)
    res['st'] = ST
    res['graph'] = graph
    return res


def test_total_label():
    st = init_graph()['st']
    assert st.total_label() == 17.0


def test_vertex_st():
    st = init_graph()['st']
    assert pytest.approx(st.vertex_st()['y'], rel=1e-01) == 4.09
    assert pytest.approx(st.vertex_st()['x'], rel=1e-01) == 2.09
    assert pytest.approx(st.vertex_st()['z'], rel=1e-01) == 2.50
    assert pytest.approx(st.vertex_st()['w'], rel=1e-01) == 4.09


def test_edges_st():
    st = init_graph()['st']
    assert pytest.approx(st.edges_st()['a'], rel=1e-01) == 1.77
    assert pytest.approx(st.edges_st()['b'], rel=1e-01) == 2.50


def test_encode():
    st = init_graph()['st']
    g1 = nx.DiGraph()
    g1.add_nodes_from(range(1, 3))
    g1.add_edge(1, 2, label='a')
    g1.nodes[1]['label'] = 'x'
    assert pytest.approx(st.encode(g1), rel=1e-01) == 21.44


def test_encode_singleton_vertex():
    st = init_graph()['st']
    assert pytest.approx(st.encode_singleton_vertex('x'), rel=1e-01) == 12.67


def test_encode_singleton_edge():
    st = init_graph()['st']
    assert pytest.approx(st.encode_singleton_edge('a'), rel=1e-01) == 14.35
