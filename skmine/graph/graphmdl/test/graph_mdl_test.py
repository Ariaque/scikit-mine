import networkx as nx
from skmine.graph.graphmdl.graph_mdl import GraphMDL
import pytest

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


def test_fit():
    with pytest.raises(ValueError):
        GraphMDL().fit(None)

    mdl = GraphMDL(debug=True)
    mdl.fit(g)
    # mdl.summary()
    assert mdl.description_length() != 0.0

    assert GraphMDL().fit(g, timeout=0.01).description_length() != 0


def test_patterns():
    assert len(GraphMDL().fit(g).patterns()) == 3


def test_description_length():
    assert pytest.approx(GraphMDL().fit(g).description_length(), rel=1e-01) == 144.8


def test_initial_description_length():
    assert pytest.approx(GraphMDL().fit(g).initial_description_length(), rel=1e-01) == 256.3
