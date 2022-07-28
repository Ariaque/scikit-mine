import networkx as nx
from skmine.graph.graphmdl.graph_mdl import GraphMDl
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
        GraphMDl().fit(None)

    mdl = GraphMDl()
    mdl.fit(g)
    # mdl.summary()
    assert mdl.description_length != 0.0
    # assert len(mdl.patterns) != 0
