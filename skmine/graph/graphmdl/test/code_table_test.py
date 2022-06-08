import networkx as nx
import pytest
from ..code_table import *
from ..code_table_row import CodeTableRow


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
