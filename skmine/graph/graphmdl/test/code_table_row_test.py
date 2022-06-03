import networkx as nx
import pytest
from ..code_table_row import CodeTableRow

pattern = nx.DiGraph()
pattern.add_node(1)
pattern.add_node(2)
pattern.add_edge(1, 2)

row = CodeTableRow(pattern)


def test_pattern():
    assert len(row.pattern().nodes()) == 2
    assert len(row.pattern().edges()) == 1


def test_pattern_code():
    assert row.pattern_code() is None


def test_set_pattern_code():
    row.set_pattern_code(3.0)
    assert row.pattern_code() == 3.0


def test_pattern_port_code():
    assert row.pattern_port_code() is None


def test_embeddings():
    assert len(row.embeddings()) == 0


def test_set_embeddings():
    row.set_embeddings([0, 1, 2, 3])
    assert len(row.embeddings()) == 4
    assert row.embeddings()[0] == 0
    assert row.embeddings()[3] == 3
