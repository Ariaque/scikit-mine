""" Utils function"""
import math
from collections import Counter
import networkx as nx
from networkx import Graph
from networkx.algorithms import isomorphism as iso
""" Compute description length
  
  Parameters
  -----------
  value : double 
      a label instance number
  total : double 
      total number of label
      
  Returns
  -------
    float   .
"""


def log2(value, total):
    return -math.log2(value / total)


def count_edge_label(graph: Graph):
    """ Compute edge label instance in a graph and store the result

       Parameters
       ----------
       graph:Graph
         the treated graph

       Returns
       -------
         dict
    """

    edges = dict()
    j = 1
    for u, v, d in graph.edges(data=True):
        if 'label' in d:
            for i in d['label']:
                edges[j] = i
                j = j + 1
    return dict([(i, j) for i, j in Counter(edges.values()).items()])


def count_vertex_label(graph: Graph):
    """ Compute vertex label instance in a graph and store the result

         Parameters
         ----------
         graph:Graph
           the treated graph

         Returns
         -------
           dict
      """

    vertex = dict()
    j = 1
    for u, d in graph.nodes(data=True):
        if 'label' in d:
            for i in d['label']:
                vertex[j] = i
                j = j + 1

    return dict([(i, j) for i, j in Counter(vertex.values()).items()])


def get_total_label(graph: Graph):
    """ Compute total number of labels in the graph
       Parameters
       ----------
       graph:Graph
         the treated graph

       Returns
       -------
         float
    """

    labels = count_edge_label(graph)
    labels.update(count_vertex_label(graph))
    total = 0.0  # The total number of labels in the graph
    for i in labels.values():
        total = total + i

    return total


def binomial(n, k):
    """ Compute the binomial coefficient for a given n and given k. Also called "n choose k"

    Parameters
    ----------
    n
    k

    Returns
    -------
      float
    """
    if k > n:
        raise ValueError("k should be lower than n in binomial coefficient")
    elif k == 0:
        return 1
    elif k > n / 2:
        return binomial(n, n - k)

    return n * binomial(n - 1, k - 1) / k


def universal_integer_encoding(x):
    """ Compute universal codeword sets and representation for integers from 1

    Parameters
    ----------
    x

    Returns
    -------
      int
    """
    if x < 1:
        raise ValueError()
    else:
        return math.floor(math.log2(x)) + 2 * math.floor(math.log2(math.floor(math.log2(x)) + 1)) + 1


def universal_integer_encoding_with0(x):
    """ Compute universal codeword sets and representation for integers from 0

    Parameters
    ----------
    x

    Returns
    -------
    int
    """
    if x < 0:
        raise ValueError()
    else:
        return universal_integer_encoding(x + 1)


def encode(pattern: Graph, standard_table):
    """ Compute a graph description length

    Parameters
    ----------
    standard_table
    pattern

    Returns
    -------
    float
    """
    edges = count_edge_label(pattern)  # count each pattern edge label occurrences
    vertex = count_vertex_label(pattern)  # count each pattern vertex label occurrences

    # Get total number of label in the standard table
    total_label = len(standard_table.vertex_st()) + len(standard_table.edges_st())

    vertex_number = len(pattern.nodes())

    total_label_description = math.log2(total_label)  # description length for all labels
    vertex_number_description = universal_integer_encoding_with0(vertex_number)  # description length for all vertex

    # Compute description length for vertex
    vertex_description = dict()
    for u, v in vertex.items():
        desc = standard_table.vertex_st()[u] + universal_integer_encoding_with0(v) + math.log2(
            binomial(vertex_number, v))
        vertex_description[u] = desc

    # Compute description length for edges

    edges_description = dict()
    for a, b in edges.items():
        desc = standard_table.edges_st()[a] + universal_integer_encoding_with0(b) + math.log2(
            binomial(int(math.pow(vertex_number, 2)), b))
        edges_description[a] = desc

    # Compute description length through description length of edges and vertex
    description_length = 0.0
    for i in vertex_description.values():
        description_length = description_length + i
    for j in edges_description.values():
        description_length = description_length + j

    return description_length + total_label_description + vertex_number_description


def encode_vertex_singleton(standard_table, vertex_label):
    """ Compute a vertex singleton description length

        Parameters
        ----------
        standard_table
        vertex_label
        Returns
        -------
        float
    """
    if vertex_label == "":
        raise ValueError()
    else:
        # Get total number of label in the standard table
        total_label = len(standard_table.vertex_st()) + len(standard_table.edges_st())
        total_label_description = math.log2(total_label)  # description length for all labels
        vertex_number_description = universal_integer_encoding_with0(1)  # description length for all vertex

        # Compute description length for vertex
        desc = standard_table.vertex_st()[vertex_label] + universal_integer_encoding_with0(1) + math.log2(
            binomial(1, 1))

        return desc + total_label_description + vertex_number_description


def encode_edge_singleton(standard_table, edge_label):
    """ Compute an edge singleton description length

            Parameters
            ----------
            standard_table
            edge_label

            Returns
            -------
            float
    """
    if edge_label == "":
        raise ValueError()
    else:
        # Get total number of label in the standard table
        total_label = len(standard_table.vertex_st()) + len(standard_table.edges_st())
        total_label_description = math.log2(total_label)  # description length for all labels
        vertex_number_description = universal_integer_encoding_with0(2)  # description length for all vertex

        # Compute description length for vertex
        desc = standard_table.edges_st()[edge_label] + universal_integer_encoding_with0(1) + math.log2(binomial(4, 1))

        return desc + total_label_description + vertex_number_description


def encode_singleton(standard_table, arity, label):
    """ Compute a singleton description length by her arity

        Parameters
        ----------
        standard_table
        label
        arity
        Returns
        -------
        float
    """

    if arity == 1:
        return encode_vertex_singleton(standard_table, label)
    elif arity == 2:
        return encode_edge_singleton(standard_table, label)
    else:
        raise ValueError("arity should must be 1 or 2")


def get_embeddings(pattern, graph):
    # Create functions to compare node and edge label
    node_match = iso.categorical_node_match('label', '')
    edge_match = iso.categorical_edge_match('label', '')
    comp = {
        'node_match': node_match,
        'edge_match': edge_match
    }
    graph_matcher = None
    # Create matcher according the graph type (directed or no)

    if nx.is_directed(graph):
        graph_matcher = iso.DiGraphMatcher(graph, pattern, **comp)
    else:
        graph_matcher = iso.GraphMatcher(graph, pattern, **comp)

    return list(graph_matcher.subgraph_isomorphisms_iter())
