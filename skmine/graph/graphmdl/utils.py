""" Utils function"""
import cProfile
import io
import math
import pstats
from collections import Counter
import networkx as nx
from networkx import Graph
from networkx.algorithms import isomorphism as iso
from ..graphmdl.candidate import Candidate


def log2(value, total):
    """ Compute log2
    Parameters
    ----------
    value
    total

    Returns
    -------
    float
    """
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
            if type(d['label']) is not str:
                for i in d['label']:
                    edges[j] = i
                    j = j + 1
            else:
                edges[j] = d['label']
                j += 1
        else:
            raise ValueError("The edge must have a label")

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
            if type(d['label']) is tuple:
                for i in d['label']:
                    vertex[j] = i
                    j = j + 1
            else:
                vertex[j] = d['label']
                j += 1
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
        raise ValueError(f"{k} should be lower than {n} in binomial coefficient")
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
        raise ValueError(f"{x} should be higher than 1")
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
        raise ValueError(f"{x} should be higher than 0")
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

    vertex_total = len(pattern.nodes())

    total_label_description = math.log2(
        len(standard_table.vertex_lc()) + len(standard_table.edges_lc()))  # description length for all labels
    vertex_number_description = universal_integer_encoding_with0(vertex_total)  # description length for all vertex

    # Compute description length for vertex
    vertex_description = dict()
    for u, v in count_vertex_label(pattern).items():
        vertex_description[u] = standard_table.vertex_lc()[u] + universal_integer_encoding_with0(v) + math.log2(
            binomial(vertex_total, v))

    # Compute description length for edges

    edges_description = dict()
    for a, b in count_edge_label(pattern).items():
        edges_description[a] = standard_table.edges_lc()[a] + universal_integer_encoding_with0(b) + math.log2(
            binomial(int(math.pow(vertex_total, 2)), b))

    # Compute description length through description length of edges and vertex
    description_length = 0.0
    for i in vertex_description.values():
        description_length = description_length + i
    for j in edges_description.values():
        description_length = description_length + j

    # Delete the unnecessary variable
    del vertex_description
    del edges_description
    del vertex_total

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
    if vertex_label == "" or vertex_label is None:
        raise ValueError("You should give a vertex label ")
    else:
        # Compute description length for vertex
        desc = standard_table.vertex_lc()[vertex_label] + universal_integer_encoding_with0(1) + math.log2(
            binomial(1, 1))

        return desc + math.log2(
            len(standard_table.vertex_lc()) + len(standard_table.edges_lc())) + universal_integer_encoding_with0(1)


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
    if edge_label == "" or edge_label is None:
        raise ValueError("You should give an edge label")
    else:
        # Compute description length for vertex
        desc = standard_table.edges_lc()[edge_label] + universal_integer_encoding_with0(1) + math.log2(binomial(4, 1))

        return desc + math.log2(
            len(standard_table.vertex_lc()) + len(standard_table.edges_lc())) + universal_integer_encoding_with0(2)


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


def _node_match(node1, node2):
    """ Compare two given nodes
    Parameters
    ---------
    node1
    node2
    Returns
    -------
    bool
    """
    if 'label' in node1 and 'label' in node2:
        res = list()
        for i in node2['label']:
            res.append(i in node1['label'])
        return not (False in res)
    elif 'label' not in node1 and 'label' in node2:
        return False
    else:
        return True


def _edge_match(edge1, edge2):
    """ Compare two given edges
    Parameters
    ----------
    edge1
    edge2
    Returns
    -------
    bool
    """
    if 'label' in edge1 and 'label' in edge2:
        return edge1['label'] == edge2['label']
    else:
        return True


def get_embeddings(pattern, graph):
    """ Provide the embeddings of a pattern in a given graph
    Parameters
    ----------
    pattern
    graph
    Returns
    -------
    list
    """

    # Create functions to compare node and edge label
    comp = {
        'node_match': _node_match,
        'edge_match': _edge_match
    }
    graph_matcher = None
    # Create matcher according the graph type (directed or no)

    if nx.is_directed(graph):
        graph_matcher = iso.DiGraphMatcher(graph, pattern, **comp)
    else:
        graph_matcher = iso.GraphMatcher(graph, pattern, **comp)

    return list(graph_matcher.subgraph_monomorphisms_iter())


def is_vertex_singleton(pattern):
    """ Check if a given pattern is a vertex singleton pattern
    Parameters
    ---------
    pattern
    Returns
    ---------
    bool
    """
    return len(pattern.nodes()) == 1 and get_total_label(pattern) == 1


def is_edge_singleton(pattern):
    """ Check if a given pattern is a edge singleton pattern
    Parameters
    ---------
    pattern
    Returns
    ---------
    bool
    """
    if len(pattern.nodes()) == 2:
        # Check first if the nodes haven't labels
        if "label" not in pattern.nodes(data=True)[1] and "label" not in pattern.nodes(data=True)[2]:
            # Check if the edge have exactly one label
            return count_edge_label(pattern) is not None and len(count_edge_label(pattern).values()) == 1 and \
                   list(count_edge_label(pattern).values())[0] == 1
        else:
            return False
    else:
        return False


def get_support(embeddings):
    """ Compute the pattern support in the graph according a minimum image based support
    Parameters
    ---------
    embeddings

    Returns
    -------
    int
    """

    if len(embeddings) != 0:
        node_embeddings = dict()
        # Compute for each pattern node, the graph nodes who can replace,
        # and store it in a dictionary
        for e in embeddings:
            for i in e.items():
                if i[1] in node_embeddings:
                    node_embeddings[i[1]].add(i[0])
                else:
                    node_embeddings[i[1]] = set()

        # Compute for each pattern node,the total number of graph nodes who could replace,
        # and store it in a dictionary
        return min({key: len(value) for key, value in node_embeddings.items()}.values())
    else:
        return 0


def get_label_index(label, values):
    """ Provide a label index in the values
    Parameters
    ----------
    label
    values
    Returns
    ----------
    int
    """
    if label in values:
        return values.index(label)
    else:
        raise ValueError(f"{label} should be in the {values}")


def get_node_label(key, index, graph):
    """ Provide a particular node label in a given graph by the index and the node key
    Parameters
    ----------
    key
    index
    graph
    Returns
    ---------
    str
    """
    if key in graph.nodes():
        if len(graph.nodes(data=True)[key]['label']) > index and type(graph.nodes(data=True)[key]['label']) is tuple:
            return graph.nodes(data=True)[key]['label'][index]
        else:
            return graph.nodes(data=True)[key]['label']
    else:
        raise ValueError(f"{index} shouldn't be out of bounds and {key} should be a graph node")


def get_edge_label(start, end, graph):
    """ Provide a particular edge label in a given graph by the edge start, edge end
    Parameters
    ---------
    start
    end
    graph

    Returns
    ---------
    str
    """
    if start in graph.nodes() and end in graph.nodes():
        if (start, end) in list(graph.edges()) and 'label' in graph[start][end]:
            return graph[start][end]['label']
        elif (end, start) in list(graph.edges()) and 'label' in graph[end][start]:
            return graph[end][start]['label']
        else:
            raise ValueError(f"{start}-{end} should be a graph edge and should have a label")
    else:
        raise ValueError(f"{start} and {end} should be a graph nodes")


def is_without_edge(pattern):
    """ Check if the pattern is without edge
    Parameters
    ----------
    pattern

    Returns
    ---------
    bool
    """
    return len(pattern.edges()) == 0


def _get_node_labels(node, graph):
    """ Provide the given node label in the given graph
    Parameters
    ----------
    node
    graph
    Returns
    -------
    str || int
    """
    if 'label' in graph.nodes[node]:
        return graph.nodes[node]['label']
    else:
        return node


def display_graph(graph: Graph):
    """ Display a given graph in a specific string sequence
    Parameters
    ---------
    graph
    Returns
    --------
    str
    """
    msg = ""
    if len(graph.edges()) != 0:
        for edge in graph.edges(data=True):
            if "label" in edge[2]:
                msg += "{}--{}-->{}".format(_get_node_labels(edge[0], graph),
                                            edge[2]['label'], _get_node_labels(edge[1], graph)) + "\n"
            else:
                msg += "{}--->{}".format(_get_node_labels(edge[0], graph),
                                         _get_node_labels(edge[1], graph)) + "\n"
    else:
        msg += "{}".format(_get_node_labels(1, graph))
    return msg


def get_edge_in_embedding(embedding, pattern):
    """ Provide the pattern edge who are in an embedding
    Parameters
    ----------
    embedding
    pattern
    Returns
    -------
    set"""
    keys = list(embedding.keys())
    values = list(embedding.values())
    edges = set()
    i = 0
    while i <= len(keys) - 1:
        j = i
        while j <= len(keys) - 1:
            if (values[i], values[j]) in list(pattern.edges()):
                edges.add((values[i], values[j]))
            if (values[j], values[i]) in list(pattern.edges()):
                edges.add((values[j], values[i]))
            j += 1
        i += 1

    del keys
    del values

    return edges


def get_key_from_value(data, value):
    """ Provide a key in the dictionary from the value
    Parameters
    ----------
    data
    value
    Returns
    --------
    int
    """
    return [k for k, v in data.items() if v == value][0]


def get_two_nodes_all_port(node1, node2, rewritten_graph):
    """ Provide all port in a rewritten graph for a potential candidate nodes
     Parameters
     ----------
     node1
     node2
     rewritten_graph

     Returns
     -------
     list
     """
    # check for all port edges if the node 1 and the node2 are port neighbors
    res = set()
    for node in rewritten_graph.nodes(data=True):
        if 'is_Pattern' not in node[1]:
            edge1 = (node1, node[0])
            edge2 = (node2, node[0])
            cpt = 0
            for edge in rewritten_graph.in_edges(node[0]):
                if edge == edge1 or edge == edge2:
                    cpt += 1
            if cpt == 2:
                res.add(node[0])
    return list(res)


def get_all_candidate_ports_labels_tuple(rewritten_graph, ports, first_node, second_node, inverse=False):
    """ Provide a list with port label tuple for a given two nodes who are a candidate
    Parameters
    ---------
    rewritten_graph
    ports : a complete list of data port
    first_node
    second_node
    inverse : boolean to decide for the tuple elements positions

    Returns
    -------
    list"""
    port = []
    for p in ports:
        first_port = get_edge_label(first_node, p, rewritten_graph)
        second_port = get_edge_label(second_node, p, rewritten_graph)
        if not inverse:
            port.append((first_port, second_port))
        else:
            port.append((second_port, first_port))

    return port


def is_isomorphic(graph, pattern):
    """ Check if two graph is isomorphic
    Parameters
    ---------
    graph
    pattern
    Returns
    -------
    bool"""
    opt = {
        'node_match': _node_match,
        'edge_match': _edge_match
    }
    graph_matcher = None
    # Create matcher according the graph type (directed or no)

    if nx.is_directed(graph):
        graph_matcher = iso.DiGraphMatcher(graph, pattern, **opt)
    else:
        graph_matcher = iso.GraphMatcher(graph, pattern, **opt)

    return graph_matcher.is_isomorphic()


def get_automorphisms(graph):
    """ Provide a pattern automorphisms
    Parameters
    ----------
    graph
    Returns
    ---------
    list
    """
    opt = {
        'node_match': _node_match,
        'edge_match': _edge_match
    }
    graph_matcher = None
    # Create matcher according the graph type (directed or no)

    if nx.is_directed(graph):
        graph_matcher = iso.DiGraphMatcher(graph, graph, **opt)
    else:
        graph_matcher = iso.GraphMatcher(graph, graph, **opt)

    automorphisms = set()
    for auto in list(graph_matcher.isomorphisms_iter())[1:]:
        for i, j in auto.items():
            if i != j:
                automorphisms.add((i, j))
    return automorphisms


# To review begin

"""def generate_candidates(rewritten_graph, code_table):
    candidates = list()
    for node in rewritten_graph.nodes(data=True):
        if 'is_Pattern' in node[1] and node[1]['is_Pattern'] is True:  # Check if the node is a pattern
            for e in rewritten_graph.out_edges(node[0], data=True):  # iterate into the pattern out edges
                i = 0
                in_edges = list(rewritten_graph.in_edges(e[1], data=True))
                while i <= len(in_edges) - 1:  # iterate into in_edges of the pattern neighbor who is a port
                    if node[0] != in_edges[i][0]:
                        e2 = in_edges[i]
                        first_pattern = node[1]['label']
                        second_pattern = rewritten_graph.nodes[e2[0]]['label']
                        all_candidate_port = get_two_nodes_all_port(node[0], e2[0], rewritten_graph)

                        # respect the candidate pattern order

                        # If both candidate element are the pattern
                        if 'is_singleton' not in node[1] and 'is_singleton' not in rewritten_graph.nodes[e2[0]]:

                            if int(first_pattern.split('P')[1]) < int(second_pattern.split('P')[1]):
                                ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port,
                                                                             node[0], e2[0])
                                c = Candidate(first_pattern, second_pattern, ports)
                            else:
                                ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port,
                                                                             node[0], e2[0], True)
                                c = Candidate(second_pattern, first_pattern, ports)

                            c.first_pattern = code_table.rows()[int(c.first_pattern_label.split('P')[1])].pattern()
                            c.second_pattern = code_table.rows()[int(c.second_pattern_label.split('P')[1])].pattern()

                        # if one node is a pattern and the second a singleton
                        elif 'is_singleton' in node[1] and 'is_singleton' not in rewritten_graph.nodes[e2[0]]:
                            ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port, node[0],
                                                                         e2[0], True)
                            c = Candidate(second_pattern, first_pattern, ports)
                            c.first_pattern = code_table.rows()[int(c.first_pattern_label.split('P')[1])].pattern()

                        elif 'is_singleton' not in node[1] and 'is_singleton' in rewritten_graph.nodes[e2[0]]:
                            ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port, node[0],
                                                                         e2[0])
                            c = Candidate(first_pattern, second_pattern, ports)
                            c.first_pattern = code_table.rows()[int(c.first_pattern_label.split('P')[1])].pattern()

                        else:  # if both of the candidate elements are the singleton
                            if node[1]['label'] < rewritten_graph.nodes[e2[0]]['label']:
                                ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port,
                                                                             node[0], e2[0])
                                c = Candidate(first_pattern, second_pattern, ports)
                            else:
                                ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port,
                                                                             node[0], e2[0], True)
                                c = Candidate(second_pattern, first_pattern, ports)

                        for p in all_candidate_port:
                            c.data_port.add(p)
                        candidates.append(c)
                    i += 1
    return candidates"""


def get_port_candidates(patterns_list):
    """ Provide the tuple of two nodes that are neighbors of the same port
    Parameters
    ----------
    patterns_list: the port neighbors list
    Returns
    -------
    set
    """
    res = set()
    i = 0
    while i <= len(patterns_list) - 1:
        j = i + 1
        while j <= len(patterns_list) - 1:
            res.add((patterns_list[i], patterns_list[j]))
            j += 1
        i += 1
    return res


def generate_candidates(rewritten_graph, code_table):
    """ Search in the rewritten graph, the pattern who share a same port
    Parameters
    ----------
    rewritten_graph
    code_table
    Returns
    ----------
    list
    """
    candidates = list()
    ports = [node[0] for node in rewritten_graph.nodes(data=True) if 'is_Pattern' not in node[1]]
    for port in ports:
        patterns = [edge[0] for edge in rewritten_graph.in_edges(port)]  # the port neighbor
        port_candidates = get_port_candidates(patterns)
        del patterns

        for c in port_candidates:
            first_pattern = rewritten_graph.nodes[c[0]]
            second_pattern = rewritten_graph.nodes[c[1]]
            all_candidate_port = get_two_nodes_all_port(c[0], c[1], rewritten_graph)
            candidate_ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port,
                                                                   c[0], c[1])
            candidate_ports_inverse = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port,
                                                                           c[0], c[1], True)
            # respect the candidate pattern order

            # If both candidate element are the pattern
            if 'is_singleton' not in first_pattern and 'is_singleton' not in second_pattern:

                if int(first_pattern['label'].split('P')[1]) < int(second_pattern['label'].split('P')[1]):
                    candidate = Candidate(first_pattern['label'], second_pattern['label'], candidate_ports)
                else:
                    candidate = Candidate(second_pattern['label'], first_pattern['label'], candidate_ports_inverse)

                candidate.first_pattern = code_table.rows()[int(candidate.first_pattern_label.split('P')[1])].pattern()
                candidate.second_pattern = code_table.rows()[
                    int(candidate.second_pattern_label.split('P')[1])].pattern()

            # if one node is a pattern and the second a singleton
            elif 'is_singleton' in first_pattern and 'is_singleton' not in second_pattern:
                candidate = Candidate(second_pattern['label'], first_pattern['label'], candidate_ports_inverse)
                candidate.first_pattern = code_table.rows()[int(candidate.first_pattern_label.split('P')[1])].pattern()

            elif 'is_singleton' not in first_pattern and 'is_singleton' in second_pattern:
                candidate = Candidate(first_pattern['label'], second_pattern['label'], candidate_ports)
                candidate.first_pattern = code_table.rows()[int(candidate.first_pattern_label.split('P')[1])].pattern()
            # if both of the candidate pattern are singletons
            else:  # if both of the candidate elements are the singleton
                if first_pattern['label'] < second_pattern['label']:
                    candidate = Candidate(first_pattern['label'], second_pattern['label'], candidate_ports)
                else:
                    candidate = Candidate(second_pattern['label'], first_pattern['label'], candidate_ports_inverse)

            # Store all candidates' data_port to see if there are exclusive port
            for p in all_candidate_port:
                candidate.data_port.add(p)

            # Add the candidates to the list of candidates

            """if candidate not in candidates:
                _complete_candidate(rewritten_graph, candidate, code_table, candidates)
                candidates.append(candidate)
            else:
                temp_candidate = get_candidate_from_data(candidates, candidate)
                for p in candidate.data_port:
                    temp_candidate.data_port.add(p)
                _complete_candidate(rewritten_graph, temp_candidate, code_table, candidates)"""
            # candidates.append(candidate)
    return candidates


def generate_candidates_2(rewritten_graph, code_table):
    """ Search in the rewritten graph, the pattern who share a same port
    Parameters
    ----------
    rewritten_graph
    code_table
    Returns
    ----------
    list
    """
    candidates = list()
    ports = [node[0] for node in rewritten_graph.nodes(data=True) if 'is_Pattern' not in node[1]]
    for port in ports:
        patterns = [edge[0] for edge in rewritten_graph.in_edges(port)]  # the port neighbor
        port_candidates = get_port_candidates(patterns)
        # del patterns
        for c in port_candidates:
            first_pattern = rewritten_graph.nodes[c[0]]
            second_pattern = rewritten_graph.nodes[c[1]]
            all_candidate_port = get_two_nodes_all_port(c[0], c[1], rewritten_graph)

            # respect the candidate pattern order

            # If both candidate element are the pattern
            if 'is_singleton' not in first_pattern and 'is_singleton' not in second_pattern:

                if int(first_pattern['label'].split('P')[1]) < int(second_pattern['label'].split('P')[1]):
                    ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port,
                                                                 c[0], c[1])
                    candidate = Candidate(first_pattern['label'], second_pattern['label'], ports)
                else:
                    ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port,
                                                                 c[0], c[1], True)
                    candidate = Candidate(second_pattern['label'], first_pattern['label'], ports)

                candidate.first_pattern = code_table.rows()[int(candidate.first_pattern_label.split('P')[1])].pattern()
                candidate.second_pattern = code_table.rows()[
                    int(candidate.second_pattern_label.split('P')[1])].pattern()

            # if one node is a pattern and the second a singleton
            elif 'is_singleton' in first_pattern and 'is_singleton' not in second_pattern:
                ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port, c[0],
                                                             c[1], True)
                candidate = Candidate(second_pattern['label'], first_pattern['label'], ports)
                candidate.first_pattern = code_table.rows()[int(candidate.first_pattern_label.split('P')[1])].pattern()

            elif 'is_singleton' not in first_pattern and 'is_singleton' in second_pattern:
                ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port, c[0],
                                                             c[1])
                candidate = Candidate(first_pattern['label'], second_pattern['label'], ports)
                candidate.first_pattern = code_table.rows()[int(candidate.first_pattern_label.split('P')[1])].pattern()
            # if both of the candidate pattern are singletons
            else:  # if both of the candidate elements are the singleton
                if first_pattern['label'] < second_pattern['label']:
                    ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port,
                                                                 c[0], c[1])
                    candidate = Candidate(first_pattern['label'], second_pattern['label'], ports)
                else:
                    ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port,
                                                                 c[0], c[1], True)
                    candidate = Candidate(second_pattern['label'], first_pattern['label'], ports)

            # Store all candidates' data_port to see if there are exclusive port
            for p in all_candidate_port:
                candidate.data_port.add(p)

            # Add the candidates to the list of candidates

    return candidates


def compute_pattern_usage(rewritten_graph, pattern_format, ports):
    """ Compute pattern usage in the rewritten graph
    Parameters
    ---------
    rewritten_graph
    pattern_format: The pattern label concatenated with the port label
    ports
    Returns
    -------
    float
    """
    pattern_usage = 0
    ports_infos = get_port_node_infos(rewritten_graph)
    # compute for each given ports, the information about his pattern neighbors
    for port in ports:
        infos = ports_infos[port]
        # compute how many times, the pattern is the port neighbors
        for i in infos:
            if i in pattern_format:
                pattern_usage += 1
        # If the pattern have different appearances as neighbors,
        # return the pattern total usage sliced by the number of appearances

    return pattern_usage / len(pattern_format)


def compute_candidate_usage(rewritten_graph, candidate, code_table, candidates):
    """ Compute an estimated usage for a  given candidate
    Parameters
    -----------
    rewritten_graph
    candidate
    code_table
    candidates
    """
    p1_format = [candidate.first_pattern_label + p[0] for p in
                 candidate.port]  # format of the first candidate pattern with its ports
    p2_format = [candidate.second_pattern_label + p[1] for p in
                 candidate.port]  # format of the second candidate pattern with its ports

    # Compute each candidate patterns usage
    first_pattern_usage = compute_pattern_usage(rewritten_graph, p1_format, list(candidate.data_port))
    second_pattern_usage = compute_pattern_usage(rewritten_graph, p2_format, list(candidate.data_port))

    del p1_format
    del p2_format

    # Compute the candidate estimated usage according the candidate particularities
    if candidate.first_pattern_label == candidate.second_pattern_label and \
            (candidate == candidate.inverse() or candidate.inverse() not in candidates):
        candidate.set_usage(int(first_pattern_usage / 2))
    elif candidate.first_pattern is not None and candidate.second_pattern is not None:
        if is_without_edge(candidate.first_pattern) and not is_without_edge(candidate.second_pattern):
            # if only the second pattern doesn't have edges
            candidate.set_usage(second_pattern_usage)
        elif is_without_edge(candidate.second_pattern) and not is_without_edge(candidate.first_pattern):
            # if only the second pattern doesn't have edges
            candidate.set_usage(first_pattern_usage)
        elif is_without_edge(candidate.first_pattern) and is_without_edge(candidate.second_pattern):
            candidate.set_usage(compute_pattern_embeddings(rewritten_graph, candidate.first_pattern_label))
        else:
            # if both of the pattern have edges
            candidate.set_usage(min(first_pattern_usage, second_pattern_usage))

    elif candidate.second_pattern is not None and candidate.first_pattern is None:
        raise ValueError("The singleton should be the second pattern")

    elif candidate.first_pattern is not None and candidate.second_pattern is None:

        # A pattern with a singleton
        if code_table.is_ct_edge_singleton(candidate.second_pattern_label):
            # the singleton is an edge singleton
            if is_without_edge(candidate.first_pattern):
                # if the pattern doesn't have edge
                candidate.set_usage(second_pattern_usage)
            else:
                candidate.set_usage(min(first_pattern_usage, second_pattern_usage))
        else:
            # The singleton is a vertex singleton
            candidate.set_usage(first_pattern_usage)

    else:
        # Only singleton case
        if code_table.is_ct_edge_singleton(candidate.first_pattern_label) \
                and not code_table.is_ct_edge_singleton(candidate.second_pattern_label):
            # only first pattern is an edge singleton
            candidate.set_usage(first_pattern_usage)
        elif not code_table.is_ct_edge_singleton(candidate.first_pattern_label) \
                and code_table.is_ct_edge_singleton(candidate.second_pattern_label):
            # only second pattern is an edge singleton
            candidate.set_usage(second_pattern_usage)
        else:
            # both singleton is vertex singleton
            candidate.set_usage(min(first_pattern_usage, second_pattern_usage))


def compute_pattern_embeddings(rewritten_graph, pattern):
    """ Compute pattern embeddings in the rewritten graph
    Parameters
    ----------
    rewritten_graph
    pattern
    Returns
    --------
    int
    """
    res = 0
    for node in rewritten_graph.nodes(data=True):
        if node[1]['label'] == pattern:
            res += 1
    return res


def get_candidate_from_data(data, candidate):
    """ Provide a candidate who match with a given candidate from a list
    Parameters
    -----------
    data
    candidate
    Returns
    -------
    Candidate
    """
    for c in data:
        if candidate == c:
            return c


def is_candidate_port_exclusive(candidates, candidate, port):
    """ Check if a port are neighbors who are not the candidate nodes number
     Parameters
     -----------
     candidates
     candidate
     port
     Returns
     ----------
     bool
     """
    # search in the candidates list if there is one
    # who are the given candidate data port as his data port

    res = [c == candidate for c in candidates for p in c.data_port if p == port]
    """for c in candidates:
        for p in c.data_port:
            if p == port:
                res.append(c == candidate)"""

    return not (False in res)


def _order_candidates(candidate: Candidate):
    """Provide the candidate elements to order candidates
    Parameters
    ----------
    candidate
    Returns
    -------
    list
    """
    return [candidate.usage, candidate.exclusive_port_number]


def _compare_candidate(candidates):
    """ Sort a given candidates list
    Parameters
    -----------
    candidates
    Returns
    -------
    list
    """
    candidates.sort(reverse=True, key=_order_candidates)
    return candidates


def get_candidates(rewritten_graph, code_table):
    """ Get the restricted list of candidates
    Parameters
    ----------
    rewritten_graph
    code_table
    Returns
    -------
    list
    """
    res = []
    # generate an exhaustive list of candidates
    candidates = generate_candidates(rewritten_graph, code_table)
    # Assemble similar candidate,
    # assemble also their ports
    for candidate in candidates:
        if candidate not in res:
            res.append(candidate)
        else:
            c = res[res.index(candidate)]
            for p in candidate.data_port:
                c.data_port.add(p)

    # Compute candidate estimated usage, exclusive port and merge pattern
    for r in candidates:
        compute_candidate_usage(rewritten_graph, r, code_table, candidates)
        exclusive_port_number = 0
        # Search exclusive port
        for port in r.data_port:
            if is_candidate_port_exclusive(candidates, r, port):
                exclusive_port_number += 1
        r.exclusive_port_number = exclusive_port_number

        # create singleton pattern
        if r.first_pattern is None and r.second_pattern is not None:
            raise ValueError("The second pattern should be the singleton")
        elif r.first_pattern is not None and r.second_pattern is None:
            r.second_pattern = create_singleton_pattern(r.second_pattern_label, code_table)
        elif r.first_pattern is None and r.second_pattern is None:
            r.first_pattern = create_singleton_pattern(r.first_pattern_label, code_table)
            r.second_pattern = create_singleton_pattern(r.second_pattern_label, code_table)

        # r.final_pattern = merge_candidate(r)  # create merge pattern
        # Compute candidate merge pattern description length
        # r.compute_description_length(code_table.label_codes())

        # Filter the list by combining candidates with isomorphic merge patterns
    """ i = 0
    while i <= len(res) - 1:
        j = i + 1
        while j <= len(res) - 1:
            if is_isomorphic(res[i].final_pattern, res[j].final_pattern):
                couple = _compare_candidate([res[i], res[j]])
                res.remove(couple[1])
            j += 1
        i += 1 """

    # sort the definitive candidate list by estimated usage, exclusive port,
    # and description length of the merge pattern
    # res.sort(reverse=True, key=_order_candidates)  # Sort the list
    return res


# to review end

def create_candidate_first_pattern(pattern, graph):
    """ Create a given candidate first pattern nodes and edges
    Parameters
    ---------
    pattern
    graph
    """
    # create the pattern nodes with their labels in the given graph
    for node in pattern.nodes(data=True):
        if 'label' in node[1]:
            graph.add_node(node[0], label=node[1]['label'])
        else:
            graph.add_node(node[0])
    # create the pattern edges with their labels  in the given graph
    for edge in pattern.edges(data=True):
        graph.add_edge(edge[0], edge[1], label=edge[2]['label'])


def create_candidate_second_pattern(pattern, graph, ports):
    """ Create a given candidate second pattern nodes and edges,
        and connected it with the existent graph elements by the port
        Parameters
        ---------
        pattern
        graph
        ports
    """
    mapping = dict()  # Mapping between pattern nodes and graph nodes number
    second_port = [p[1] for p in ports]  # pattern ports number
    first_port = [p[0] for p in ports]  # graph ports number

    # create the non-port pattern nodes with their labels in the graph
    # for the pattern port node add only the labels if it's necessary
    for node in pattern.nodes(data=True):
        if node[0] not in second_port:  # if the node isn't a port node
            mapping[node[0]] = len(graph.nodes()) + 1  # Add the node to the mapping
            if 'label' in node[1]:
                graph.add_node(len(graph.nodes()) + 1, label=node[1]['label'])
            else:
                graph.add_node(len(graph.nodes()) + 1)
        else:  # if the node is a port
            index = second_port.index(node[0])  # Get the node mapping
            if 'label' in node[1]:
                if 'label' in graph.nodes[first_port[index]]:
                    new_label = _get_new_label(node[1]['label'], graph.nodes[first_port[index]]['label'])
                    graph.nodes[first_port[index]]['label'] = new_label
                else:
                    graph.nodes[first_port[index]]['label'] = node[1]['label']

    # Set the mapping for the pattern port node
    for p in ports:
        mapping[p[1]] = p[0]

    # Create pattern edges with their labels
    for edge in pattern.edges(data=True):
        if edge[0] in second_port:
            port = first_port[second_port.index(edge[0])]
            graph.add_edge(port, mapping[edge[1]], label=edge[2]['label'])
        elif edge[1] in second_port:
            port = first_port[second_port.index(edge[1])]
            graph.add_edge(mapping[edge[0]], port, label=edge[2]['label'])
        elif edge[0] == edge[1] and edge[0] in port[1]:
            port = first_port[second_port.index(edge[1])]
            graph.add_edge(port[0], port, label=edge[2]['label'])
        else:
            graph.add_edge(mapping[edge[0]], mapping[edge[1]], label=edge[2]['label'])


def create_singleton_pattern(label, code_table):
    """ Create a graph who represent a singleton
    Parameters
    ----------
    label
    code_table
    Returns
    -------
    Graph
    """
    pattern = nx.DiGraph()
    if code_table.is_ct_edge_singleton(label):
        pattern.add_nodes_from(range(1, 3))
        pattern.add_edge(1, 2, label=label)
    elif code_table.is_ct_vertex_singleton(label):
        pattern.add_node(1, label=label)
    else:
        raise ValueError("The label should be a vertex or an edge label")
    return pattern


def _get_new_label(first_label, second_label):
    if type(second_label) is str:
        if type(first_label) is str:
            if first_label != second_label:
                return first_label, second_label
            else:
                return second_label
        else:
            label = list()
            label.append(second_label)
            for l in first_label:
                if l not in label:
                    label.append(l)
            return label
    else:
        label = list(second_label)
        if type(first_label) is str:
            if first_label not in second_label:
                label.append(first_label)
        else:
            for l in first_label:
                if l not in label:
                    label.append(l)
        return label


def merge_candidate(candidate):
    """ Merge a candidate pattern
    Parameters
    -----------
    candidate
    Returns
    --------
    Graph
    """
    graph = nx.DiGraph()
    ports = []
    for p in candidate.port:
        port = (int(p[0].split('v')[1]), int(p[1].split('v')[1]))
        ports.append(port)
    # Create first pattern node and edges
    create_candidate_first_pattern(candidate.first_pattern, graph)
    # Create second pattern node and edges
    create_candidate_second_pattern(candidate.second_pattern, graph, ports)
    return graph


def count_port_node(rewritten_graph):
    """ Count the port number in a rewritten graph
    Parameters
    ----------
    rewritten_graph
    Returns
    --------
    int
    """
    numb = 0
    for node in rewritten_graph.nodes(data=True):
        if 'is_Pattern' not in node[1]:
            numb += 1
    return numb


def get_pattern_node_infos(rewritten_graph):
    """ Provide pattern node information from the rewritten graph
    Parameters
    ----------
    rewritten_graph
    Returns
    --------
    dict
    """
    pattern_node = dict()
    for node in rewritten_graph.nodes(data=True):
        if 'is_Pattern' in node[1]:
            res = []
            for edge in rewritten_graph.edges(node[0], data=True):
                res.append(edge[2]['label'])

        if node[1]['label'] in pattern_node:
            pattern_node[node[1]['label']].append(res)
        else:
            pattern_node[node[1]['label']] = []
            pattern_node[node[1]['label']].append(res)

        if 'is_singleton' in node[1]:
            if node[1]['is_singleton'] is True:
                if 'singleton' not in pattern_node[node[1]['label']]:
                    pattern_node[node[1]['label']].append('singleton')
            else:
                raise ValueError("is_singleton should be true or shouldn't exist")

    return pattern_node


def get_port_node_infos(rewritten_graph):
    """ Provide port node information from the rewritten graph
    Parameters
    ----------
    rewritten_graph
    Returns
    -------
    dict
    """
    port_node = dict()
    for node in rewritten_graph.nodes(data=True):
        if 'is_Pattern' not in node[1]:
            for edge in rewritten_graph.in_edges(node[0], data=True):
                if node[0] in port_node:
                    port_node[node[0]].append(rewritten_graph.nodes(data=True)[edge[0]]['label'] + edge[2]['label'])
                else:
                    port_node[node[0]] = []
                    port_node[node[0]].append(rewritten_graph.nodes(data=True)[edge[0]]['label'] + edge[2]['label'])
    return port_node


def get_port_node(rewritten_graph, node):
    """ Provide an embedding vertex port
    Parameters
    ----------
    rewritten_graph
    node
    Returns
    --------
    set
    """
    res = set()
    for edge in rewritten_graph.out_edges(node, data=True):
        res.add(int(edge[2]['label'].split('v')[1]))
    return res


def get_graph_from_file(file):
    file = open(file)
    lines = [line.split(' ') for line in file][3:]
    graph = nx.DiGraph()
    for line in lines:
        if line[0] == 'v':
            graph.add_node(int(line[1]) + 1, label=line[3].split('\n')[0])
        else:
            graph.add_edge(int(line[1]) + 1, int(line[2]) + 1, label=line[3].split('\n')[0])
    file.close()
    return graph
