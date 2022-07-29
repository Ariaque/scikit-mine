import time
from collections import defaultdict

import numpy
import pandas

from .code_table_row import CodeTableRow
import math
import networkx as nx
import skmine.graph.graphmdl.utils as utils


def _order_rows(row: CodeTableRow):
    """ Provide the row elements to order rows
    Parameters
    ----------
    row
    Returns
    --------
    tuple
    """
    length = utils.get_total_label(row.pattern())
    support = utils.get_support(row.embeddings())
    return length, support


def is_node_marked(node_number, graph, cover_marker, label):
    """ Check if a particular node label  is already marked
    Parameters
    -----------
    node_number
    graph
    cover_marker
    label
    Returns
    ---------
    bool

    """
    if node_number in graph.nodes() and label in graph.nodes(data=True)[node_number]['label']:
        node = graph.nodes(data=True)[node_number]
        return 'cover_mark' in node and label in node['cover_mark'] and node['cover_mark'][label] == cover_marker
    else:
        raise ValueError(f"{node_number} should be a graph node and {label} a node label")


def is_node_labels_marked(node_number, graph, cover_marker, labels):
    """ Check if one or more particular node label are marked
    Parameters
    -----------
    node_number
    graph
    cover_marker
    labels
    Returns
    --------
    bool"""
    if type(labels) is tuple:
        return not (False in [is_node_marked(node_number, graph, cover_marker, label) for label in labels])
    else:
        return is_node_marked(node_number, graph, cover_marker, labels)


def mark_node(node_number, graph, cover_marker, label):
    """ Mark a node in the graph by the cover marker
    Parameters
    ---------
    node_number
    graph
    cover_marker
    label
    """

    if label is not None:
        # check if the pattern node are many labels
        if type(label) is tuple:  # if yes, mark each label if it is a graph node label
            for j in label:
                if j in graph.nodes(data=True)[node_number]['label']:
                    if 'cover_mark' in graph.nodes(data=True)[node_number]:
                        graph.nodes(data=True)[node_number]['cover_mark'][j] = cover_marker
                    else:
                        graph.nodes(data=True)[node_number]['cover_mark'] = {j: cover_marker}

        else:  # if no, mark the unique label

            if 'cover_mark' in graph.nodes(data=True)[node_number]:
                graph.nodes(data=True)[node_number]['cover_mark'][label] = cover_marker
            else:
                graph.nodes(data=True)[node_number]['cover_mark'] = {label: cover_marker}
    else:
        raise ValueError("label shouldn't be empty or none")


def is_edge_marked(start, end, graph, cover_marker, label):
    """ Check if an edge is already marked
      Parameters
      -----------
      start
      end
      graph
      cover_marker
      label

      Returns
      ---------
      bool
      """
    if start in graph.nodes() and end in graph.nodes() and \
            graph[start][end] is not None and 'label' in graph[start][end]:
        edge = graph[start][end]
        return 'cover_mark' in edge and label in edge['cover_mark'] and edge['cover_mark'][label] == cover_marker
    else:
        raise ValueError(f"{start} and {end} should be a graph node and {start}-{end} a graph edge."
                         f"Also the edge should have a label ")


def mark_edge(start, end, graph, cover_marker, label):
    """ Mark an edge in the graph by the cover marker
    Parameters
    ---------
    start
    end
    graph
    cover_marker
    label
    """
    if label is not None and label in graph[start][end]['label']:
        if 'cover_mark' in graph[start][end]:
            graph[start][end]['cover_mark'][label] = cover_marker
        else:
            graph[start][end]['cover_mark'] = {label: cover_marker}

    else:
        raise ValueError("label shouldn't be empty or none and must be an edge label ")


def is_embedding_marked(embedding, pattern, graph, cover_marker):
    """ Check if an embedding is already marked
    Parameters
    ----------
    embedding
    pattern
    graph
    cover_marker

    Returns
    ---------
    bool
    """
    # keys = list(embedding.keys())
    # values = list(embedding.values())
    # i = 0
    res = []
    if len(utils.get_edge_in_embedding(embedding, pattern)):
        for edge in utils.get_edge_in_embedding(embedding, pattern):
            label = utils.get_edge_label(edge[0], edge[1], pattern)
            node1 = utils.get_key_from_value(embedding, edge[0])
            node2 = utils.get_key_from_value(embedding, edge[1])
            if (node1, node2) in list(graph.edges(node1)):
                if ('cover_mark' in graph[node1][node2]
                        and label in graph[node1][node2]['cover_mark']
                        and is_edge_marked(node1, node2, graph, cover_marker, label)):
                    res.append(True)
            elif (node2, node1) in list(graph.edges(node2)):
                if ('cover_mark' in graph[node2][node1]
                        and label in graph[node2][node1]['cover_mark']
                        and is_edge_marked(node2, node1, graph, cover_marker, label)):
                    res.append(True)
            else:
                res.append(False)
        return True in res
    else:
        return ValueError("This embedding should have edges")


def mark_embedding(embedding, graph, pattern, cover_marker):
    """ Mark an embedding of a pattern in the graph with a given cover_marker
    Parameters
    -----------
    embedding
    graph
    pattern
    cover_marker
    """

    keys = list(embedding.keys())
    values = list(embedding.values())
    i = 0
    if len(utils.get_edge_in_embedding(embedding, pattern)):
        for edge in utils.get_edge_in_embedding(embedding, pattern):
            label = utils.get_edge_label(edge[0], edge[1], pattern)
            node1 = utils.get_key_from_value(embedding, edge[0])
            node2 = utils.get_key_from_value(embedding, edge[1])
            if (node1, node2) in list(graph.edges(node1)):
                mark_edge(node1, node2, graph, cover_marker, label)
            elif (node2, node1) in list(graph.edges(node2)):
                mark_edge(node2, node1, graph, cover_marker, label)

            if 'label' in pattern.nodes(data=True)[edge[0]]:
                mark_node(node1, graph, cover_marker, pattern.nodes(data=True)[edge[0]]['label'])

            if 'label' in pattern.nodes(data=True)[edge[1]]:
                mark_node(node2, graph, cover_marker, pattern.nodes(data=True)[edge[1]]['label'])

    else:
        return ValueError("This embedding should have edges")


def get_node_label_number(node_number, graph):
    """ compute the number of label for a given node_number in the graph
    Parameters
    ----------
    node_number
    graph
    Returns
    ---------
    int
    """
    if node_number in graph.nodes():
        if 'label' in graph.nodes(data=True)[node_number]:
            return len(graph.nodes(data=True)[node_number]['label'])
        else:
            return 0
    else:
        raise ValueError(f"{node_number} should be a graph node")


def search_port(graph, embedding, cover_marker, pattern, port_usage):
    """ Search all node who are port for data or for pattern
    Parameters
    ----------
    graph
    embedding
    cover_marker
    port_usage
    pattern
    Returns
    -------
    tuple
    """
    res = []
    # Search pattern ports
    keys = list(embedding.keys())  # node number in graph
    values = list(embedding.values())  # node number in pattern
    i = 0
    while i <= len(keys) - 1:
        if not is_node_edges_marked(graph, keys[i], pattern, values[i]) \
                or not is_node_all_labels_marked(keys[i], graph, cover_marker):
            res.append((keys[i], values[i]))
            if 'port' not in graph.nodes[keys[i]]:  # if the node is not already marked as port
                graph.nodes[keys[i]]['port'] = True  # mark it

            if values[i] in port_usage:
                port_usage[values[i]] = port_usage[values[i]] + 1
            else:
                port_usage[values[i]] = 1
        i += 1

    return res


def is_node_edges_marked(graph, graph_node, pattern, pattern_node):
    """ Check if the graph and the pattern corresponding node are the same edges number
    Parameters
    ----------
    graph
    graph_node
    pattern
    pattern_node

    Returns
    ---------
    bool
    """
    if graph_node in graph.nodes() and pattern_node in pattern.nodes():
        if graph.edges(graph_node) is not None and pattern.edges(pattern_node) is not None:
            return len(graph.edges(graph_node)) == len(pattern.edges(pattern_node))
        else:
            return True
    else:
        raise ValueError(f"{graph_node} should be a nod of the graph and {pattern_node}, a node of pattern")


def is_node_all_labels_marked(node_number, graph, cover_marker):
    """ Check if all node labels are marked by the cover_marker
    Parameters
    ----------
    node_number
    graph
    cover_marker

    Returns
    --------
    bool"""

    if node_number in graph.nodes():
        if 'label' in graph.nodes(data=True)[node_number]:
            response = []
            for label in graph.nodes(data=True)[node_number]['label']:
                response.append(is_node_marked(node_number, graph, cover_marker, label))

            return not (False in response)
        else:
            return True
    else:
        raise ValueError(f"{node_number} should be a graph node and must have a label ")


def create_pattern_node(rewritten_graph, pattern_number, ports):
    """ Add a new node with its edges in the rewritten graph who represents a pattern embedding
       Parameters
       ----------
       rewritten_graph
       pattern_number : the pattern row number in the code table rows
       ports
       """
    if len(ports) == 1:
        last_node = len(rewritten_graph.nodes())
        rewritten_graph.add_node(last_node + 1, label=f"P{pattern_number}", is_Pattern=True)
        create_rewrite_edge(rewritten_graph, last_node + 1, ports[0][0], pattern_port=ports[0][1])
    elif len(ports) > 1:
        last_node = len(rewritten_graph.nodes())
        node = last_node + 1
        rewritten_graph.add_node(node, label=f"P{pattern_number}", is_Pattern=True)
        for p in ports:
            create_rewrite_edge(rewritten_graph, node, p[0], pattern_port=p[1])
    else:
        last_node = len(rewritten_graph.nodes())
        rewritten_graph.add_node(last_node + 1, label=f"P{pattern_number}", is_Pattern=True)


def create_vertex_singleton_node(rewritten_graph, node_label, node_number):
    """ Add a new node with its edges in the rewritten graph who represents a vertex singleton embedding
       Parameters
       ----------
       rewritten_graph
       node_label
       node_number
       """
    last_node = len(rewritten_graph.nodes())
    rewritten_graph.add_node(last_node + 1, label=f"{node_label}", is_Pattern=True, is_singleton=True)
    create_rewrite_edge(rewritten_graph, last_node + 1, node_number, pattern_port=1)


def create_edge_singleton_node(rewritten_graph, edge_label, first_node, second_node):
    """ Add a new node with its edges in the rewritten graph who represents an edge singleton embedding
    Parameters
    ----------
    rewritten_graph
    edge_label
    first_node
    second_node
    """
    last_node = len(rewritten_graph.nodes())
    rewritten_graph.add_node(last_node + 1, label=f"{edge_label}", is_Pattern=True, is_singleton=True)
    create_rewrite_edge(rewritten_graph, last_node + 1, first_node, pattern_port=1)
    create_rewrite_edge(rewritten_graph, last_node + 1, second_node, pattern_port=2)


def create_rewrite_edge(rewritten_graph, rewritten_node, data_node, **kwargs):
    """ Add a new edge between a node and a port in the rewritten graph
    Parameters
    -----------
    rewritten_graph
    rewritten_node
    data_node : the port node in the initial data graph
    """
    if str(data_node) in dict(rewritten_graph.nodes(data='label')).values():
        port_node = utils.get_key_from_value(dict(rewritten_graph.nodes(data='label')), str(data_node))
        rewritten_graph.add_edge(rewritten_node, port_node, label=f"v{kwargs['pattern_port']}")
    else:
        last = len(rewritten_graph.nodes())
        rewritten_graph.add_node(last + 1, label=f"{data_node}")
        rewritten_graph.add_edge(rewritten_node, last + 1, label=f"v{kwargs['pattern_port']}")


def row_cover(row: CodeTableRow, graph, cover_marker, rewritten_graph, row_number, timeout=0):
    """ Cover a code table row on the graph with a given marker
    Parameters
    ----------
    row
    graph
    cover_marker
    rewritten_graph
    row_number
    timeout
    """
    cover_usage = 0
    port_usage = dict()
    if not utils.is_without_edge(row.pattern()):
        for embedding in row.embeddings():
            if not is_embedding_marked(embedding, row.pattern(), graph, cover_marker):
                mark_embedding(embedding, graph, row.pattern(), cover_marker)
                ports = search_port(graph, embedding, cover_marker, row.pattern(), port_usage)
                cover_usage = cover_usage + 1
                # create the pattern in the rewritten graph
                create_pattern_node(rewritten_graph, row_number, ports)
    else:
        for embedding in row.embeddings():
            keys = list(embedding.keys())
            values = list(embedding.values())
            i = 0
            while i <= len(keys) - 1:
                label = row.pattern().nodes(data=True)[values[i]]['label']
                # check if all pattern node labels are marked in the graph node
                if not is_node_labels_marked(keys[i], graph, cover_marker, label):
                    ports = search_port(graph, embedding, cover_marker, row.pattern(), port_usage)
                    cover_usage = cover_usage + 1
                    mark_node(keys[i], graph, cover_marker, label)
                    # create the pattern in the rewritten graph
                    create_pattern_node(rewritten_graph, row_number,
                                        ports)  # with experimental rewritten graph implementation

                i = i + 1

    row.set_pattern_port_usage(port_usage)
    row.set_pattern_usage(cover_usage)


def singleton_cover(graph, cover_marker, rewritten_graph):
    """ Cover the code table with the singleton pattern
    Parameters
    -----------
    graph
    cover_marker
    rewritten_graph
    Returns
    --------
    tuple : who contains first dictionary for vertex usage and other for edge usage
    """
    edge_singleton_usage = defaultdict(int)
    vertex_singleton_usage = defaultdict(int)

    # for edge label
    for edge in graph.edges(data=True):
        if not is_edge_marked(edge[0], edge[1], graph, cover_marker, edge[2]['label']):
            edge_singleton_usage[edge[2]['label']] += 1
            mark_edge(edge[0], edge[1], graph, cover_marker, edge[2]['label'])  # Mark the edge
            # Mark the edge nodes as data port
            graph.nodes[edge[0]]['port'] = True
            graph.nodes[edge[1]]['port'] = True

            # create the singleton embedding in the rewritten graph
            create_edge_singleton_node(rewritten_graph, edge[2]['label'], edge[0], edge[1])
    # for node label
    for node in graph.nodes(data=True):
        if 'label' in node[1]:
            if type(node[1]['label']) is not str:
                for label in node[1]['label']:
                    if not is_node_marked(node[0], graph, cover_marker, label):
                        vertex_singleton_usage[label] += 1
                        mark_node(node[0], graph, cover_marker, label)
                        graph.nodes[node[0]]['port'] = True  # mark the node as port
                        # create the singleton embedding in the rewritten graph
                        create_vertex_singleton_node(rewritten_graph, label, node[0])
            else:
                if not is_node_marked(node[0], graph, cover_marker, node[1]['label']):
                    vertex_singleton_usage[node[1]['label']] += 1
                    mark_node(node[0], graph, cover_marker, node[1]['label'])
                    graph.nodes[node[0]]['port'] = True  # mark as port
                    # create the singleton embedding in the rewritten graph
                    create_vertex_singleton_node(rewritten_graph, node[1]['label'], node[0])

    return vertex_singleton_usage, edge_singleton_usage


class CodeTable:
    """
        Code table inspired by Krimp algorithm
        It's composed by the non singleton pattern and their information (usage, ports) who represented by a row
        And the singleton pattern stored in a dictionary if there is one
        The separation between singleton and non-singleton pattern make easily their treatments and reduces the execution time.
    """

    def __init__(self, standard_table, graph):
        self._standard_table = standard_table  # A initial code table
        self._rows = []  # All rows of the code table
        self._description_length = 0.0  # the description length of this code table
        self._data_graph = graph  # The graph where we want to apply the code table elements
        """ We don't store singleton pattern ports because each singleton pattern node is its port,
            then the vertex singleton ports are {1:1}, node 1 and usage 1,
            and the edge singleton, {1:1, 2:1}, node 1, usage 1, node 2, usage 1.
            Thus singleton ports are intuitive 
        """
        self._vertex_singleton_usage = dict()  # singleton vertex usage
        self._edge_singleton_usage = dict()  # singleton edge usage
        self._singleton_code_length = dict()  # singleton pattern code length
        self._rewritten_graph = nx.DiGraph()  # A directed graph who represents the rewritten graph

    def add_row(self, row: CodeTableRow):
        """ Add a new row at the code table
        Parameters
        ----------
        row
        """
        self._rows.append(row)  # Add the new row at the end of the list

        # compute the row pattern embeddings and store them
        row.set_embeddings(utils.get_embeddings(row.pattern(), self._data_graph))

        # sort the list who contains the row according a reverse order and a specific key
        self._rows.sort(reverse=True, key=_order_rows)

    def remove_row(self, row: CodeTableRow):
        """ Remove a row on the code table rows
        Parameters
        ---------
        row
        """
        del self._rows[self.rows().index(row)]
        self._rows.sort(reverse=True, key=_order_rows)

    def rows(self):
        """ Provide code table rows
        Returns
        -------
        list
        """
        return self._rows

    def cover(self, timeout=0, debug=False):
        """ Make the cover for the code table,
            the cover marker is get from the data graph
        """
        b = time.time()
        # Get the cover marker and increment it
        if 'cover_marker' in self._data_graph.graph:
            self._data_graph.graph['cover_marker'] += 1
        else:
            self._data_graph.graph['cover_marker'] = 1

        cover_marker = self._data_graph.graph['cover_marker']  # the current cover marker

        self._rewritten_graph = nx.DiGraph()  # reset the rewritten graph before each cover

        for row in self._rows:
            row_cover(row, self._data_graph, cover_marker, self._rewritten_graph, self.rows().index(row), timeout)

        res = singleton_cover(self._data_graph, cover_marker, self._rewritten_graph)

        self._vertex_singleton_usage = res[0]  # Store vertex singleton usage
        self._edge_singleton_usage = res[1]  # Store edge singleton usage

        usage_sum = self._compute_usage_sum()  # Get the total of the rows usage
        # compute each row code length and description length
        for row in self._rows:
            row.compute_code_length(usage_sum)
            row.compute_description_length(self._standard_table)

        self._compute_singleton_code(usage_sum)  # compute singleton code length
        if debug:
            print(f"cover time...{time.time() - b}")

    def compute_description_length(self):
        """ Compute the total description length
        Returns
        -------
        float
        """
        self.compute_ct_description_length()

        return self._description_length + self.compute_rewritten_graph_description()

    def compute_ct_description_length(self):
        """ Compute this code table description length """
        if len(self._rewritten_graph.nodes()) != 0:  # Check if the cover is done
            # check if the code table is already covered
            description_length = 0.0
            for row in self._rows:
                if row.pattern_usage() != 0:
                    description_length += row.description_length()

            description_length += self._compute_singleton_description_length()

            self._description_length = description_length
        else:
            raise ValueError("You should cover the code table before computing his description")

    def compute_rewritten_graph_description(self):
        """ Compute description_length of the rewritten graph
        Returns
        -------
        float
        """
        if len(self._rewritten_graph.nodes()) != 0:
            desc = 0.0
            for node in self._rewritten_graph.nodes(data=True):
                if 'is_Pattern' in node[1] and 'is_singleton' not in node[1]:
                    row_number = int(node[1]['label'].split('P')[1])
                    embed_port = utils.get_port_node(self._rewritten_graph, node[0])
                    desc += self._compute_embedding_pattern_description(row_number, embed_port)
                elif 'is_Pattern' in node[1] and 'is_singleton' in node[1]:
                    desc += self._compute_embedding_singleton_description(node[1]['label'])
                else:
                    desc += 0.0

            return desc
        else:
            raise ValueError("You should first cover the code table")

    def description_length(self):
        """ Provide the code table description length
        Returns
        -------
        float
        """
        return self._description_length

    def rewritten_graph(self):
        """ Provide the code table rewritten graph
        Returns
        -------
        object
        """
        return self._rewritten_graph

    def label_codes(self):
        """ Provide the label codes
        Returns
        -------
        object
        """
        return self._standard_table

    def is_ct_edge_singleton(self, label):
        """ provide if a given label is an edge singleton label
        Parameters
        ----------
        label
        Returns
        -------
        bool
        """
        return label in self._edge_singleton_usage

    def is_ct_vertex_singleton(self, label):
        """ provide if a given label is a vertex singleton label
        Parameters
        ----------
        label
        Returns
        -------
        bool
        """
        return label in self._vertex_singleton_usage

    def data_port(self):
        """ Provide all graph data port
        Returns
        -------
        list
        """
        data_port = []
        for node in self._data_graph.nodes(data=True):
            if 'port' in node[1]:
                data_port.append(node[0])

        return data_port

    def _compute_usage_sum(self):
        """ Compute the total of usage for this code table elements
        Returns
        -------
        float : the usage sum """
        usage_sum = 0.0
        for row in self._rows:
            usage_sum += row.pattern_usage()

        if len(self._vertex_singleton_usage.keys()) != 0:
            for value in self._vertex_singleton_usage.values():
                usage_sum += value

        if len(self._edge_singleton_usage.keys()) != 0:
            for value in self._edge_singleton_usage.values():
                usage_sum += value

        return usage_sum

    def _compute_singleton_code(self, usage_sum):
        """ Compute singleton pattern code length after cover
        Parameters
        ----------
        usage_sum : Total of pattern ( even singleton) usage in the code table
        """
        self._singleton_code_length = dict()
        if len(self._vertex_singleton_usage) != 0:
            for u, v in self._vertex_singleton_usage.items():
                if utils.log2(v, usage_sum) == -0.0:
                    self._singleton_code_length[u] = 0.0
                else:
                    self._singleton_code_length[u] = utils.log2(v, usage_sum)

        if len(self._edge_singleton_usage) != 0:
            for u, v in self._edge_singleton_usage.items():
                if utils.log2(v, usage_sum) == -0.0:
                    self._singleton_code_length[u] = 0.0
                else:
                    self._singleton_code_length[u] = utils.log2(v, usage_sum)

    def _compute_singleton_description_length(self):
        """ Compute the sum of each singleton pattern description length
        Returns
        -------
        float
        """
        desc = 0.0
        if len(self._vertex_singleton_usage.keys()) != 0:
            for key in self._vertex_singleton_usage.keys():
                desc += utils.encode_singleton(self._standard_table, 1, key)
                desc += self._singleton_code_length[key]
                # port description, the singleton have one port and one node
                # Then we have 1 + len(nodes) = 2 and log2(binomial(1,1)), port code is 0
                desc += math.log2(2)

        if len(self._edge_singleton_usage.keys()) != 0:
            for key in self._edge_singleton_usage.keys():
                desc += utils.encode_singleton(self._standard_table, 2, key)
                desc += self._singleton_code_length[key]
                # port description, the singleton have two ports and two nodes
                # Then we have 1 + len(nodes) = 3 and log2(binomial(2,2)), port code is {1:1, 2:1}
                # Then sum = 2
                desc += math.log2(3) + 2

        return desc

    def _compute_embedding_pattern_description(self, row_number, embed_port):
        """ Compute description length for pattern embedding in the rewritten graph
        Parameters
        ----------
        row_number : The pattern row
        embed_port: The embedding ports
        Returns
        ---------
        float
        """
        if len(self._rows) - 1 >= row_number >= 0:
            desc = 0.0
            row = self._rows[row_number]
            desc += row.code_length()
            desc += math.log2(len(row.port_code_length()) + 1)

            code_port_total = 0.0
            for p in embed_port:
                if p in row.port_code_length():
                    code_port_total += row.port_code_length()[p]

            desc += code_port_total
            desc += math.log2(utils.binomial(len(self._data_graph.nodes()), len(embed_port)))
            return desc
        else:
            raise ValueError("The row number is out of the bounds")

    def _compute_embedding_singleton_description(self, label):
        """ Compute description_length for singleton embedding in the rewritten graph
        Parameters
        ----------
        label : The singleton label
        Returns
        -------
        float
        """
        desc = 0.0
        if label in self._vertex_singleton_usage:
            desc += self._singleton_code_length[label]
            desc += math.log2(2)  # port_count = 2 and total_port_code = 0
            desc += math.log2(utils.binomial(len(self._data_graph.nodes()), 1))
            return desc
        elif label in self._edge_singleton_usage:
            desc += self._singleton_code_length[label]
            desc += math.log2(3)  # port_count = 2
            desc += 2  # total_port_code = 1.0 + 1.0
            desc += math.log2(utils.binomial(len(self._data_graph.nodes()), 2))
            return desc
        else:
            raise ValueError("The label should be a data graph label")

    def singleton_code_length(self):
        return self._singleton_code_length

    def display_ct(self):
        data = [row.display_row() for row in self._rows if row.pattern_usage() != 0]
        for u, v in self._singleton_code_length.items():
            data.append([u, v, '', '', '', ''])
        return pandas.DataFrame(numpy.array(data),
                                columns=['Pattern Structure', 'Pattern usage', 'Pattern code length',
                                         'Port count', 'Pattern port usage',
                                         'Pattern port code length'])

    def __str__(self) -> str:
        msg = "\n Pattern |usage |code_length |port_count |port_usage |port_code \n"
        for row in self._rows:
            msg += str(row) + "\n"

        for u, v in self._singleton_code_length.items():
            msg += "{}|{}|{}".format(u, v, self._singleton_code_length[u]) + "\n"

        return msg
