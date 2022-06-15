from collections import defaultdict

from .code_table_row import CodeTableRow
import skmine.graph.graphmdl.utils as utils


def _order_rows(row: CodeTableRow):
    """ Provide the row pattern_code to order rows
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
    if node_number in graph.nodes() and \
            label in graph.nodes(data=True)[node_number]['label']:
        node = graph.nodes(data=True)[node_number]
        if 'cover_mark' in node \
                and label in node['cover_mark'] \
                and node['cover_mark'][label] == cover_marker:

            return True

        else:
            return False
    else:
        raise ValueError(f"{node_number} must be a graph node and {label} a node label")


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
        res = list()
        for label in labels:
            res.append(is_node_marked(node_number, graph, cover_marker, label))

        return not (False in res)
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
        raise ValueError("label mustn't be empty or none")


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
    if start in graph.nodes() and \
            end in graph.nodes() and \
            graph[start][end] is not None and 'label' in graph[start][end]:

        edge = graph[start][end]
        if 'cover_mark' in edge \
                and label in edge['cover_mark'] \
                and edge['cover_mark'][label] == cover_marker:
            return True
        else:
            return False
    else:
        raise ValueError(f"{start} and {end} must be a graph node and {start}-{end} a graph edge."
                         f"Also the edge must have a label ")


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
        raise ValueError("label mustn't be empty or none and must be an edge label ")


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
    keys = list(embedding.keys())
    values = list(embedding.values())
    i = 0
    res = []
    while i < len(keys) - 1:
        if i != len(keys):
            label = utils.get_edge_label(values[i], values[i + 1], pattern)
            if 'cover_mark' in graph[keys[i]][keys[i + 1]] \
                    and label in graph[keys[i]][keys[i + 1]]['cover_mark'] \
                    and is_edge_marked(keys[i], keys[i + 1], graph, cover_marker, label):
                res.append(True)
            else:
                res.append(False)
        i = i + 1
    return True in res


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
    while i <= len(keys) - 1:
        if i != len(keys) - 1:
            mark_edge(keys[i], keys[i + 1], graph, cover_marker, pattern[values[i]][values[i + 1]]['label'])

            if 'label' in pattern.nodes(data=True)[values[i]]:
                mark_node(keys[i], graph, cover_marker, pattern.nodes(data=True)[values[i]]['label'])

        else:
            if 'label' in pattern.nodes(data=True)[values[i]]:
                mark_node(keys[i], graph, cover_marker, pattern.nodes(data=True)[values[i]]['label'])

        i = i + 1


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
        raise ValueError(f"{node_number} must be a graph node")


def search_port(graph, embedding, cover_marker, port_usage):
    """ Search all node who are port for data or for pattern
    Parameters
    ----------
    graph
    embedding
    cover_marker
    port_usage
    Returns
    -------
    dict
    """
    # Search pattern ports
    keys = list(embedding.keys())  # node number in graph
    values = list(embedding.values())  # node number in pattern
    i = 0
    while i <= len(keys) - 1:
        if not is_node_edges_marked(graph, keys[i], cover_marker) \
                or not is_node_all_labels_marked(keys[i], graph, cover_marker):

            if 'port' not in graph.nodes[keys[i]]:  # if the node is not already marked as port
                graph.nodes[keys[i]]['port'] = True  # mark it

            if values[i] in port_usage:
                port_usage[values[i]] = port_usage[values[i]] + 1
            else:
                port_usage[values[i]] = 1
        i += 1

    return port_usage


def is_node_edges_marked(graph, node_number, cover_marker):
    """ Check if all edges of a given node in a given graph are covered
    Parameters
    ----------
    graph
    node_number
    cover_marker

    Returns
    ---------
    bool
    """
    if len(graph.edges(node_number)) != 0:  # check if the node have edge
        res = []
        for edge in graph.edges(node_number):
            label = utils.get_edge_label(edge[0], edge[1], graph)
            if 'cover_mark' in graph[edge[0]][edge[1]]:
                if graph[edge[0]][edge[1]]['cover_mark'][label] == cover_marker:
                    res.append(True)
                else:
                    res.append(False)
            else:
                res.append(False)
        return not (False in res)
    else:
        return True


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
        raise ValueError(f"{node_number} must be a graph node and must have a label ")


def row_cover(row: CodeTableRow, graph, cover_marker):
    """ Cover a code table row on the graph with a given marker
    Parameters
    ----------
    row
    graph
    cover_marker
    """
    cover_usage = 0
    port_usage = dict()
    if not utils.is_without_edge(row.pattern()):
        for embedding in row.embeddings():
            if not is_embedding_marked(embedding, row.pattern(), graph, cover_marker):
                mark_embedding(embedding, graph, row.pattern(), cover_marker)
                search_port(graph, embedding, cover_marker, port_usage)
                cover_usage = cover_usage + 1

    else:
        for embedding in row.embeddings():
            keys = list(embedding.keys())
            values = list(embedding.values())
            i = 0
            while i <= len(keys) - 1:
                label = row.pattern().nodes(data=True)[values[i]]['label']
                # check if all pattern node labels are marked in the graph node
                if not is_node_labels_marked(keys[i], graph, cover_marker, label):
                    search_port(graph, embedding, cover_marker, port_usage)
                    cover_usage = cover_usage + 1
                    mark_node(keys[i], graph, cover_marker, label)

                i = i + 1

    row.set_pattern_port_code(port_usage)
    row.set_pattern_code(cover_usage)


def singleton_cover(graph, cover_marker):
    """ Cover the code table with the singleton pattern
    Parameters
    -----------
    graph
    cover_marker
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
            mark_edge(edge[0], edge[1], graph, cover_marker, edge[2]['label'])
    # for node label
    for node in graph.nodes(data=True):
        if 'label' in node[1]:
            for label in node[1]['label']:
                if not is_node_marked(node[0], graph, cover_marker, label):
                    vertex_singleton_usage[label] += 1
                    mark_node(node[0], graph, cover_marker, label)

    return vertex_singleton_usage, edge_singleton_usage


class CodeTable:
    """
        Code table inspired by Krimp algorithm
    """

    def __init__(self, standard_table, graph):
        self._standard_table = standard_table  # A initial code table
        self._rows = []  # All rows of the code table
        self._description_length = 0.0  # the description length of this code table
        self._data_graph = graph  # The graph where we want to apply the code table elements
        self._vertex_singleton_usage = dict()
        self._edge_singleton_usage = dict()

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

    def rows(self):
        """ Provide code table rows
        Returns
        -------
        list
        """
        return self._rows

    def cover(self, cover_marker):
        """ Make the cover for the code table
        Parameters
        ----------
        cover_marker
        """
        for row in self._rows:
            row_cover(row, self._data_graph, cover_marker)

        res = singleton_cover(self._data_graph, cover_marker)
        self._vertex_singleton_usage = res[0]
        self._edge_singleton_usage = res[1]

        # compute each row code length and description length
        usage_sum = self._compute_usage_sum()
        for row in self._rows:
            row.compute_code_length(usage_sum)
            row.compute_description_length(self._standard_table)

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
        float : the usage sum"""
        usage_sum = 0.0
        for row in self._rows:
            usage_sum += row.pattern_code()

        if len(self._vertex_singleton_usage.keys()) != 0:
            for value in self._vertex_singleton_usage.values():
                usage_sum += value

        if len(self._edge_singleton_usage.keys()) != 0:
            for value in self._edge_singleton_usage.values():
                usage_sum += value

        return usage_sum

    def _display_row(self, row: CodeTableRow):
        msg = "{} |{} |{} |{} |{} |{}" \
            .format(row.pattern(), row.pattern_code(), row.code_length(),
                    len(row.pattern_port_code()), row.pattern_port_code(),
                    row.port_code_length())
        return msg

    def __str__(self) -> str:
        msg = "\n Pattern |usage |code_length |port_count |port_usage |port_code \n"
        for row in self._rows:
            msg += self._display_row(row) + "\n"

        return msg
