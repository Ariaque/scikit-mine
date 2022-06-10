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
    """ Check if a node is already marked
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
    while i < len(keys) - 1:
        if i != len(keys):
            label = utils.get_edge_label(values[i], values[i + 1], pattern)
            if 'cover_mark' in graph[keys[i]][keys[i + 1]] \
                    and label in graph[keys[i]][keys[i + 1]]['cover_mark'] \
                    and is_edge_marked(keys[i], keys[i + 1], graph, cover_marker, label):
                return True
            else:
                return False
        i = i + 1


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


def search_data_port(graph, pattern, embedding):
    """ Search all node who are port for data
    Parameters
    ----------
    graph
    pattern
    embedding
    """
    keys = list(embedding.keys())
    values = list(embedding.values())
    i = 0
    while i <= len(keys) - 1:  # get all node in the graph
        if 'cover_mark' in graph.nodes[keys[i]]:  # check if the node is marked
            # if the node is marked, compare the number of marked label
            # and the number of pattern node label
            # The node is a port if the two number are different
            if get_node_label_number(values[i], pattern) != len(graph.nodes[keys[i]]['cover_mark']):
                if 'port' not in graph.nodes[keys[i]]:  # if the node is not already marked as port
                    graph.nodes[keys[i]]['port'] = True  # mark it
        i = i + 1


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
        for edge in graph.edges(node_number):
            label = utils.get_edge_label(edge[0], edge[1], graph)
            if 'cover_mark' in graph[edge[0]][edge[1]]:
                if graph[edge[0]][edge[1]]['cover_mark'][label] == cover_marker:
                    return True
                else:
                    return False
            else:
                return False
    else:
        return True


def is_node_labels_marked(node_number, graph, cover_marker):
    """ Check if all node labels are marked by the cover_marker
    Parameters
    ----------
    node_number
    graph
    cover_marker

    Returns
    --------
    bool"""

    if node_number in graph.nodes() and 'label' in graph.nodes(data=True)[node_number]:
        response = False
        for label in graph.nodes(data=True)[node_number]['label']:
            response = is_node_marked(node_number, graph, cover_marker, label)

        return response
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
                cover_usage = cover_usage + 1

            # Search pattern ports
            keys = embedding.keys()
            for i in keys:
                if not is_node_edges_marked(graph, i, cover_marker):
                    if i in port_usage:
                        port_usage[i] = port_usage[i] + 1
                    else:
                        port_usage[i] = 1
            search_data_port(graph, row.pattern(), embedding)  # search data ports
    else:
        for embedding in row.embeddings():
            keys = list(embedding.keys())
            values = list(embedding.values())
            i = 0
            while i <= len(keys) - 1:
                for label in row.pattern().nodes(data=True)[values[i]]['label']:
                    if not is_node_marked(keys[i], graph, cover_marker, label):
                        mark_node(keys[i], graph, cover_marker, label)
                        cover_usage = cover_usage + 1
                i = i + 1
            # Search pattern ports
            for n in keys:
                if not is_node_labels_marked(n, graph, cover_marker):
                    if n in port_usage:
                        port_usage[n] = port_usage[n] + 1
                    else:
                        port_usage[n] = 1

            search_data_port(graph, row.pattern(), embedding)  # search data ports

    row.set_pattern_port_code(port_usage)
    row.set_pattern_code(cover_usage)


class CodeTable:
    """
        Code table inspired by Krimp algorithm
    """

    def __init__(self, standard_table, graph):
        self._standard_table = standard_table  # A initial code table
        self._rows = []  # All rows of the code table
        self._description_length = 0.0  # the description length of this code table
        self._data_graph = graph  # The graph where we want to apply the code table elements

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

    def data_port(self):
        """ Provide all graph data port
        Returns
        -------
        list
        """
        data_port = []
        for node in self._data_graph.nodes(data=True):
            if 'port' in node:
                data_port.append(node[0])

        return data_port
