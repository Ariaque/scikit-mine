from code_table_row import CodeTableRow
import utils


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
    node = graph.nodes(data=True)[node_number]
    if 'cover_mark' in node \
            and label in node['cover_mark'] \
            and node['cover_mark'][label] == cover_marker:
        return True
    else:
        return False


def mark_node(node_number, graph, cover_marker, pattern):
    """ Mark a node in the graph by the cover marker
    Parameters
    ---------
    node_number
    graph
    cover_marker
    pattern
    """
    label = pattern.nodes(data=True)[node_number]['label']
    if 'cover_mark' in graph.nodes(data=True)[node_number]:
        graph.nodes(data=True)[node_number]['cover_mark'][label] = cover_marker
    else:
        graph.nodes(data=True)[node_number]['cover_mark'] = {label: cover_marker}


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
    edge = graph[start][end]
    if 'cover_mark' in edge \
            and label in edge['cover_mark'] \
            and edge['cover_mark'][label] == cover_marker:
        return True
    else:
        return False


def mark_edge(start, end, graph, cover_marker, pattern):
    """ Mark an edge in the graph by the cover marker
    Parameters
    ---------
    start
    end
    graph
    cover_marker
    pattern
    """
    label = pattern[start][end]['label']
    if 'cover_mark' in graph[start][end]:
        graph[start][end]['cover_mark'][label] = cover_marker
    else:
        graph[start][end]['cover_mark'] = {label: cover_marker}


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
    i = 0
    while i < len(keys) - 1:
        if i != len(keys):
            label = utils.get_edge_label(embedding[i], embedding[i + 1], 0, pattern)
            if 'cover_mark' in graph[i][i + 1] \
                    and label in graph[i][i + 1]['cover_mark'] \
                    and graph[i][i + 1]['cover_mark'][label] == cover_marker:
                return True
            else:
                return False
        i = i + 1


def mark_embedding(embedding, graph, pattern, cover_marker):
    keys = list(embedding.keys())
    i = 0
    while i < len(keys) - 1:
        if i != len(keys):
            mark_edge(i, i + 1, graph, cover_marker, pattern)
            mark_node(i, graph, cover_marker, pattern)
        else:
            mark_node(i, graph, cover_marker, pattern)

        i = i + 1


def row_cover(row: CodeTableRow, graph, cover_marker):
    cover_usage = 0
    if not utils.is_without_edge(row.pattern()):
        for embedding in row.embeddings():
            if not is_embedding_marked(embedding, row.pattern(), graph, cover_marker):
                mark_embedding(embedding, graph, row.pattern(), cover_marker)
                usage = usage + 1
    else:
        raise NotImplemented()


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
        # sort the list who contains the row according a reverse order and a specific key
        self._rows.sort(reverse=True, key=_order_rows)

        # compute the row pattern embeddings and store them
        row.set_embeddings(utils.get_embeddings(row.pattern(), self._data_graph))
