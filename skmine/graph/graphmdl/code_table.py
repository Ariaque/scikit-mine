from code_table_row import CodeTableRow
import utils


def _order_rows(row: CodeTableRow):
    """ Provide the row pattern_code to order rows
    Parameters
    ----------
    row
    Returns
    --------
    float
    """
    return row.pattern_code()


class CodeTable:
    """
        Code table inspired by Krimp algorithm
    """

    def __init__(self, standard_table, graph):
        self._standard_table = standard_table  # A initial code table
        self._rows = []  # All rows of the code table
        self._description_length = 0.0  # the description length of this code table
        self._data_graph = graph    # The graph where we want to apply the code table elements

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


