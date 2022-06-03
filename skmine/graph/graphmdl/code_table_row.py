class CodeTableRow:
    """
        Object to represent a row of the code table
    """

    def __init__(self, pattern, pattern_code=None, pattern_port_code=None):
        self._pattern = pattern
        self._pattern_code = pattern_code
        self._pattern_port_code = pattern_port_code
        self._embeddings = []

    def pattern(self):
        """ Provide the row pattern
        Returns
        -------
        object
        """
        return self._pattern

    def pattern_code(self):
        """ Provide the row pattern code
        Returns
        -------
        float
        """
        return self._pattern_code

    def set_pattern_code(self, pattern_code):
        """ Set the row pattern code
        Parameters
        ----------
        pattern_code
        """
        self._pattern_code = pattern_code

    def pattern_port_code(self):
        """ Provide the port code of the row pattern
        Returns
        -------
        dict
        """
        return self._pattern_port_code

    def set_embeddings(self, embeddings):
        """ Set the pattern row embeddings
        Parameters
        ----------
        embeddings
        """
        self._embeddings = embeddings

    def embeddings(self):
        """ Provide the pattern row embeddings
        Returns
        -------
        list
        """
        return self._embeddings

    # def add_port_code(self,):
