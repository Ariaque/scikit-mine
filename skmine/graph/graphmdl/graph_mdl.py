import copy
import logging
import time
import networkx
from skmine.base import BaseMiner
from skmine.graph.graphmdl.label_codes import LabelCodes
from skmine.graph.graphmdl.code_table import CodeTable
from skmine.graph.graphmdl.code_table_row import CodeTableRow
from skmine.graph.graphmdl import utils


def _order_pruning_rows(row):
    """ Sort the pruning row by their usage"""
    return row.pattern_usage()


class GraphMDL(BaseMiner):
    """
        Graph Minimum Description Length

        GraphMDL is an implementation in python of the GraphMDL+ algorithm make by Francesco Bariatti.
        It's exist moreover the Java implementation at 'https://gitlab.inria.fr/fbariatt/phd_project',
        also there is some functionalities not implemented here.

        It's an algorithm inspired by Krimp algorithm, and based on Minimum Description Length(MDL) principle
        to finding patterns in data represent by graphs, also it can be used as anytime algorithm

        Parameters
        -----------
        debug: bool, default=False
            Either to activate debug print or not

        References
        ----------
        Francesco Bariatti. GraphMDL+ - PHD. PhD thesis
        Francesco Bariatti, Peggy Cellier, and Sébastien Ferré. GraphMDL+ : interleaving
            the generation and MDL-based selection of graph patterns. In Proceedings of the
            36th Annual ACM Symposium on Applied Computing


    """

    def __init__(self, debug=False):
        self._data = None
        self._label_codes = None
        self._code_table = None
        self._rewritten_graph = None
        self._description_length = 0.0
        self._patterns = set()
        self._already_test = []  # list of the candidates already tested
        self._pruning_rows = []  # list of the rows who can prune
        self._debug = debug
        self._old_usage = dict()  # Mapper for the rows old usage, useful for the pruning
        self._timeout = 0
        self._initial_description_length = 0

    def _init_graph_mdl(self):
        """ Initialize the GraphMDL+ algorithm elements such as the label code,
            the initial code table (CT0), cover it to create the first rewritten graph"""
        if not self._debug:
            utils.MyLogger().setLevel(logging.WARNING)
        self._already_test = []
        self._label_codes = LabelCodes(self._data)  # label codes creation
        # CT0 creation
        self._code_table = CodeTable(self._label_codes, self._data)
        # CT0 cover
        self._code_table.cover(debug=self._debug)
        self._rewritten_graph = self._code_table.rewritten_graph()
        self._description_length = self._code_table.compute_kgmdl_total_description_length()
        self._initial_description_length = self._description_length
        utils.MyLogger().info(f"\n initial CT \n {self._code_table}")
        utils.MyLogger().info("GraphMDL+ run ...")
        utils.MyLogger().info(f"Initial description length = {round(self._description_length, 2)}")

    def fit(self, D, timeout=0):
        """ Fit GraphMDl+ on a given data graph
            Parameters
            ----------
            D : networkx graph where all edges are labeled
            timeout: int, default=0
            Maximum of the algorithm execution time
            It's useful for the anytime aspect of the algorithm.
            Returns
            -------
            GraphMDL
        """
        if timeout != 0:
            self._timeout = timeout
        if D is None:
            raise ValueError("You should give a graph")
        else:
            self._data = D
            self._init_graph_mdl()
            if self._timeout != 0:
                self._anytime_graph_mdl_with_timeout(self._timeout)
                return self
            else:
                self._graph_mdl()
                return self

    def _graph_mdl(self):
        """ Non-anytime graphmdl+ algorithm
        Returns
        -------
        GraphMDL
        """
        stop = False
        while not stop:
            utils.MyLogger().info("Candidate generation and sort start .....")
            b = time.time()
            candidates = utils.generate_candidates(self._rewritten_graph, self._code_table)
            candidates.sort(reverse=True, key=self._order_candidates)  # sort the candidates list
            utils.MyLogger().info(f"Candidate generation and sort end.....time ={time.time() - b}")
            utils.MyLogger().info(f"candidates number = {len(candidates)}")
            utils.MyLogger().info("GraphMDL+ best Ct search start .....")
            if len(candidates) != 0:
                b = time.time()
                for candidate in candidates:
                    if candidate not in self._already_test:
                        # Add a candidate to a ct, cover and compute description length
                        self._compute_old_usage()
                        row = CodeTableRow(candidate.final_pattern())
                        self._code_table.add_row(row)
                        self._code_table.cover(debug=self._debug)
                        temp_code_length = self._code_table.compute_total_description_length()
                        self._already_test.append(candidate)
                        # if the new ct is better than the old, break and generate new candidates
                        # with the new ct
                        if temp_code_length < self._description_length:
                            self._rewritten_graph = self._code_table.rewritten_graph()
                            self._description_length = temp_code_length
                            utils.MyLogger().info("GraphMDL+ new ct found.....")
                            utils.MyLogger().info(f"New description length = {self._description_length}")
                            utils.MyLogger().info(f"New pattern added : {utils.display_graph(row.pattern())}")
                            self._compute_pruning_candidates()
                            self._pruning()
                            utils.MyLogger().info(f"search time = {time.time() - b}")
                            del candidates
                            break
                        elif temp_code_length > self._description_length and candidates.index(candidate) == len(
                                candidates) - 1:
                            self._code_table.remove_row(row)
                            utils.MyLogger().info(f"Search time = {time.time() - b}")
                            utils.MyLogger().debug("None best code table found so stop the algorithm")
                            del candidates
                            stop = self._graph_mdl_end()
                            break
                        else:
                            self._code_table.remove_row(row)
                    else:
                        utils.MyLogger().debug("Already test")
            else:
                stop = self._graph_mdl_end()

        return self

    def _anytime_graph_mdl_with_timeout(self, timeout):
        """ Anytime graph mdl with timeout
        Parameters
        ----------
        timeout:int
            Maximum execution time
        Returns
        --------
        GraphMDL
        """
        begin = time.time()
        current = 0
        while current < timeout:
            utils.MyLogger().info("Candidate generation and sort start .....")
            b = time.time()
            candidates = utils.generate_candidates(self._rewritten_graph, self._code_table)
            candidates.sort(reverse=True, key=self._order_candidates)
            utils.MyLogger().info(f"Candidate generation and sort end ..........time ={time.time() - b}")
            utils.MyLogger().info(f"candidates number {len(candidates)}")
            utils.MyLogger().info("GraphMDL best Ct search start .....")
            current = time.time() - begin
            if self._stop_by_time(current, timeout):
                break
            if len(candidates) != 0:
                b = time.time()
                for candidate in candidates:
                    if candidate not in self._already_test:
                        # Add a candidate to a ct, cover and compute description length
                        current = time.time() - begin
                        if self._stop_by_time(current, timeout):
                            break
                        self._compute_old_usage()
                        row = CodeTableRow(candidate.final_pattern())
                        self._code_table.add_row(row)
                        self._code_table.cover()
                        temp_code_length = self._code_table.compute_kgmdl_total_description_length()
                        self._already_test.append(candidate)
                        # if the new ct is better than the old, break and generate new candidates
                        # with the new ct
                        if temp_code_length < self._description_length:
                            self._rewritten_graph = self._code_table.rewritten_graph()
                            self._description_length = temp_code_length
                            utils.MyLogger().info("New DL ", self._description_length)
                            utils.MyLogger().info(f"new pattern added: {utils.display_graph(row.pattern())}")
                            utils.MyLogger().info(f"search time = {time.time() - b}")
                            self._compute_pruning_candidates()
                            self._pruning()
                            break
                        elif temp_code_length > self._description_length and candidates.index(candidate) == len(
                                candidates) - 1:
                            utils.MyLogger().info("None best code table found")
                            utils.MyLogger().info(f"search time = {time.time() - b}")
                            self._code_table.remove_row(row)
                            self._graph_mdl_end()
                            current = timeout
                            break
                        else:
                            # if the candidate not improve the result, remove it to the code table
                            self._code_table.remove_row(row)
                    else:
                        utils.MyLogger().debug("Already test")
            else:
                self._graph_mdl_end()
                current = timeout
        return self

    def _stop_by_time(self, passed_time, timeout):
        """ Check if the passed time surpasses the timeout
        Parameters
        ---------
        passed_time : int
            The passed time
        timeout : int
            The maximum time
        Returns
        --------
        bool
        """
        if passed_time >= timeout:
            return self._graph_mdl_end()
        else:
            return False

    def _graph_mdl_end(self):
        """ Complete the graph mdl algorithm by cover the code table and stop
        Returns
        -------
        bool
        """
        utils.MyLogger().info("GraphMDL+ end .....")
        self._code_table.cover(debug=self._debug)
        self._rewritten_graph = self._code_table.rewritten_graph()
        self._description_length = self._code_table.compute_kgmdl_total_description_length()
        utils.MyLogger().info(f"Final description length = {round(self._description_length, 2)}")
        utils.MyLogger().info(f"Number of patterns found = {len(self.patterns())}")

        return True

    def _order_candidates(self, candidate):
        """Provide the candidate elements to order candidates
        Parameters
        ----------
        candidate
        Returns
        -------
        list
        """
        return [candidate.usage, candidate.exclusive_port_number,
                -self._label_codes.encode(candidate.final_pattern())]

    def _compute_old_usage(self):
        """ Store patterns usage before compute new code table patterns usage """
        self._old_usage = dict()
        for r in self._code_table.rows():
            self._old_usage[r.pattern()] = r.pattern_usage()

    def _compute_pruning_candidates(self):
        """ Find the row where their usage has decreased since the last usage,
        it's the step before the code table pruning"""
        for r in self._code_table.rows():
            if r.pattern() in self._old_usage.keys():
                if r.pattern_usage() < self._old_usage[r.pattern()]:
                    self._pruning_rows.append(r)

    def _pruning(self):
        """ Make the code table pruning as krimp pruning
           That's consist of remove row in code table who are unnecessary,
            because without them the code table is better """
        utils.MyLogger().info(f"Pruning start ....")
        self._compute_old_usage()  # compute old pattern usage
        self._pruning_rows.sort(key=_order_pruning_rows)  # sort the pruning rows
        for r in self._pruning_rows:
            self._code_table.remove_row(r)
            self._code_table.cover()
            if self._code_table.compute_kgmdl_total_description_length() < self._description_length:
                self._rewritten_graph = self._code_table.rewritten_graph()
                self._description_length = self._code_table.compute_kgmdl_total_description_length()
                self._pruning_rows.remove(r)
                self._compute_pruning_candidates()  # recompute the pruning candidates
                self._pruning()
            else:
                self._code_table.add_row(r)

        utils.MyLogger().info("Pruning end ....")

    def patterns(self):
        """ Provide the algorithm found patterns
        Returns
        -------
        set
        """
        self._patterns = set()
        if self._code_table is not None:
            for r in self._code_table.rows():
                if r.code_length() != 0:
                    self._patterns.add(r.pattern())

            for s in self._code_table.singleton_code_length().keys():
                self._patterns.add(utils.create_singleton_pattern(s, self._code_table))

            return self._patterns

        else:
            print("Fit The algorithm firstly")

    def description_length(self):
        """ Provide the description length
        Returns
        --------
        float
        """
        return self._description_length

    def initial_description_length(self):
        """ Provide the initial description length
        Returns
        -------
        float
        """
        return self._initial_description_length

    def discover(self, *args, **kwargs):
        """ Provide a summary of the algorithm execution"""
        if self._code_table is not None:
            print(self._code_table.display_ct())
            print("final description length : ", self._description_length)
            print("Non singleton patterns found : ")
            for p in self.patterns():
                print('\n', utils.display_graph(p))
            if len(self._code_table.singleton_code_length()) != 0:
                print("Singleton patterns found : ")
                for s in self._code_table.singleton_code_length().keys():
                    print("\n", s)
        else:
            print("Fit the graphmdl+ algorithm firstly")

    def _code_table_row_to_json(self):
        pass

    def to_json(self):
        res = dict()
        # Add the description lengths ( that of the total, the code table and the rewritten graph)
        res["description_lengths"] = {"total": self._description_length,
                                      "data": self._code_table.compute_kgmdl_total_description_length(),
                                      "model": self._code_table.description_length()}

        res["edge_standard_table"] = self._label_codes.edges_lc()
        res["kind"] = "simple_directed"
        res["patterns"] = self._code_table.to_json()
        res["vertex_standard_table"] = self._label_codes.vertex_lc()
        return res
