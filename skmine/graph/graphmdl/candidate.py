
class Candidate:

    def __init__(self, first_pattern, second_pattern, port):
        self.first_pattern_label = first_pattern
        self.second_pattern_label = second_pattern
        self.first_pattern = None
        self.second_pattern = None
        self.port = port
        self.data_port = set()
        self.usage = 0
        self.exclusive_port_number = 0

    def set_usage(self, embed_number):
        self.usage = embed_number

    def __str__(self) -> str:
        return "<{},{},{}>".format(self.first_pattern_label, self.second_pattern_label, self.port)

    def __eq__(self, o: object) -> bool:
        return o.first_pattern_label == self.first_pattern_label \
               and o.second_pattern_label == self.second_pattern_label\
               and o.port == self.port


