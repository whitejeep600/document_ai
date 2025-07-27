from enum import Enum


class RequestField:
    pass

class PubLayNetLabel(Enum):
    TEXT = 1
    TITLE = 2
    LIST = 3
    TABLE = 4
    FIGURE = 5
