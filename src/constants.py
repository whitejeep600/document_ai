
from enum import Enum
from typing import Optional


class RequestField:
    pass


class PubLayNetLabel(Enum):
    TEXT = 1
    TITLE = 2
    LIST = 3
    TABLE = 4
    FIGURE = 5

    @classmethod
    def from_text(cls, text: str) -> Optional["PubLayNetLabel"]:
        # Unrecognized labels are returned as None
        return {
            "text": cls.TEXT,
            "title": cls.TITLE,
            "list": cls.LIST,
            "table": cls.TABLE,
            "figure": cls.FIGURE
        }.get(text, None)

    def to_text(self) -> str:
        if self == PubLayNetLabel.TEXT:
            return "text"
        elif self == PubLayNetLabel.TITLE:
            return "title"
        elif self == PubLayNetLabel.LIST:
            return "list"
        elif self == PubLayNetLabel.TABLE:
            return "table"
        elif self == PubLayNetLabel.FIGURE:
            return "figure"
