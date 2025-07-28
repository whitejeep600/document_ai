from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


@dataclass
class BBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @classmethod
    def from_xyxy(cls, x_min: int, y_min: int, x_max: int, y_max: int) -> "BBox":
        return BBox(x_min, y_min, x_max, y_max)

    @classmethod
    def from_xywh(cls, x_min: int, y_min: int, width: int, height: int) -> "BBox":
        return BBox(x_min, y_min, x_min + width, y_min + height)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def start_point(self) -> tuple[int, int]:
        return self.x_min, self.y_min

    def end_point(self) -> tuple[int, int]:
        return self.x_max, self.y_max


class PubLayNetCategory(Enum):
    TEXT = "text"
    TITLE = "title"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"

    @classmethod
    def from_text(cls, text: str) -> Optional["PubLayNetCategory"]:
        if text in [category.value for category in PubLayNetCategory]:
            return PubLayNetCategory(text)
        else:
            # Unrecognized categories
            return None

    @classmethod
    def from_category_code(cls, code: int) -> "PubLayNetCategory":
        return {
            1: cls.TEXT,
            2: cls.TITLE,
            3: cls.LIST,
            4: cls.TABLE,
            5: cls.FIGURE,
        }[code]


@dataclass
class DetectionOrAnnotation:
    category: PubLayNetCategory
    bbox: BBox


@dataclass
class DocumentImageSample:
    image: np.ndarray
    annotations: list[DetectionOrAnnotation]
    image_filename: str
