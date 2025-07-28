from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constants import PubLayNetLabel


@dataclass
class BBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @classmethod
    def from_xyxy(cls, x_min: int, y_min: int, x_max: int, y_max: int) -> "BBox":
        return BBox(
            x_min,
            y_min,
            x_max,
            y_max
        )

    @classmethod
    def from_xywh(cls, x_min: int, y_min: int, width: int, height: int) -> "BBox":
        return BBox(
            x_min,
            y_min,
            x_min + width,
            y_min + height
        )

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def start_point(self) -> tuple[int, int]:
        return self.x_min, self.y_min

    def end_point(self) -> tuple[int, int]:
        return self.x_max, self.y_max


@dataclass
class DocumentImageAnnotation:
    label: PubLayNetLabel
    bbox: BBox


@dataclass
class DocumentImageSample:
    image: np.ndarray
    annotations: list[DocumentImageAnnotation]
    image_filename: str
