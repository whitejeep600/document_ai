from dataclasses import dataclass

import numpy as np

from src.constants import PubLayNetLabel


@dataclass
class BBox:
    x_min: int
    y_min: int
    width: int
    height: int


@dataclass
class DocumentImageAnnotation:
    label: PubLayNetLabel
    bbox: BBox


@dataclass
class DocumentImageSample:
    image: np.ndarray
    annotations: list[DocumentImageAnnotation]
    image_filename: str
