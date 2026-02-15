import numpy as np
from enum import IntEnum
from dataclasses import dataclass


class SourceType(IntEnum):
    NONE = 0
    RAW = 1
    RASTER = 2
    PROCESSED = 3


@dataclass
class ImageData:
    image: np.ndarray
    source: SourceType

    def shape(self):
        return self.image.shape