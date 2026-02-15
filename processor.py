import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional
from model import Color, Group, GroupPack


def entire(val1: float, val2: float) -> bool:
    if val1 < val2:
        val1, val2 = val2, val1
    return val1 // val2 == val1 / val2


class ImageProcessor:
    @staticmethod
    def gray(image: np.ndarray) -> np.ndarray:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    @staticmethod
    def threshold(image: np.ndarray, on_value: int) -> np.ndarray:
        if image.ndim > 2:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        _, threshold = cv.threshold(image, on_value, 255, 0)
        #return threshold

        # Применяем адаптивную бинаризацию
        # blockSize=15 и C=2 - параметры, которые можно вынести в настройки
        # threshold = cv.adaptiveThreshold(
        #     image,
        #     255,
        #     cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv.THRESH_BINARY_INV,  # Инвертируем, чтобы линии были белыми (как ожидает findContours)
        #     blockSize=15,
        #     C=2
        # )
        return threshold

    @staticmethod
    def hull_points(image: np.ndarray) -> GroupPack:
        if len(image.shape) != 2:
            raise AttributeError("Изображение неверного формата")
        groups = []
        contours, hierarchy = cv.findContours(
            image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            hull: np.ndarray = cv.convexHull(cnt, returnPoints=True)
            hull = hull.squeeze(1)
            groups.append(Group(hull))
        return GroupPack(groups)

    @staticmethod
    def draw_points(image: np.ndarray, *points, **settings) -> None:
        points: List[tuple] = [point.to_tuple() for point in points]
        thickness = settings.get("thickness") or 2
        color = settings.get("color") or Color.White
        for point in points:
            cv.circle(image, point, thickness, color, -1)

    @staticmethod
    def _crop_vertical(image: np.ndarray, on_value=255, **add) -> Tuple[int, int]:
        left, right = None, None
        dim_h, dim_w = image.shape
        image = list(zip(*image))
        for i, row in enumerate(image):
            if on_value in row:
                left = i
                break
        for i, row in enumerate(image[::-1]):
            if on_value in row:
                right = dim_w - i
                break

        left = left if left is not None else 0
        right = right if right is not None else dim_w

        first_add = add.get("first", 0)
        second_add = add.get("second", 0)
        return int(left + first_add), int(right + second_add)

    @staticmethod
    def _crop_horizontal(image: np.ndarray, on_value=255, **add) -> Tuple[int, int]:
        top, down = None, None
        dim_h, dim_w = image.shape
        for i, row in enumerate(image):
            if on_value in row:
                top = i
                break
        for i, row in enumerate(image[::-1]):
            if on_value in row:
                down = dim_h - i
                break

        top = top if top is not None else 0
        down = down if down is not None else dim_h

        first_add = add.get("first", 0)
        second_add = add.get("second", 0)
        return int(top + first_add), int(down + second_add)

    @staticmethod
    def crop(image: np.ndarray, top_crop: int = 0) -> np.ndarray:
        one = ImageProcessor._crop_horizontal(image, first=top_crop)
        two = ImageProcessor._crop_vertical(image)
        one, two = (two[0], one[0]), (two[1], one[1])
        image = image[one[1]:two[1], one[0]:two[0]]
        return image

    @staticmethod
    def resize(image: np.ndarray, width: int, height: int, interpolation: Optional[int] = None) -> np.ndarray:
        interpolation = interpolation or cv.INTER_AREA
        return cv.resize(image, (width, height), interpolation=interpolation)

    @staticmethod
    def repeate(image: np.ndarray, axis: int, amount: int = 2) -> np.ndarray:
        if amount < 2:
            return image
        concated = image.copy()
        for _ in range(amount - 1):
            concated = np.concatenate((concated, image), axis=axis)
        return concated

    @staticmethod
    def concat(origin: np.ndarray, appendix: np.ndarray, axis: int) -> np.ndarray:
        origin_shape: Tuple[int, int] = origin.shape
        appendix_shape: Tuple[int, int] = appendix.shape
        if not entire(origin_shape[0], appendix_shape[0]) or not entire(origin_shape[1], appendix_shape[1]):
            raise AttributeError(
                "[!] Несовпадение форм исследуемых изображений")
        return np.concatenate((origin, appendix), axis=axis)

    @staticmethod
    def masking(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if base.ndim != 2:
            base = cv.cvtColor(base, cv.COLOR_BGR2GRAY)
        if mask.ndim != 2:
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        if base.shape != mask.shape:
            print(base.shape)
            print(mask.shape)
            raise ValueError("Растр и маска разных размеров")
        return cv.bitwise_and(base, base, mask=mask)