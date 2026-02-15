# import cv2 as cv
# import numpy as np
# import math
# from model import Point, Section
# from settings import WindowSettings, RasterSettings
# from paths import save_data
#
#
# def _get_line_shift(distance, angle):
#     sin = math.sin(math.radians(angle + 90))
#     cos = math.cos(math.radians(angle + 90))
#     return int(distance * cos), int(distance * sin)
#
#
# def _get_lines_amount(center: Point, distance):
#     return 2 * int(Point.distance(Point(0, 0), center) / distance)
#
#
# class RasterFactory:
#     def __init__(self, win_settings: WindowSettings, raster_settings: RasterSettings):
#         self.width = win_settings.width
#         self.height = win_settings.height
#         self.settings = raster_settings
#
#     @property
#     def raster(self):
#         return self._raster
#
#     @raster.setter
#     def raster(self, value):
#         self._raster = value
#
#     def process(self):
#         angle_rad = np.deg2rad(self.settings.angle)
#         distance = self.settings.distance
#         thickness = self.settings.thickness
#         period = distance+thickness
#
#         x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
#         projection = x*np.sin(angle_rad)+y*np.cos(angle_rad)
#
#         self._raster = np.where((projection%period)<thickness, 0, 255).astype(np.uint8)
#
#         return self._raster

# import cv2 as cv  # <-- 1. ДОБАВЛЕН ИМПОРТ
# import numpy as np
# import math
# from model import Point, Section
# from settings import WindowSettings, RasterSettings
# from paths import save_data
#
#
# def _get_line_shift(distance, angle):
#     sin = math.sin(math.radians(angle + 90))
#     cos = math.cos(math.radians(angle + 90))
#     return int(distance * cos), int(distance * sin)
#
#
# def _get_lines_amount(center: Point, distance):
#     # Эта функция уже существует в файле
#     return 2 * int(Point.distance(Point(0, 0), center) / distance)
#
#
# class RasterFactory:
#     def __init__(self, win_settings: WindowSettings, raster_settings: RasterSettings):
#         self.width = win_settings.width
#         self.height = win_settings.height
#         self.settings = raster_settings
#
#         # --- 2. ДОБАВЛЕНЫ СЛЕДУЮЩИЕ 2 СТРОКИ ---
#         # Получаем центр из настроек окна
#         self.center = win_settings.center
#         # Рассчитываем необходимое количество линий
#         self.amount = _get_lines_amount(self.center, self.settings.distance)
#
#     @property
#     def raster(self):
#         return self._raster
#
#
#     @raster.setter
#     def raster(self, value):
#         self._raster = value
#
#
#     # --- 3. ЗАМЕНЕННЫЙ МЕТОД PROCESS ---
#     def process(self):
#         """
#         Новая функция process, предоставленная пользователем,
#         с добавленной инициализацией self._raster.
#         """
#         angle = self.settings.angle
#         distance = self.settings.distance
#         thickness = self.settings.thickness
#         offset = self.settings.offset
#         color = self.settings.color  #
#         length = math.ceil(2 * Point.distance(Point(0, 0), self.center))
#
#         # Инициализируем растр как белый (255) холст нужного размера
#         # Это необходимо, т.к. cv.line будет рисовать на self._raster
#         self._raster = np.full((self.height, self.width), 255, dtype=np.uint8)
#
#         xshift, yshift = _get_line_shift(distance, angle)
#         xoffset, yoffset = _get_line_shift(offset, angle)
#         high_point = Point(self.center.cox + self.amount *
#                            xshift, self.center.coy + self.amount * yshift)
#         normal = Section(self.center, high_point)
#         # [cite: 31, 32, 33]
#         perp = Section.perp(normal, length)
#
#         for i in range(self.amount, -self.amount, -1):
#             pta = (perp.pta.cox + xoffset, perp.pta.coy + yoffset)
#             ptb = (perp.ptb.cox + xoffset, perp.ptb.coy + yoffset)
#
#             # cv.line будет использовать первый элемент color (0)
#             # для рисования черным на 2D-массиве
#             cv.line(self._raster, pta, ptb, color, thickness)
#
#             perp.shift(xshift, yshift)
#
#         return self._raster

# --- Вставьте это в ВЕРХНЮЮ часть файла factory.py ---

import cv2 as cv  # [cite: 227]
import numpy as np
import math
from model import Point, Section
from settings import WindowSettings, RasterSettings
#from paths import save_data


# ИСПРАВЛЕНИЕ 1: Полностью заменяем _get_line_shift
def _get_line_shift(distance, angle):
    """
    Вычисляет вектор СДВИГА (перпендикуляр к линиям)
    для координатной системы OpenCV (+y = вниз).
    0° -> (0, +k) -> вертик. сдвиг -> гориз. линии
    45° -> (+k, +k) -> диаг. сдвиг -> '/' линии
    90° -> (+k, 0) -> гориз. сдвиг -> вертик. линии
    """
    # Мы используем (90 - angle) для преобразования
    # тригонометрического угла (0°=H) в систему CV.
    rad = math.radians(90 - angle)
    cos = math.cos(rad)
    sin = math.sin(rad)
    return int(distance * cos), int(distance * sin)


def _get_lines_amount(center: Point, distance):
    # Эта функция уже существует в файле
    # Оставляем ее без изменений
    return 2 * int(Point.distance(Point(0, 0), center) / distance)


class RasterFactory:
    def __init__(self, win_settings: WindowSettings, raster_settings: RasterSettings):
        # Оставляем __init__ без изменений [cite: 227, 228]
        self.width = win_settings.width
        self.height = win_settings.height
        self.settings = raster_settings
        self.center = win_settings.center
        self.amount = _get_lines_amount(self.center, self.settings.distance)

    @property
    def raster(self):
        # Оставляем без изменений
        return self._raster

    @raster.setter
    def raster(self, value):
        # Оставляем без изменений
        self._raster = value

    # ИСПРАВЛЕНИЕ 2: Полностью заменяем метод process
    def process(self):
        """
        Новая, робастная функция process.
        Она заменяет сложную логику из .
        Эта функция НЕ использует Section.perp из model.py.
        """
        angle = self.settings.angle
        distance = self.settings.distance
        thickness = self.settings.thickness
        offset_val = self.settings.offset
        color = self.settings.color

        # 1. Инициализируем растр
        self._raster = np.full((self.height, self.width), 255, dtype=np.uint8)

        # 2. Получаем вектор СДВИГА (перпендикуляр к линиям)
        #    (используя нашу новую _get_line_shift)
        xshift, yshift = _get_line_shift(distance, angle)
        if xshift == 0 and yshift == 0:
            # Предохранитель, если distance = 0
            xshift = 1

            # 3. Получаем вектор ЛИНИИ (поворот вектора сдвига на 90°)
        #    (x, y) -> (-y, x) (поворот против часовой)
        ldx = -yshift
        ldy = xshift

        # 4. Нормализуем и удлиняем вектор линии
        line_mag = math.sqrt(ldx ** 2 + ldy ** 2)
        if line_mag < 1e-6:
            line_mag = 1  # Предотвращаем деление на ноль

        # Длина должна быть больше диагонали экрана
        diag_length = math.sqrt(self.width ** 2 + self.height ** 2)

        ldx_norm = ldx / line_mag
        ldy_norm = ldy / line_mag

        # 5. Смещение (offset)
        #    (используя нашу новую _get_line_shift)
        xoffset, yoffset = _get_line_shift(offset_val, angle)

        # 6. self.amount  - это кол-во линий для
        #    покрытия 2*радиуса от (0,0) до центра.
        #    Старый цикл [cite: 233] был (amount, -amount).
        #    Мы будем использовать тот же диапазон для
        #    гарантированного покрытия экрана.

        for i in range(-self.amount, self.amount):
            # 7. Рассчитываем центральную точку i-й линии
            cx = self.center.cox + i * xshift + xoffset
            cy = self.center.coy + i * yshift + yoffset

            # 8. Рассчитываем концы линии (от центра в обе стороны)
            pta = (int(cx - ldx_norm * diag_length),
                   int(cy - ldy_norm * diag_length))
            ptb = (int(cx + ldx_norm * diag_length),
                   int(cy + ldy_norm * diag_length))

            # 9. Рисуем
            cv.line(self._raster, pta, ptb, color, thickness)

        return self._raster