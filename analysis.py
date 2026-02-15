import math
import cv2 as cv
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from model import Color, Point, DeformType
from image_data import ImageData, SourceType
from processor import ImageProcessor


@dataclass
class DistanceAggregator:
    muar_point: Point
    template_point: Point
    distance: float

@dataclass
class DeformationVector:
    muar_point: Point
    template_point: Point
    distance: float
    dx: float # Вектор смещения по X
    dy: float # Вектор смещения по Y

class ProcessedDataFields:
    TEMPLATE_IMAGE = "Template"
    MUAR_IMAGE = "Muar"
    TEMPLATE_POINTS = "TemplatePoints"
    MUAR_POINTS = "MuarPoints"
    ALL_POINTS_BY_ROW = "AllPointsByRow"
    MIN_DISTANCES = "MinDistances"
    PERSENTILES = "Persentiles"


BY_DEFORM_MSG = {DeformType.noneDeform: "Нет дефектов",
                 DeformType.inDeform: "Вогнутость",
                 DeformType.outDeform: "Выпуклось"}


class AnalizatorBaseException(Exception):
    """ Базовый класс ошибок анализатора """


class AnalizatorAttributeError(AnalizatorBaseException):
    """ Переданы неправильные входные данные """


class Analizator:
    """ Складывает обработанное изображение с растром и анализирует данные """

    # def __init__(self, base_raster: ImageData, over_raster: ImageData, processed_image: ImageData):
    #     if (
    #             base_raster.source is not SourceType.RASTER
    #             or over_raster.source is not SourceType.RASTER
    #             or processed_image.source is not SourceType.PROCESSED
    #     ):
    #         raise AnalizatorAttributeError("[!] Переданы неправильные входные данные "
    #                                        f"{base_raster.source} {over_raster.source} {processed_image.source}")
    #     if processed_image.image.ndim > 2:
    #         _processed_image = ImageProcessor.threshold(
    #             processed_image.image, 127)
    #     else:
    #         _processed_image = processed_image.image
    #     _processed_image = ImageProcessor.resize(
    #         _processed_image, 1024, 768, interpolation=cv.INTER_AREA)
    #     self._base_raster = base_raster
    #     self._over_raster = over_raster
    #     self._processed_image = ImageData(
    #         _processed_image, SourceType.PROCESSED)
    #     self.processed_data = {}
    #     self._process()

    def __init__(self, base_raster: ImageData, over_raster: ImageData, processed_image: ImageData,
                 # Принимаем данные калибровки
                 calibration_p50: float = 4.0,
                 calibration_ratio: float = 2.1,# Принимаем целевой размер из GUI
                 target_width: int = 1024,
                 target_height: int = 768):
                 # === КОНЕЦ ИЗМЕНЕНИЯ ===:

        if (
                base_raster.source is not SourceType.RASTER
                or over_raster.source is not SourceType.RASTER
                or processed_image.source is not SourceType.PROCESSED
        ):
            raise AnalizatorAttributeError("[!] Переданы неправильные входные данные "
                                           f"{base_raster.source} {over_raster.source} {processed_image.source}")

        # 1. Определяем целевой размер (жестко задан в коде)


        # 2. Подготавливаем processed_image (оно уже 1-канальное после adaptive_threshold)
        _processed_image = processed_image.image

        # 3. Принудительно изменяем РАЗМЕР ВСЕХ ИЗОБРАЖЕНИЙ
        _processed_image = ImageProcessor.resize(
            _processed_image, target_width, target_height, interpolation=cv.INTER_AREA)

        _base_raster_img = ImageProcessor.resize(
            base_raster.image, target_width, target_height, interpolation=cv.INTER_AREA)

        _over_raster_img = ImageProcessor.resize(
            over_raster.image, target_width, target_height, interpolation=cv.INTER_AREA)

        # 4. Убедимся, что растры 1-канальные (важно для masking)
        if _base_raster_img.ndim > 2:
            _base_raster_img = ImageProcessor.gray(_base_raster_img)
        if _over_raster_img.ndim > 2:
            _over_raster_img = ImageProcessor.gray(_over_raster_img)

        # 5. Сохраняем обработанные данные
        self._base_raster = ImageData(_base_raster_img, base_raster.source)
        self._over_raster = ImageData(_over_raster_img, over_raster.source)
        self._processed_image = ImageData(_processed_image, SourceType.PROCESSED)

        # 6. Сохраняем данные калибровки
        self.calibration_p50 = calibration_p50
        self.calibration_ratio = calibration_ratio

        self.processed_data = {}
        self._process()

    def _row_border_coord_template(self):
        t_points = self.template_points
        y_points = [point.coy for point in t_points]
        y_points = sorted(list(set(y_points)))
        h_half = (y_points[1] - y_points[0]) / 2
        ranges = [(i, [y_co - h_half, y_co + h_half])
                  for i, y_co in enumerate(y_points, start=1)]
        return ranges

    def _set_points_into_near_row(self):
        muar_rows = []
        template_ranges = self._row_border_coord_template()
        m_points = self.muar_points
        for row in template_ranges:
            for point in m_points:
                if point.coy >= row[1][0] and not point.coy >= row[1][1]:
                    muar_rows.append((row[0], point))
        template_rows = []
        t_points = self.template_points
        for row in template_ranges:
            for point in t_points:
                if point.coy >= row[1][0] and not point.coy >= row[1][1]:
                    template_rows.append((row[0], point))
        return template_rows, muar_rows

    def _sort_points_by_rows(self):
        self._set_points_into_near_row()
        t_points_rows_sorted, m_points_rows_sorted = self._set_points_into_near_row()
        t_points_by_rows = defaultdict(list)
        m_points_by_rows = defaultdict(list)
        for t_point in t_points_rows_sorted:
            t_points_by_rows[t_point[0]].append(t_point[1])
        for m_point in m_points_rows_sorted:
            m_points_by_rows[m_point[0]].append(m_point[1])
        self.processed_data[ProcessedDataFields.ALL_POINTS_BY_ROW] = {
            "T": t_points_by_rows, "M": m_points_by_rows}

    def _row_distance_aggregate(self, template_row_points: List[Point], muar_row_points: List[Point]):
        muar_to_template_dist_aggregates = []
        for mrp in muar_row_points:
            min_dist = math.inf
            t_point = None
            for trp in template_row_points:
                dist = math.dist(mrp.to_tuple(), trp.to_tuple())
                if dist < min_dist:
                    min_dist = dist
                    t_point = trp
        #     if t_point:
        #         muar_to_template_dist_aggregates.append(
        #             DistanceAggregator(mrp, t_point, min_dist))
        # return muar_to_template_dist_aggregates
            if t_point:
                dx = mrp.cox - t_point.cox
                dy = mrp.coy - t_point.coy
                muar_to_template_dist_aggregates.append(
                    DeformationVector(mrp, t_point, min_dist, dx, dy))  # Используем новый класс
        return muar_to_template_dist_aggregates

    # def _point_distance_analysis(self):
    #     distance_aggregators = []
    #     template_points_by_row: dict = self.processed_data[ProcessedDataFields.ALL_POINTS_BY_ROW]["T"]
    #     muar_points_by_row: dict = self.processed_data[ProcessedDataFields.ALL_POINTS_BY_ROW]["M"]
    #     selected_rows_count = min(
    #         [max(list(template_points_by_row.keys())), max(list(muar_points_by_row.keys()))])
    #     for i in range(selected_rows_count):
    #         distance_aggregators.extend(self._row_distance_aggregate(
    #             template_points_by_row[i], muar_points_by_row[i]))
    #
    #     self.processed_data[ProcessedDataFields.MIN_DISTANCES] = distance_aggregators

    def _calc_persentiles(self):
        distance_aggregators = self.processed_data[ProcessedDataFields.MIN_DISTANCES]
        distances = [dist_agg.distance for dist_agg in distance_aggregators]
        persent50 = np.percentile(distances, 50)
        persent90 = np.percentile(distances, 90)
        persent99 = np.percentile(distances, 99)
        self.processed_data[ProcessedDataFields.PERSENTILES] = (
            persent50, persent90, persent99)

    def _process(self):
        template = ImageProcessor.masking(
            self._base_raster.image, self._over_raster.image)
        muar = ImageProcessor.masking(
            self._processed_image.image, self._over_raster.image)
        self.processed_data[ProcessedDataFields.TEMPLATE_IMAGE] = template
        self.processed_data[ProcessedDataFields.MUAR_IMAGE] = muar
        t_points = ImageProcessor.hull_points(template).centers
        m_points = ImageProcessor.hull_points(muar).centers
        self.processed_data[ProcessedDataFields.TEMPLATE_POINTS] = t_points
        self.processed_data[ProcessedDataFields.MUAR_POINTS] = m_points
        # self._sort_points_by_rows()
        # self._point_distance_analysis()
        # self._calc_persentiles()
        # [ИСПРАВЛЕНИЕ] Эта функция вызывает всю ошибку.
        # self._sort_points_by_rows()

        self._point_distance_analysis()
        self._calc_persentiles()

    @property
    def template_image(self):
        return self.processed_data[ProcessedDataFields.TEMPLATE_IMAGE]

    @property
    def muar_image(self):
        return self.processed_data[ProcessedDataFields.MUAR_IMAGE]

    @property
    def template_points(self) -> List[Point]:
        return self.processed_data[ProcessedDataFields.TEMPLATE_POINTS]

    @property
    def muar_points(self) -> List[Point]:
        return self.processed_data[ProcessedDataFields.MUAR_POINTS]

    @property
    def distanses(self):
        return self.processed_data[ProcessedDataFields.MIN_DISTANCES]

    @property
    def persentiles(self):
        return self.processed_data[ProcessedDataFields.PERSENTILES]

    def has_deform(self):
        persentiles = self.processed_data[ProcessedDataFields.PERSENTILES]
        persentiles50 = persentiles[0]
        persentiles99 = persentiles[2]
        if persentiles50 > 4:
            return True
        if persentiles99 / persentiles50 > 2.1:
            return True
        return False

    def get_deformation_analysis(self) -> (DeformType, float):
        """
        Возвращает тип деформации и медианную величину отклонения (P50).
        Использует данные калибровки для отсечения шума.
        """
        persentiles = self.processed_data[ProcessedDataFields.PERSENTILES]
        persentiles50 = persentiles[0]
        persentiles99 = persentiles[2]

        # Используем данные калибровки, переданные в __init__
        # Устанавливаем порог срабатывания, например, в 3 раза выше "шума"
        noise_threshold_p50 = self.calibration_p50
        # Порог разброса, например, в 1.5 раза выше "шума"
        noise_threshold_ratio = self.calibration_ratio

        has_significant_deformation = False
        if persentiles50 > noise_threshold_p50:
            has_significant_deformation = True

        # Проверяем разброс
        if (persentiles99 / (persentiles50 + 1e-6)) > noise_threshold_ratio:
            has_significant_deformation = True

        if not has_significant_deformation:
            # Отклонение ниже калибровочного порога
            return DeformType.noneDeform, persentiles50

            # Если деформация значительная, определяем ее тип
        deform_type = self._analyze_deformation_type()

        # Возвращаем тип и медианную величину отклонения
        return deform_type, persentiles50

#изначальный вариант
    # def _poster_select_great_heights(self, poster: np.ndarray):
    #     color = Color.Red
    #     select_on = self.processed_data[ProcessedDataFields.PERSENTILES][1]
    #     for dist_aggregate in self.processed_data[ProcessedDataFields.MIN_DISTANCES]:
    #         #dist_aggregate: DistanceAggregator
    #         dist_aggregate: DeformationVector
    #         if dist_aggregate.distance >= select_on:
    #             point1 = dist_aggregate.muar_point.to_tuple()
    #             point2 = dist_aggregate.template_point.to_tuple()
    #             cv.line(poster, point1, point2, color, 2)

#стат. очистка
    # def _poster_select_great_heights(self, poster: np.ndarray):
    #     """
    #     Метод 2: Статистическая очистка.
    #     Рисует только "топ 10%" (P90) от данных,
    #     предварительно очищенных от "выбросов" (P99).
    #     """
    #     color = Color.Red  # BGR-цвет для красного
    #
    #     all_vectors: List[DeformationVector] = self.processed_data[ProcessedDataFields.MIN_DISTANCES]
    #     if not all_vectors:
    #         return
    #
    #     # 1. Получаем 99-й перцентиль (порог "артефакта")
    #     # persentiles = (p50, p90, p99)
    #     p99_threshold = self.processed_data[ProcessedDataFields.PERSENTILES][2]
    #
    #     # 2. Создаем "очищенный" список векторов (все, что ниже P99)
    #     filtered_vectors = [v for v in all_vectors if v.distance <= p99_threshold]
    #
    #     if not filtered_vectors:
    #         # Все векторы были отфильтрованы как выбросы
    #         return
    #
    #     # 3. Извлекаем "очищенные" дистанции
    #     filtered_distances = [v.distance for v in filtered_vectors]
    #
    #     # 4. Считаем НОВЫЙ 90-й перцентиль (топ 10%) ОТ ОЧИЩЕННЫХ данных
    #     new_p90_threshold = np.percentile(filtered_distances, 90)
    #
    #     # 5. Рисуем только те векторы из очищенных,
    #     #    которые попали в новый топ 10%
    #     for v in filtered_vectors:
    #         if v.distance >= new_p90_threshold:
    #             point1 = v.muar_point.to_tuple()
    #             point2 = v.template_point.to_tuple()
    #             cv.line(poster, point1, point2, color, 2)

#тепловая окраска смещений
    def _poster_select_great_heights(self, poster: np.ndarray):
        """
        Метод 3: Тепловая карта.
        Раскрашивает векторы в градиенте от синего до красного.
        - Ниже P50: не рисуем (шум)
        - P50: Синий
        - P99: Красный
        - Выше P99: Тоже красный (обрезаем)
        """

        all_vectors: List[DeformationVector] = self.processed_data[ProcessedDataFields.MIN_DISTANCES]
        if not all_vectors:
            return

        # 1. Получаем пороги P50 (минимум) и P99 (максимум)
        persentiles = self.processed_data[ProcessedDataFields.PERSENTILES]
        min_thresh = persentiles[0]  # P50 - "нулевой" уровень
        max_thresh = persentiles[2]  # P99 - "максимальный" уровень

        # 2. Проходим по *каждому* вектору
        for v in all_vectors:
            dist = v.distance

            # 3. Отфильтровываем шум (все, что ниже P50)
            if dist < min_thresh:
                continue

            # 4. Нормализуем значение в диапазон 0-1
            # (P50 -> 0.0, P99 -> 1.0)

            # (dist - min_thresh) - сколько "превышение" над P50
            # (max_thresh - min_thresh) - весь наш рабочий диапазон
            norm_value = (dist - min_thresh) / (max_thresh - min_thresh + 1e-6)

            # Обрезаем значения:
            # все, что > 1.0 (т.е. > P99), приравниваем к 1.0
            norm_value_clipped = np.clip(norm_value, 0, 1)

            # 5. Превращаем [0, 1] в 8-битное значение [0, 255]
            # cv2.COLORMAP_JET: 0 - синий, 255 - красный
            value_uint8 = np.uint8(norm_value_clipped * 255)

            # 6. Получаем BGR-цвет из тепловой карты
            # Создаем 1-пиксельный массив, чтобы cv.applyColorMap сработал
            color_lookup = np.array([value_uint8], dtype=np.uint8)
            color_bgr_array = cv.applyColorMap(color_lookup, cv.COLORMAP_JET)

            # Извлекаем BGR-кортеж
            color_bgr_tuple = tuple(int(c) for c in color_bgr_array[0][0])

            # 7. Рисуем линию
            point1 = v.muar_point.to_tuple()
            point2 = v.template_point.to_tuple()
            cv.line(poster, point1, point2, color_bgr_tuple, 2)





    def _get_template_center(self) -> Point:
        """Находит геометрический центр эталонных точек."""
        t_points = self.template_points
        if not t_points:
            # Fallback на центр изображения (размер жестко задан в __init__)
            return Point(1024 // 2, 768 // 2)

        x_center = int(sum(p.cox for p in t_points) / len(t_points))
        y_center = int(sum(p.coy for p in t_points) / len(t_points))
        return Point(x_center, y_center)

    def _analyze_deformation_type(self) -> DeformType:
        """
        Анализирует поле векторов смещения для определения типа деформации.
        """
        vectors: List[DeformationVector] = self.processed_data[ProcessedDataFields.MIN_DISTANCES]
        if not vectors:
            return DeformType.noneDeform

        # 1. Находим "идеальный" центр (относительно него ищем "к" или "от")
        center = self._get_template_center()

        radial_shifts = []

        for v in vectors:
            # 2. Вектор от центра к ИДЕАЛЬНОЙ (template) точке
            vec_to_template_x = v.template_point.cox - center.cox
            vec_to_template_y = v.template_point.coy - center.coy

            norm = math.sqrt(vec_to_template_x ** 2 + vec_to_template_y ** 2)
            if norm == 0:
                continue  # Точка в центре, пропускаем

            # 3. Единичный вектор направления "от центра"
            unit_vec_x = vec_to_template_x / norm
            unit_vec_y = vec_to_template_y / norm

            # 4. Проекция вектора смещения (v.dx, v.dy) на радиальный вектор
            # (скалярное произведение)
            radial_shift = (v.dx * unit_vec_x) + (v.dy * unit_vec_y)

            # radial_shift > 0: смещение ОТ центра (выпуклость)
            # radial_shift < 0: смещение К центру (вогнутость)
            radial_shifts.append(radial_shift)

        if not radial_shifts:
            return DeformType.noneDeform

        # 5. Анализируем результат
        median_shift = np.median(radial_shifts)

        # Порог для определения радиального смещения (абсолютный)
        RADIAL_SHIFT_THRESHOLD = 0.1  # 1 пиксель

        if median_shift > RADIAL_SHIFT_THRESHOLD:
            return DeformType.outDeform  # Выпуклость
        elif median_shift < -RADIAL_SHIFT_THRESHOLD:
            return DeformType.inDeform  # Вогнутость
        else:
            return DeformType.noneDeform

    def poster(self, select_persentile90=True):
        poster_shape = self._processed_image.shape()
        poster_shape = (poster_shape[0], poster_shape[1], 3)
        poster = np.ones(poster_shape, dtype='uint8')*255
        t_hull_group = self.template_points
        m_hull_group = self.muar_points
        t_color = Color.Green
        m_color = Color.Blue
        for t_point in t_hull_group:
            cv.circle(poster, t_point.to_tuple(), 2, t_color, -1)
        for m_point in m_hull_group:
            cv.circle(poster, m_point.to_tuple(), 2, m_color, -1)
        if select_persentile90:
            self._poster_select_great_heights(poster)

        return poster

    def _calculate_nearest_vectors(self, all_template_points: List[Point], all_muar_points: List[Point]):
        """
        Для каждой точки муара (mrp) находит БЛИЖАЙШУЮ точку шаблона (trp)
        на всей 2D-плоскости и вычисляет истинный вектор смещения.
        """
        deformation_vectors = []

        if not all_template_points or not all_muar_points:
            return []

        # Примечание: Этот поиск O(N*M) - медленный.
        # В будущем его можно оптимизировать через K-D Tree,
        # но для корректной работы этого достаточно.

        for mrp in all_muar_points:
            min_dist = math.inf
            t_point = None

            # Ищем по ВСЕМ точкам шаблона
            for trp in all_template_points:
                dist = math.dist(mrp.to_tuple(), trp.to_tuple())
                if dist < min_dist:
                    min_dist = dist
                    t_point = trp

            if t_point:
                # Теперь dx и dy вычисляются корректно,
                # даже если смещение было вертикальным
                dx = mrp.cox - t_point.cox
                dy = mrp.coy - t_point.coy
                deformation_vectors.append(
                    DeformationVector(mrp, t_point, min_dist, dx, dy))

        return deformation_vectors

    def _point_distance_analysis(self):

        # [ИСПРАВЛЕНИЕ] Убираем всю старую логику, основанную на рядах
        # template_points_by_row: dict = self.processed_data[ProcessedDataFields.ALL_POINTS_BY_ROW]["T"]
        # muar_points_by_row: dict = self.processed_data[ProcessedDataFields.ALL_POINTS_BY_ROW]["M"]
        # ... (и т.д.) ...

        # [ИСПРАВЛЕНИЕ] Новая логика:

        # 1. Берем ВСЕ точки
        all_template_points: List[Point] = self.processed_data[ProcessedDataFields.TEMPLATE_POINTS]
        all_muar_points: List[Point] = self.processed_data[ProcessedDataFields.MUAR_POINTS]

        # 2. Вычисляем векторы на основе ближайшего соседа в 2D
        distance_aggregators = self._calculate_nearest_vectors(
            all_template_points,
            all_muar_points
        )

        self.processed_data[ProcessedDataFields.MIN_DISTANCES] = distance_aggregators