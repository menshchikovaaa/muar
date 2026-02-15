import cv2
import numpy as np
from scipy import signal, ndimage
from typing import Dict, Any


class LinePatternAnalyzer:
    def __init__(self):
        self.results = {
            'line_width': 0,  # Ширина ЧЕРНОЙ линии
            'line_step': 0,  # Шаг (период) между линиями (черная + белая)
            'line_angle': 0,  # Угол наклона ЧЕРНЫХ линий
            'confidence': 0.0
        }

    def analyze_image(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Анализ изображения с черными линиями на белом фоне.
        """
        if image_array is None:
            return self.results

        try:
            # Конвертируем в градации серого, если нужно
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_array

            # 1. Определение угла наклона линий
            angle = self._detect_line_angle(gray)
            print(f"✅ Определенный угол линий: {angle:.2f}°")

            # 2. Определение ширины и шага линий
            width, step = self._get_width_and_step(gray, angle)
            print(f"✅ Ширина черной линии: {width:.2f}px, Шаг: {step:.2f}px")

            # 3. Расчет уверенности
            confidence = self._calculate_confidence(gray, width, step)

            self.results = {
                'line_width': width,
                'line_step': step,
                'line_angle': abs(angle),
                'confidence': confidence
            }

            return self.results

        except Exception as e:
            print(f"❌ Ошибка анализа изображения: {e}")
            return self.results



    def _detect_line_angle(self, image: np.ndarray) -> float:
        """
        Определение угла наклона линий с помощью вероятностного преобразования Хафа.
        """
        inverted = cv2.bitwise_not(image)

        # Улучшаем качество изображения для тонких линий
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(inverted, kernel, iterations=1)

        edges = cv2.Canny(dilated, 40, 120, apertureSize=3)

        # Используем вероятностное преобразование Хафа
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=50,
                                minLineLength=50, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Вычисляем угол по координатам начала и конца линии
            if abs(x2 - x1) > 0.1:  # Избегаем деления на ноль
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            else:
                angle = 90.0  # Вертикальная линия

            # Нормализуем угол в диапазон [-90, 90)
            if angle < -90:
                angle += 180
            elif angle >= 90:
                angle -= 180

            angles.append(angle)

        # Статистическая обработка
        if angles:
            angles_array = np.array(angles)

            # Фильтрация выбросов
            if len(angles_array) > 5:
                q1, q3 = np.percentile(angles_array, [25, 75])
                iqr = q3 - q1
                mask = (angles_array >= q1 - 1.5 * iqr) & (angles_array <= q3 + 1.5 * iqr)
                filtered_angles = angles_array[mask]

                if len(filtered_angles) > 0:
                    return float(np.median(filtered_angles))

            return float(np.median(angles_array))

        return 0.0
    def _get_width_and_step(self, image: np.ndarray, angle: float) -> tuple:
        """
        Определение ширины и шага линий путем сканирования изображения
        перпендикулярно линиям, без поворота всего изображения.
        """
        # Получаем центр изображения
        h, w = image.shape
        center_x, center_y = w // 2, h // 2

        # Угол, перпендикулярный линиям, в радианах
        perp_angle_rad = np.deg2rad(angle + 90)

        # Длина сканирующей линии (диагональ изображения)
        line_length = int(np.sqrt(h ** 2 + w ** 2))

        # Генерируем координаты точек вдоль этой перпендикулярной линии
        t = np.linspace(-line_length / 2, line_length / 2, line_length)
        x_coords = center_x + t * np.cos(perp_angle_rad)
        y_coords = center_y + t * np.sin(perp_angle_rad)

        # Отбираем только те координаты, которые находятся в пределах изображения
        valid_mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
        coords = np.vstack((y_coords[valid_mask], x_coords[valid_mask]))

        # Извлекаем значения пикселей вдоль линии (профиль)
        # ndimage.map_coordinates идеально подходит для этого
        profile = ndimage.map_coordinates(image, coords)

        # Бинаризуем профиль: 0 - черное, 1 - белое
        threshold = (profile.min() + profile.max()) / 2
        binary_profile = (profile > threshold).astype(np.uint8)

        # Находим переходы между черным и белым
        transitions = np.diff(binary_profile)
        change_indices = np.where(transitions != 0)[0]

        if len(change_indices) < 2:
            print("Недостаточно переходов цвета для анализа")
            return (10, 30)  # Значения по умолчанию

        # Рассчитываем длины отрезков (черных и белых)
        run_lengths = np.diff(change_indices)

        # Определяем, с какого цвета начинается профиль
        # и разделяем длины на черные и белые
        if binary_profile[change_indices[0]] == 0:  # Переход 1->0, начался черный
            black_runs = run_lengths[0::2]
            white_runs = run_lengths[1::2]
        else:  # Переход 0->1, начался белый
            white_runs = run_lengths[0::2]
            black_runs = run_lengths[1::2]

        if len(black_runs) == 0 or len(white_runs) == 0:
            return (10, 30)

        # Медианная ширина черной линии
        median_width = np.median(black_runs)

        # Медианный шаг (период) = ширина черной + ширина белой
        # Для более точного расчета берем медиану расстояний между началами черных линий
        steps = [black_runs[i] + white_runs[i] for i in range(min(len(black_runs), len(white_runs)))]
        median_step = np.median(steps) if steps else median_width + np.median(white_runs)

        return median_width, median_step

    def _calculate_confidence(self, image: np.ndarray, width: float, step: float) -> float:
        """
        Расчет уверенности в результатах.
        """
        confidence = 1.0
        # Штраф за нелогичные значения
        if width <= 1 or step <= width:
            confidence *= 0.5
        # Штраф за слишком большой шаг относительно ширины
        if step / (width + 1e-6) > 10:
            confidence *= 0.7
        # Учет контрастности изображения
        contrast = image.std()
        confidence *= np.clip(contrast / 100.0, 0.5, 1.0)
        return round(max(0.1, confidence), 2)