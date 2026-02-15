import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Any, Dict
from line_analyzer import LinePatternAnalyzer

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QGroupBox, QSplitter,
    QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox,
    QTabWidget, QScrollArea, QSizePolicy, QMessageBox,
    QGridLayout, QHeaderView, QTableWidget, QTableWidgetItem,
    QDockWidget, QComboBox, QSlider, QProgressBar
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QAction, QPalette, QColor
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QIntValidator

import api
from settings import WindowSettings
from analysis import Analizator, BY_DEFORM_MSG, DeformType


class TextureType:
    WORKING_RASTER = "WORKING_RASTER"
    OBJECT_RASTER = "OBJECT_RASTER"
    IMAGINARY_RASTER = "IMAGINARY_RASTER"
    PROCESSED_OR = "PROCESSED_OR"
    MOIRE_PATTERN = "MOIRE_PATTERN"


class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 250)
        self.setText("Изображение не загружено")
        self.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")

    def set_image(self, image_array: np.ndarray):
        if image_array is None:
            self.clear()
            self.setText("Изображение не загружено")
            return

        try:
            height, width = image_array.shape[:2]
            format = QImage.Format.Format_Invalid
            if len(image_array.shape) == 2:  # Grayscale
                bytes_per_line = width
                format = QImage.Format.Format_Grayscale8
            else:  # Color
                if image_array.shape[2] == 3:
                    # Предполагаем RGB формат
                    bytes_per_line = 3 * width
                    format = QImage.Format.Format_RGB888
                elif image_array.shape[2] == 4:
                    bytes_per_line = 4 * width
                    format = QImage.Format.Format_RGBA8888

            if format != QImage.Format.Format_Invalid:
                qimage = QImage(image_array.data, width, height, bytes_per_line, format)
                pixmap = QPixmap.fromImage(qimage)
                self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
            else:
                self.setText("Неподдерживаемый формат изображения")


        except Exception as e:
            print(f"Ошибка отображения изображения: {e}")
            self.setText("Ошибка загрузки изображения")


class RasterControlPanel(QGroupBox):
    def __init__(self, title, on_create_callback, on_load_callback, on_save_callback, parent=None):
        super().__init__(title, parent)
        self.on_create_callback = on_create_callback
        self.on_load_callback = on_load_callback
        self.on_save_callback = on_save_callback
        self.line_analyzer = LinePatternAnalyzer()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Параметры растра
        params_group = QGroupBox("Параметры растра")
        params_layout = QGridLayout(params_group)

        angle_label = QLabel("Угол (°):")
        angle_label.setFixedWidth(100)
        params_layout.addWidget(angle_label, 0, 0)
        self.angle_input = QSpinBox()
        self.angle_input.setRange(0, 180)
        self.angle_input.setValue(15)
        params_layout.addWidget(self.angle_input, 0, 1)

        params_layout.addWidget(QLabel("Шаг линий (толщина белых линий):"), 1, 0)
        self.distance_input = QSpinBox()
        self.distance_input.setRange(1, 100)
        self.distance_input.setValue(20)
        params_layout.addWidget(self.distance_input, 1, 1)

        params_layout.addWidget(QLabel("Толщина черных линий:"), 2, 0)
        self.thickness_input = QSpinBox()
        self.thickness_input.setRange(1, 100)
        self.thickness_input.setValue(10)
        params_layout.addWidget(self.thickness_input, 2, 1)

        layout.addWidget(params_group)

        # Кнопки управления
        buttons_layout = QHBoxLayout()

        self.create_btn = QPushButton("Создать")
        self.create_btn.clicked.connect(self.on_create)
        buttons_layout.addWidget(self.create_btn)

        self.load_btn = QPushButton("Загрузить")
        self.load_btn.clicked.connect(self.on_load)
        buttons_layout.addWidget(self.load_btn)

        self.save_btn = QPushButton("Сохранить")
        self.save_btn.clicked.connect(self.on_save)
        buttons_layout.addWidget(self.save_btn)

        layout.addLayout(buttons_layout)

    def on_create(self):
        thickness = self.thickness_input.value()
        space = self.distance_input.value()
        period = thickness + space  # Вычисляем ПЕРИОД
        params = {
            'angle': self.angle_input.value(),
            'distance': period,
            'thickness': self.thickness_input.value()
        }
        self.on_create_callback(params)

    def on_load(self):
        self.on_load_callback()

    def on_save(self):
        self.on_save_callback()

    def get_params(self):
        thickness = self.thickness_input.value()
        space = self.distance_input.value()
        period = thickness + space  # Вычисляем ПЕРИОД
        return {
            'angle': self.angle_input.value(),
            'distance': period,
            'thickness': self.thickness_input.value()
        }

    def set_params(self, angle: int, distance: int, thickness: int):
        """Установка параметров извне"""
        space = max(1, distance - thickness)
        self.angle_input.setValue(angle)
        self.distance_input.setValue(space)
        self.thickness_input.setValue(thickness)

    def analyze_load(self, image_array: np.ndarray):
        """
        Автоматический анализ изображения с черными линиями на белом фоне
        """
        if image_array is not None:
            try:
                results = self.line_analyzer.analyze_image(image_array)

                if results['confidence'] > 0.3:
                    # Округляем значения для интерфейса
                    angle = max(0, min(360, round(results['line_angle'])))
                    period = max(1, min(100, round(results['line_step'])))
                    thickness = max(1, min(100, round(results['line_width'])))
                    space = max(1, period - thickness)
                    # Устанавливаем значения в поля ввода
                    self.angle_input.setValue(angle)
                    self.distance_input.setValue(space)
                    self.thickness_input.setValue(thickness)

                    return results
                else:
                    print(f"Низкая уверенность анализа: {results['confidence']}")
                    return None

            except Exception as e:
                print(f"Ошибка анализа изображения: {e}")
                return None
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ муаровых картин")
        self.setGeometry(100, 100, 900, 600)

        self.storage = Storage()
        self.wr_control_panel = None
        self.project_btn = None
        self.stop_project_btn = None  # Перемещена на вкладку ОР
        self.ir_control_panel = None

        # == Атрибуты для кнопок (для update_ui_state) ==
        self.or_preview_btn = None
        self.or_capture_btn = None
        self.or_load_btn = None
        self.or_save_btn = None
        self.process_or_btn = None
        self.analyze_moire_btn = None
        # ==

        self.setup_ui()

    def setup_ui(self):
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Правая панель - содержимое
        self.content_stack = QTabWidget()

        main_layout.addWidget(self.content_stack)

        # Создаем вкладки для каждого раздела
        self.setup_setting_tab()
        self.setup_working_raster_tab()
        self.setup_object_raster_tab()
        self.setup_processing_tab()
        self.setup_imaginary_raster_tab()
        self.setup_analysis_tab()

        # == Устанавливаем начальное состояние UI ==
        self.update_ui_state()
        # ==

    def setup_setting_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 1. Группа для пользовательских параметров
        params_group = QGroupBox("Параметры калибровки/анализа")
        params_layout = QGridLayout(params_group)

        params_layout.addWidget(QLabel("Расстояние от проектора до объекта - О`C (м):"), 0, 0)
        self.settings_dist1_input = QDoubleSpinBox()
        self.settings_dist1_input.setRange(0.01, 10000.0)
        self.settings_dist1_input.setSingleStep(0.01)
        self.settings_dist1_input.setDecimals(2)
        self.settings_dist1_input.setValue(self.storage.setting_distance_1)
        params_layout.addWidget(self.settings_dist1_input, 0, 1)

        params_layout.addWidget(QLabel("Расстояние от плоскости мнимого растра до объекта - K1C (м):"), 1, 0)
        self.settings_dist2_input = QDoubleSpinBox()
        self.settings_dist2_input.setRange(0.01, 10000.0)
        self.settings_dist2_input.setSingleStep(0.01)
        self.settings_dist2_input.setDecimals(2)
        self.settings_dist2_input.setValue(self.storage.setting_distance_2)
        params_layout.addWidget(self.settings_dist2_input, 1, 1)

        params_layout.addWidget(QLabel("Угол β между оптическими осями видеокамеры и проектора\nв точке их пересечения на объекте (°):"), 2, 0)
        self.settings_angle_input = QDoubleSpinBox()
        self.settings_angle_input.setRange(-360.0, 360.0)
        self.settings_angle_input.setValue(self.storage.setting_angle)
        params_layout.addWidget(self.settings_angle_input, 2, 1)

        lable_scheme = QLabel("<b>СХЕМА</b> <i>(наведите курсор для просмотра)</i>")
        lable_scheme.setTextFormat(Qt.TextFormat.RichText)
        lable_scheme.setToolTip("<img src='img/scheme.png' width='700'>")
        lable_scheme.setStyleSheet("""
            QLabel {
                border: 1px solid #e0e0e0;
                border-radius: 2px;
                padding: 8px;
            }
            QLabel:hover {
                background-color: #e0e0e0;
            }
        """)
        lable_scheme.setAlignment(Qt.AlignmentFlag.AlignCenter)
        params_layout.addWidget(lable_scheme, 3, 0, 1, 2)

        layout.addWidget(params_group)

        # 2. Группа настроек проектора
        projector_group = QGroupBox("Настройки проектора")
        projector_layout = QGridLayout(projector_group)

        projector_layout.addWidget(QLabel("Выберите проектор:"), 0, 0)
        self.projector_selector = QComboBox()
        projector_layout.addWidget(self.projector_selector, 0, 1)

        self.projector_res_label = QLineEdit()
        self.projector_res_label.setReadOnly(True)
        self.projector_res_label.setText("Разрешение: (не выбрано)")
        projector_layout.addWidget(self.projector_res_label, 1, 1)

        refresh_projectors_btn = QPushButton("Обновить список доступных проекторов")
        refresh_projectors_btn.clicked.connect(self.populate_projector_list)
        projector_layout.addWidget(refresh_projectors_btn, 2, 1)

        self.projector_selector.currentIndexChanged.connect(self.on_projector_selected)

        layout.addWidget(projector_group)

        # 3. Группа настроек камеры
        camera_group = QGroupBox("Настройки видеокамеры")
        camera_layout = QGridLayout(camera_group)

        camera_layout.addWidget(QLabel("Выберите камеру:"), 0, 0)
        self.camera_selector = QComboBox()
        camera_layout.addWidget(self.camera_selector, 0, 1)

        camera_layout.addWidget(QLabel("Выберите разрешение:"), 1, 0)
        self.camera_res_selector = QComboBox()
        self.camera_res_selector.setEnabled(False)
        camera_layout.addWidget(self.camera_res_selector, 1, 1)

        refresh_cameras_btn = QPushButton("Обновить список доступных камер")
        refresh_cameras_btn.clicked.connect(self.populate_camera_list)
        camera_layout.addWidget(refresh_cameras_btn, 2, 1)

        self.camera_selector.currentIndexChanged.connect(self.on_camera_selected)

        layout.addWidget(camera_group)

        # 4. Кнопка сохранения
        save_btn = QPushButton("Записать настройки")
        save_btn.clicked.connect(self.save_app_settings)
        layout.addWidget(save_btn)

        layout.addStretch(1)  # Добавляем растяжение, чтобы прижать все вверх
        self.content_stack.addTab(tab, "Настройки")

        # 5. Запускаем поиск оборудования при старте
        self.populate_projector_list()
        self.populate_camera_list()

    def setup_working_raster_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Панель управления
        control_panel = RasterControlPanel(
            "Рабочий растр",
            self.create_working_raster,
            self.load_working_raster,
            self.save_working_raster
        )
        layout.addWidget(control_panel)
        self.wr_control_panel = control_panel

        project_group = QGroupBox()
        project_layout = QHBoxLayout(project_group)

        self.project_btn = QPushButton("Спроецировать рабочий растр")
        self.project_btn.clicked.connect(self.project_working_raster)
        self.project_btn.setEnabled(False)  # Изначально неактивна
        project_layout.addWidget(self.project_btn)

        # == КНОПКА СТОП ПЕРЕМЕЩЕНА ==
        # self.stop_project_btn = QPushButton("Остановить проецирование")
        # self.stop_project_btn.clicked.connect(self.stop_projection)
        # self.stop_project_btn.setEnabled(False)
        # project_layout.addWidget(self.stop_project_btn)

        layout.addWidget(project_group)

        # Превью изображения
        self.wr_preview = ImageViewer()
        layout.addWidget(self.wr_preview, 1)

        # Информация о файле
        self.wr_info = QLineEdit()
        self.wr_info.setReadOnly(True)
        self.wr_info.setPlaceholderText("Информация о файле...")
        layout.addWidget(self.wr_info)

        tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.content_stack.addTab(tab, "Рабочий растр")

    def setup_object_raster_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Кнопки управления
        buttons_layout = QHBoxLayout()

        self.or_preview_btn = QPushButton("Включить камеру")
        self.or_preview_btn.clicked.connect(self.start_camera_preview)
        buttons_layout.addWidget(self.or_preview_btn)

        self.or_capture_btn = QPushButton("Сделать снимок")
        self.or_capture_btn.clicked.connect(self.capture_object_raster)
        buttons_layout.addWidget(self.or_capture_btn)

        self.or_load_btn = QPushButton("Загрузить")
        self.or_load_btn.clicked.connect(self.load_object_raster)
        buttons_layout.addWidget(self.or_load_btn)

        self.or_save_btn = QPushButton("Сохранить")
        self.or_save_btn.clicked.connect(self.save_object_raster)
        buttons_layout.addWidget(self.or_save_btn)

        layout.addLayout(buttons_layout)

        # == КНОПКА СТОП ДОБАВЛЕНА СЮДА ==
        self.stop_project_btn = QPushButton("Остановить проецирование")
        self.stop_project_btn.clicked.connect(self.stop_projection)
        self.stop_project_btn.setEnabled(False)
        layout.addWidget(self.stop_project_btn)
        # ==

        # Превью изображения
        # self.or_preview = ImageViewer()
        # layout.addWidget(self.or_preview)
        #####
        # Превью изображения с камеры
        self.camera_preview = ImageViewer()
        layout.addWidget(self.camera_preview)

        # Таймер для обновления preview камеры
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_preview)
        #####

        # Информация о файле
        self.or_info = QLineEdit()
        self.or_info.setReadOnly(True)
        self.or_info.setPlaceholderText("Информация о файле...")
        layout.addWidget(self.or_info)

        self.content_stack.addTab(tab, "Объектный растр")

    def setup_imaginary_raster_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Панель управления
        control_panel = RasterControlPanel(
            "Мнимый растр",
            self.create_imaginary_raster,
            self.load_imaginary_raster,
            self.save_imaginary_raster
        )
        layout.addWidget(control_panel)
        self.ir_control_panel = control_panel

        # Превью изображения
        self.ir_preview = ImageViewer()
        layout.addWidget(self.ir_preview, 1)

        # Информация о файле
        self.ir_info = QLineEdit()
        self.ir_info.setReadOnly(True)
        self.ir_info.setPlaceholderText("Информация о файле...")
        layout.addWidget(self.ir_info)

        tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.content_stack.addTab(tab, "Мнимый растр")

    def setup_processing_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        # === НАЧАЛО ИЗМЕНЕНИЯ ===
        # Группа для параметров обработки
        params_group = QGroupBox("Параметры обработки")
        params_layout = QGridLayout(params_group)

        # Добавляем поле для ввода порога бинаризации
        params_layout.addWidget(QLabel("Порог бинаризации (0-255):"), 0, 0)
        self.processing_threshold_input = QSpinBox()
        self.processing_threshold_input.setRange(0, 255)
        # Устанавливаем 50, т.к. это было значение по умолчанию в ImageProcessor.threshold
        self.processing_threshold_input.setValue(50)
        params_layout.addWidget(self.processing_threshold_input, 0, 1)

        layout.addWidget(params_group)
        # === КОНЕЦ ИЗМЕНЕНИЯ ===

        # Кнопки обработки
        self.process_or_btn = QPushButton("Обработать объектный растр")
        self.process_or_btn.clicked.connect(self.process_object_raster)
        layout.addWidget(self.process_or_btn)

        # Превью обработанного изображения
        self.processed_preview = ImageViewer()
        layout.addWidget(self.processed_preview, 1)
        tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.content_stack.addTab(tab, "Обработка")

    def setup_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Кнопка анализа
        self.analyze_moire_btn = QPushButton("Анализировать муаровую картину")
        self.analyze_moire_btn.clicked.connect(self.analyze_moire_pattern)
        layout.addWidget(self.analyze_moire_btn)

        # Результаты анализа
        results_group = QGroupBox("Результаты анализа")
        results_layout = QGridLayout(results_group)

        results_layout.addWidget(QLabel("Вид деформации:"), 0, 0)
        self.deformation_type = QLineEdit()
        self.deformation_type.setReadOnly(True)
        results_layout.addWidget(self.deformation_type, 0, 1)

        results_layout.addWidget(QLabel("Отклонение:"), 1, 0)
        self.deviation = QLineEdit()
        self.deviation.setReadOnly(True)
        results_layout.addWidget(self.deviation, 1, 1)

        layout.addWidget(results_group)

        # Превью муаровой картины
        self.moire_preview = ImageViewer()
        layout.addWidget(self.moire_preview, 1)
        tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.content_stack.addTab(tab, "Анализ")

    def update_ui_state(self):
        """
        Обновляет состояние (активность) вкладок и кнопок
        в зависимости от текущего этапа рабочего процесса.
        """

        current_tab_index = self.content_stack.currentIndex()

        # --- Определяем ключи ---
        WR = TextureType.WORKING_RASTER
        OR = TextureType.OBJECT_RASTER
        IR = TextureType.IMAGINARY_RASTER
        PROCESSED = TextureType.PROCESSED_OR

        # --- Собираем информацию о состоянии ---
        settings_saved = self.storage.settings_saved
        wr_exists = WR in self.storage.objects and self.storage.objects[WR] is not None
        is_projecting = api.is_projecting()
        or_exists = OR in self.storage.objects and self.storage.objects[OR] is not None
        ir_exists = IR in self.storage.objects and self.storage.objects[IR] is not None
        processed_exists = PROCESSED in self.storage.objects and self.storage.objects[PROCESSED] is not None

        has_projector = bool(self.storage.available_screens)
        has_camera = bool(self.storage.available_cameras)

        # --- Состояние кнопок камеры ---
        # is_previewing должен быть True, только если и таймер активен,
        # и бэкенд-камера физически работает
        is_previewing = self.camera_timer.isActive() and api.is_camera_processing()

        # --- Сбрасываем состояние ---
        # Вкладки
        self.content_stack.setTabEnabled(1, False)
        self.content_stack.setTabEnabled(2, False)
        self.content_stack.setTabEnabled(3, False)
        self.content_stack.setTabEnabled(4, False)
        self.content_stack.setTabEnabled(5, False)

        # Кнопки (по умолчанию выключены)
        self.wr_control_panel.create_btn.setEnabled(False)
        self.wr_control_panel.load_btn.setEnabled(False)
        self.wr_control_panel.save_btn.setEnabled(False)
        self.project_btn.setEnabled(False)

        self.or_preview_btn.setEnabled(False)
        self.or_capture_btn.setEnabled(False)
        self.or_load_btn.setEnabled(False)
        self.or_save_btn.setEnabled(False)
        self.stop_project_btn.setEnabled(False)

        self.ir_control_panel.create_btn.setEnabled(False)
        self.ir_control_panel.load_btn.setEnabled(False)
        self.ir_control_panel.save_btn.setEnabled(False)

        self.process_or_btn.setEnabled(False)
        self.analyze_moire_btn.setEnabled(False)

        # --- Этап 0: Настройки ---
        self.content_stack.setTabEnabled(0, True)
        if not settings_saved:
            self.content_stack.setCurrentIndex(current_tab_index)
            return

        # --- Этап 1: Рабочий растр ---
        self.content_stack.setTabEnabled(1, True)
        self.wr_control_panel.create_btn.setEnabled(True)
        self.wr_control_panel.load_btn.setEnabled(True)

        if wr_exists:
            self.wr_control_panel.save_btn.setEnabled(True)
            self.update_project_buttons_state()  # Обновляет self.project_btn

        # --- [ИСПРАВЛЕНИЕ ЛОГИКИ] ---
        # Мы больше не выходим, если нет проекции.
        # Вместо этого мы управляем активностью вкладок.

        # --- Этап 2: Объектный растр ---
        # Этап 2 активен, ТОЛЬКО если идет проекция
        self.content_stack.setTabEnabled(2, is_projecting)
        if is_projecting:
            self.or_preview_btn.setEnabled(has_camera and not is_previewing)
            self.or_capture_btn.setEnabled(is_previewing)
            self.or_load_btn.setEnabled(True)
            self.stop_project_btn.setEnabled(True)

            if or_exists:
                self.or_save_btn.setEnabled(True)

        # --- Последующие этапы ---
        # Они зависят от наличия ОР и МР, а не от проекции

        if not or_exists:
            self.content_stack.setCurrentIndex(current_tab_index)
            return


        # --- Этап 4: Обработка ---
        self.content_stack.setTabEnabled(3, True)
        self.process_or_btn.setEnabled(True)

        if not processed_exists:
            self.content_stack.setCurrentIndex(current_tab_index)
            return

        # --- Этап 3: Мнимый растр ---
        self.content_stack.setTabEnabled(4, True)
        self.ir_control_panel.create_btn.setEnabled(True)
        self.ir_control_panel.load_btn.setEnabled(True)

        if ir_exists:
            self.ir_control_panel.save_btn.setEnabled(True)

        if not ir_exists:
            self.content_stack.setCurrentIndex(current_tab_index)
            return


        # --- Этап 5: Анализ ---
        self.content_stack.setTabEnabled(5, True)
        self.analyze_moire_btn.setEnabled(True)

        self.content_stack.setCurrentIndex(current_tab_index)

    def create_working_raster(self, params):
        reply = QMessageBox.question(
            self,
            'Подтверждение создания',
            f'Вы уверены, что хотите создать новый рабочий растр со следующими параметрами?\n\n'
            f'Параметры:\n'
            f'• Угол: {params['angle']}°\n'
            f'• Шаг линий: {params['distance'] - params['thickness']} px\n'
            f'• Толщина линий: {params['thickness']} px',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            print("Создание рабочего растра отменено")
            return
        try:
            #win_settings = WindowSettings()
            # --- ИЗМЕНЕНИЕ ---
            # Используем разрешение проектора из настроек
            win_settings: WindowSettings
            proj_index = self.storage.selected_projector_index

            if proj_index is not None and proj_index < len(self.storage.available_screens):
                screen = self.storage.available_screens[proj_index]
                win_settings = WindowSettings(width=screen.width, height=screen.height)
                print(f"Создание растра с разрешением проектора: {screen.width}x{screen.height}")
            else:
                QMessageBox.warning(self, "Внимание",
                                    "Проектор не выбран. Растр будет создан с разрешением по умолчанию (1024x768).")
                win_settings = WindowSettings()  # Разрешение по умолчанию
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            raster_settings = api.raster_settings(
                params['angle'], params['distance'], params['thickness']
            )
            raster = api.create_raster(win_settings, raster_settings)
            self.storage.objects[TextureType.WORKING_RASTER] = raster
            self.wr_preview.set_image(raster.image)
            self.wr_info.setText("Создан новый рабочий растр")
            # self.set_project_buttons_enabled(True) # Управляется update_ui_state
            self.update_ui_state()  # ==
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка создания растра: {e}")

    def load_working_raster(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Загрузить рабочий растр", "", "Images (*.png *.jpg *.bmp *.tiff)"
            )
            if filename:
                raster = api.load_image_by_tag(filename, "raster")
                self.storage.objects[TextureType.WORKING_RASTER] = raster
                self.wr_preview.set_image(raster.image)
                # self.set_project_buttons_enabled(True) # Управляется update_ui_state
                # АВТОМАТИЧЕСКИЙ АНАЛИЗ ПРИ ЗАГРУЗКЕ
                results = self.wr_control_panel.analyze_load(raster.image)
                if results:
                    self.wr_info.setText(
                        f"Загружен: {filename.split('/')[-1]} | "
                        f"Угол: {results['line_angle']:.1f}°, "
                        f"Шаг: {results['line_step']:.1f}px, "
                        f"Ширина: {results['line_width']:.1f}px"
                    )
                else:
                    self.wr_info.setText(f"Загружен: {filename.split('/')[-1]}")
                self.update_ui_state()  # ==
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки: {e}")

    def save_working_raster(self):
        raster = self.storage.objects[TextureType.WORKING_RASTER]
        api.save_raster_image(raster.image, self)

    def project_working_raster(self):
        """Проецирование рабочего растра на проектор"""
        try:
            if TextureType.WORKING_RASTER not in self.storage.objects:
                QMessageBox.warning(self, "Предупреждение", "Рабочий растр отсутствует")
                return

            raster = self.storage.objects[TextureType.WORKING_RASTER]

            # --- ИЗМЕНЕНИЕ ---
            # Убираем диалог выбора экрана, используем сохраненные настройки
            screen_number = self.storage.selected_projector_index

            if screen_number is None or screen_number >= len(self.storage.available_screens):
                QMessageBox.warning(self, "Ошибка",
                                    "Проектор не выбран или не найден. "
                                    "Проверьте вкладку 'Настройки' и обновите список экранов.")
                return
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            # Проецируем изображение
            success = api.project_image(raster.image, screen_number)

            if success:
                # Обновляем состояние кнопок
                self.update_ui_state()
                QMessageBox.information(
                    self,
                    "Проецирование",
                    f"Рабочий растр проецируется на экран #{screen_number}\n\n"
                    f"Для остановки нажмите кнопку 'Остановить проецирование' или ESC в окне проектора."
                )
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось запустить проецирование")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка проецирования: {e}")
    def stop_projection(self):
        """Остановка проецирования"""
        try:
            api.stop_projection()
            # Обновляем состояние кнопок
            self.update_ui_state()  # ==
            QMessageBox.information(self, "Проецирование", "Проецирование остановлено")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка остановки проецирования: {e}")

    def get_available_screens(self):
        """Получить список доступных экранов"""
        try:
            import screeninfo
            return screeninfo.get_monitors()
        except:
            return [type('Screen', (), {'name': 'Основной экран', 'width': 1920, 'height': 1080})]

    def update_project_buttons_state(self):
        """Обновить состояние кнопки проецирования (вызывается из update_ui_state)"""
        is_projecting = api.is_projecting()
        has_projector = bool(self.storage.available_screens)
        wr_exists = TextureType.WORKING_RASTER in self.storage.objects

        # Кнопка "Проецировать" активна, *только* если:
        # 1. Проецирование *не* идет
        # 2. Рабочий растр *существует*
        # 3. Проектор *найден*
        self.project_btn.setEnabled(not is_projecting and wr_exists and has_projector)


    def start_camera_preview(self):
        """Запуск предпросмотра камеры"""
        try:
            # Читаем сохраненные настройки
            cam_id = self.storage.selected_camera_id
            cam_res = self.storage.selected_camera_resolution

            if cam_id is None:
                QMessageBox.warning(self, "Ошибка", "Камера не выбрана во вкладке 'Настройки'")
                return

            print(f"Запуск камеры ID: {cam_id} с разрешением {cam_res}")

            cam_settings = api.camera_settings(width=cam_res[0], height=cam_res[1])

            # Передаем ID и настройки в API
            api.start_camera_preview(src=cam_id, camera_settings_=cam_settings)

            self.camera_timer.start(33)  # ~30 FPS
            self.or_info.setText("Камера включена - наведите и нажмите 'Сделать снимок'")
            self.update_ui_state()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить камеру: {e}")

    def update_camera_preview(self):
        """Обновление preview камеры"""

        if not api.is_camera_processing():
            self.camera_timer.stop()
            self.update_ui_state()
            return

        frame = api.get_camera_frame()

        if frame is not None:
            self.camera_preview.set_image(frame)
        else:
            pass

    def capture_object_raster(self):
        """Захват кадра с камеры"""
        try:
            # Останавливаем предпросмотр
            self.camera_timer.stop()

            # Захватываем изображение
            captured_image = api.capture_from_camera()

            if captured_image and captured_image.image is not None:
                self.storage.objects[TextureType.OBJECT_RASTER] = captured_image
                self.camera_preview.set_image(captured_image.image)
                self.or_info.setText("Снимок сделан и сохранен в память")
                self.update_ui_state()  # ==
            else:
                raise Exception("Не удалось захватить изображение")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка захвата: {e}")
            self.update_ui_state()

    def load_object_raster(self):
        try:
            if self.camera_timer.isActive():
                self.camera_timer.stop()
                api.stop_camera_preview()

            filename, _ = QFileDialog.getOpenFileName(
                self, "Загрузить объектный растр", "", "Images (*.png *.jpg *.bmp *.tiff)"
            )
            if filename:
                raster = api.load_image_by_tag(filename, "raw")
                self.storage.objects[TextureType.OBJECT_RASTER] = raster
                self.camera_preview.set_image(raster.image)
                self.or_info.setText(f"Загружен: {filename.split('/')[-1]}")
                self.update_ui_state()  # ==
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки: {e}")

    def save_object_raster(self):
        raster = self.storage.objects[TextureType.OBJECT_RASTER]
        api.save_camera_image(raster.image)

    def save_imaginary_raster(self):
        raster = self.storage.objects[TextureType.IMAGINARY_RASTER]
        api.save_raster_image(raster.image)

    def create_imaginary_raster(self, params):
        reply = QMessageBox.question(
            self,
            'Подтверждение создания',
            f'Вы уверены, что хотите создать новый мнимый растр со следующими параметрами?\n\n'
            f'Параметры:\n'
            f'• Угол: {params['angle']}°\n'
            f'• Шаг линий: {params['distance'] - params['thickness']} px\n'
            f'• Толщина линий: {params['thickness']} px',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            print("Создание рабочего растра отменено")
            return

        try:
            cam_res = self.storage.selected_camera_resolution
            if not cam_res or len(cam_res) != 2:
                QMessageBox.warning(self, "Ошибка", "Разрешение камеры не задано в 'Настройках'.")
                return
            win_settings = WindowSettings(width=cam_res[0], height=cam_res[1])

            raster_settings = api.raster_settings(
                params['angle'], params['distance'], params['thickness']
            )
            raster = api.create_raster(win_settings, raster_settings)
            self.storage.objects[TextureType.IMAGINARY_RASTER] = raster
            self.ir_preview.set_image(raster.image)
            self.ir_info.setText("Создан новый мнимый растр")
            self.update_ui_state()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка создания растра: {e}")

    def load_imaginary_raster(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Загрузить мнимый растр", "", "Images (*.png *.jpg *.bmp *.tiff)"
            )
            if filename:
                raster = api.load_image_by_tag(filename, "raster")
                self.storage.objects[TextureType.IMAGINARY_RASTER] = raster
                self.ir_preview.set_image(raster.image)
                # self.ir_info.setText(f"Загружен: {filename.split('/')[-1]}")
                # АВТОМАТИЧЕСКИЙ АНАЛИЗ ПРИ ЗАГРУЗКЕ
                results = self.ir_control_panel.analyze_load(raster.image)
                if results:
                    self.ir_info.setText(
                        f"Загружен: {filename.split('/')[-1]} | "
                        f"Угол: {results['line_angle']:.1f}°, "
                        f"Шаг: {results['line_step']:.1f}px, "
                        f"Ширина: {results['line_width']:.1f}px"
                    )
                else:
                    self.ir_info.setText(f"Загружен: {filename.split('/')[-1]}")
                self.update_ui_state()  # ==
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки: {e}")

    def process_object_raster(self):
        try:
            if TextureType.OBJECT_RASTER not in self.storage.objects:
                QMessageBox.warning(self, "Предупреждение", "Объектный растр отсутствует.")
                return

            or_image = self.storage.objects[TextureType.OBJECT_RASTER]

            if or_image is None:
                QMessageBox.warning(self, "Ошибка", "Объектный растр не содержит данных")
                return

            if or_image.image is None:
                QMessageBox.warning(self, "Ошибка", "Изображение объектного растра равно None")
                return

            print(f"Размер изображения: {or_image.image.shape if or_image.image is not None else 'None'}")
            print(f"Тип источника: {or_image.source}")

            threshold_value = self.processing_threshold_input.value()
            print(f"Обработка с порогом: {threshold_value}")


            #ws = WindowSettings(1024, 768)
            cam_res = self.storage.selected_camera_resolution
            if not cam_res or len(cam_res) != 2:
                QMessageBox.warning(self, "Ошибка", "Разрешение камеры не задано в 'Настройках'.")
                return
            ws = WindowSettings(width=cam_res[0], height=cam_res[1])

            processed_image = api.processor_pipeline(or_image, threshold_value, ws)

            if processed_image is None or processed_image.image is None:
                QMessageBox.warning(self, "Ошибка", "Обработка вернула None")
                return

            self.storage.objects[TextureType.PROCESSED_OR] = processed_image
            self.processed_preview.set_image(processed_image.image)
            self.update_ui_state()  # ==
        except Exception as e:
            print(f"Ошибка обработки растра: {e}")

    def analyze_moire_pattern(self):
        try:
            if (TextureType.WORKING_RASTER not in self.storage.objects or
                    TextureType.PROCESSED_OR not in self.storage.objects or
                    TextureType.IMAGINARY_RASTER not in self.storage.objects):
                QMessageBox.warning(self, "Предупреждение", "Не все растры загружены")
                return

            # analyzer = Analizator(
            #     self.storage.objects[TextureType.WORKING_RASTER],
            #     self.storage.objects[TextureType.IMAGINARY_RASTER],
            #     self.storage.objects[TextureType.PROCESSED_OR]
            # )
                # (Стало) Получаем разрешение и калибровку из storage
            cam_res = self.storage.selected_camera_resolution
            if not cam_res or len(cam_res) != 2:
                QMessageBox.warning(self, "Ошибка", "Разрешение камеры не задано в 'Настройках'.")
                return

            analyzer = Analizator(
                self.storage.objects[TextureType.WORKING_RASTER],
                self.storage.objects[TextureType.IMAGINARY_RASTER],
                self.storage.objects[TextureType.PROCESSED_OR],
                # Передаем параметры из настроек
                target_width=cam_res[0],
                target_height=cam_res[1],
                calibration_p50=self.storage.calibration_p50,
                calibration_ratio=self.storage.calibration_ratio
            )
            # === КОНЕЦ ИЗМЕНЕНИЯ ===

            poster = analyzer.poster(select_persentile90=True)
            deform_type, deviation_value = analyzer.get_deformation_analysis()

            self.storage.objects[TextureType.MOIRE_PATTERN] = poster
            poster_image = cv2.cvtColor(poster, cv2.COLOR_BGR2RGB)
            self.moire_preview.set_image(poster_image)

            # Отображаем корректные данные
            self.deformation_type.setText(BY_DEFORM_MSG.get(deform_type, "Неизвестно"))
            self.deviation.setText(f"{deviation_value:.2f} px (медиана)")

            self.update_ui_state()  # ==

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка анализа: {e}")

    def populate_projector_list(self):
        """Находит доступные экраны и заполняет QComboBox"""
        self.projector_selector.clear()
        self.storage.available_screens = self.get_available_screens()

        if not self.storage.available_screens:
            self.projector_selector.addItem("Экраны не найдены")
            self.projector_selector.setEnabled(False)
            self.update_ui_state()  # ==
            return

        self.projector_selector.setEnabled(True)
        for i, screen in enumerate(self.storage.available_screens):
            name = f"Экран {i}: {getattr(screen, 'name', 'N/A')} ({screen.width}x{screen.height})"
            self.projector_selector.addItem(name, i)  # Сохраняем индекс в данных

        self.on_projector_selected()  # Обновить разрешение
        #self.update_ui_state()  # ==

    def on_projector_selected(self):
        """Обновляет текстовое поле с разрешением при выборе проектора"""
        try:
            index = self.projector_selector.currentData()
            if index is not None:
                screen = self.storage.available_screens[index]
                self.projector_res_label.setText(f"Разрешение: {screen.width}x{screen.height}")
        except Exception as e:
            print(f"Ошибка выбора проектора: {e}")
            self.projector_res_label.setText("Разрешение: Ошибка")

    def populate_camera_list(self):
        """Находит доступные камеры (простым перебором)"""
        # Сохраняем текущий выбор, чтобы восстановить его
        current_data = self.camera_selector.currentData()

        self.camera_selector.clear()
        self.storage.available_cameras = {}

        for i in range(5):  # Проверяем первые 5 ID
            try:
                # cap = cv2.VideoCapture(i)
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                # cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
                if cap.isOpened():
                    self.storage.available_cameras[i] = f"Камера {i}"
                    self.camera_selector.addItem(f"Камера {i}", i)
                    cap.release()
                else:
                    break
            except Exception:
                break

        if not self.storage.available_cameras:
            self.camera_selector.addItem("Камеры не найдены")
            self.camera_selector.setEnabled(False)
        else:
            self.camera_selector.setEnabled(True)
            # Восстанавливаем предыдущий выбор, если он еще доступен
            if current_data is not None:
                index = self.camera_selector.findData(current_data)
                if index >= 0:
                    self.camera_selector.setCurrentIndex(index)

        if self.camera_selector.isEnabled():
            self.on_camera_selected()

        #self.update_ui_state()  # ==

    def save_app_settings(self):
        """Сохраняет выбранные настройки в self.storage"""
        try:
            # Сохраняем параметры
            self.storage.setting_distance_1 = self.settings_dist1_input.value()
            self.storage.setting_distance_2 = self.settings_dist2_input.value()
            self.storage.setting_angle = self.settings_angle_input.value()

            # Сохраняем проектор
            self.storage.selected_projector_index = self.projector_selector.currentData()

            # Сохраняем камеру
            self.storage.selected_camera_id = self.camera_selector.currentData()
            # == ИЗМЕНЕНИЕ ==
            # Теперь мы берем разрешение из .currentData()
            res_data = self.camera_res_selector.currentData()  # Это будет кортеж (w, h)
            if res_data:
                self.storage.selected_camera_resolution = (res_data[0], res_data[1])
            else:
                # Если список пуст или "Ошибка", сбрасываем
                res_text = self.camera_res_selector.currentText()
                if "x" in res_text:  # Попытка спасти, если .currentData() не задан
                    w, h = map(int, res_text.split('x'))
                    self.storage.selected_camera_resolution = (w, h)
                else:
                    self.storage.selected_camera_resolution = (640, 480)  # Fallback
            # == КОНЕЦ ИЗМЕНЕНИЯ ==

            # == Активируем следующий этап ==
            self.storage.settings_saved = True
            self.update_ui_state()
            # ==

            QMessageBox.information(self, "Сохранено", "Настройки успешно сохранены в памяти приложения.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить настройки: {e}")

    def get_supported_resolutions(self, camera_id: int) -> list:
        """
        Проверяет камеру по списку стандартных разрешений.
        Возвращает список поддерживаемых (ширина, высота).
        """
        # Список стандартных разрешений для проверки
        COMMON_RESOLUTIONS = [
            (160, 120),
            (320, 240),
            (640, 480),
            (800, 600),
            (1024, 768),
            (1280, 720),
            (1280, 1024),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160)
        ]

        supported_resolutions = []
        cap = None

        try:
            # Используем бэкенд CAP_DSHOW для более надежной работы .set() в Windows
            # Если у вас Linux, можно просто cv2.VideoCapture(camera_id)
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return []  # Камера недоступна

            for w, h in COMMON_RESOLUTIONS:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

                # Считываем фактические установленные значения
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Если камера приняла наше разрешение, добавляем его
                if actual_w == w and actual_h == h:
                    supported_resolutions.append((w, h))

        except Exception as e:
            print(f"Ошибка при проверке разрешений камеры {camera_id}: {e}")
        finally:
            if cap:
                cap.release()

        return supported_resolutions

    def on_camera_selected(self):
        """
        Вызывается при выборе камеры в QComboBox.
        Запускает проверку разрешений и заполняет список.
        """
        self.camera_res_selector.clear()

        camera_id = self.camera_selector.currentData()  # Получаем ID камеры
        if camera_id is None:
            self.camera_res_selector.setEnabled(False)
            return

        # Сообщаем пользователю, что идет проверка (т.к. это медленно)
        self.camera_res_selector.setEnabled(False)
        self.camera_res_selector.addItem("Определение разрешений...")
        QApplication.processEvents()  # Обновляем GUI

        try:
            # Запускаем медленную проверку
            resolutions = self.get_supported_resolutions(camera_id)

            self.camera_res_selector.clear()  # Очищаем "Загрузку..."

            if not resolutions:
                self.camera_res_selector.addItem("Разрешения не найдены")
                self.camera_res_selector.setEnabled(False)
            else:
                # Заполняем список найденными разрешениями
                for w, h in resolutions:
                    self.camera_res_selector.addItem(f"{w}x{h}", (w, h))  # Сохраняем (w, h) в данные
                self.camera_res_selector.setEnabled(True)

                # Попробуем установить разрешение по умолчанию из storage
                if self.storage.selected_camera_resolution:
                    res_str = f"{self.storage.selected_camera_resolution[0]}x{self.storage.selected_camera_resolution[1]}"
                    index = self.camera_res_selector.findText(res_str)
                    if index >= 0:
                        self.camera_res_selector.setCurrentIndex(index)

        except Exception as e:
            self.camera_res_selector.clear()
            self.camera_res_selector.addItem("Ошибка")
            print(f"Ошибка on_camera_selected: {e}")


class Storage:
    def __init__(self):
        self.objects = {}
        self.analyzator: Optional[Analizator] = None
        self.distance_thick_min_diff = 5

        # ДАННЫЕ КАЛИБРОВКИ (значения по умолчанию)
        self.calibration_p50 = 4.0  # "магическое" пороговое значение P50
        self.calibration_ratio = 2.1  # "магическое" пороговое отношение
        self.is_calibrated = False

        # == НОВЫЕ НАСТРОЙКИ ==
        self.settings_saved = False  # Флаг, что настройки сохранены

        # Настройки, заданные пользователем
        self.setting_distance_1 = 10.0
        self.setting_distance_2 = 20.0
        self.setting_angle = 0.0

        # Настройки оборудования
        self.available_screens = []  # Список объектов screeninfo
        self.selected_projector_index = 0

        self.available_cameras = {}  # Словарь {id: "имя"}
        self.selected_camera_id = 0
        self.selected_camera_resolution = (640, 480)  # (width, height)


class Interface:
    @staticmethod
    def start():
        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        window = MainWindow()
        window.show()

        sys.exit(app.exec())