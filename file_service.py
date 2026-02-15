import cv2 as cv
import configparser
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import QDir

@dataclass
class FileServiceConfig:
    raster_filename: str = "raster"
    raster_extension: str = "png"
    settings_filename: str = "settings"
    settings_extension: str = "txt"
    camera_filename: str = "camera"
    camera_extension: str = "png"


class FileService:
    SAVE_DATE_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"

    def __init__(self):
        self.config = FileServiceConfig()

    def _show_success_message(self, parent_widget=None, file_path=None, file_type="файл"):
        """Показать сообщение об успешном сохранении"""
        try:
            if file_path:
                message = f"{file_type.capitalize()} успешно сохранен!\n\nПуть: {file_path}"
            else:
                message = f"{file_type.capitalize()} успешно сохранен!"

            QMessageBox.information(
                parent_widget,
                "Сохранение завершено",
                message,
                QMessageBox.StandardButton.Ok
            )
        except Exception as e:
            print(f"[!] Ошибка при показе сообщения: {e}")
    def _select_directory_dialog(self, parent_widget=None) -> Optional[str]:
        """Открытие Qt диалогового окна для выбора папки"""
        try:
            selected_dir = QFileDialog.getExistingDirectory(
                parent_widget,
                "Выберите папку для сохранения",
                "",  # Начальная директория
                QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
            )

            return selected_dir if selected_dir else None

        except Exception as e:
            print(f"[!] Ошибка при открытии диалогового окна: {e}")
            return None

    def _select_save_file_dialog(self, parent_widget=None) -> Optional[str]:
        """Диалоговое окно для выбора пути сохранения с именем файла"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                parent_widget,
                "Сохранить файл",
                default_filename,
                f"{extension.upper()} files (*.{extension})",
                options=QFileDialog.Option.DontUseNativeDialog  # Опционально, можно убрать
            )

            return file_path if file_path else None

        except Exception as e:
            print(f"[!] Ошибка при открытии диалогового окна: {e}")
            return None

    def _generate_file_path(self, file_type: str, name: str, extension: str, parent_widget=None) -> Dict[str, str]:
        """Генерация пути для сохранения файла"""
        timestamp = datetime.now().strftime(self.SAVE_DATE_FORMAT)
        filename = f"{name}-{timestamp}.{extension}"

        # Выбираем папку через диалоговое окно
        selected_dir = self._select_directory_dialog(parent_widget)
        if not selected_dir:
            raise Exception("Папка для сохранения не выбрана")

        # Создаем полный путь
        full_path = Path(selected_dir) / filename
        # Создаем директории, если они не существуют
        full_path.parent.mkdir(parents=True, exist_ok=True)

        return {
            "full_path": str(full_path),
            "filename": filename,
            "relative_path": filename
        }

    def save_image(self, image, file_type: str, parent_widget=None) -> Dict[str, str]:
        """Сохранение изображения"""
        config_mapping = {
            "raster": (self.config.raster_filename, self.config.raster_extension, "растр"),
            "camera": (self.config.camera_filename, self.config.camera_extension, "изображение камеры")
        }

        if file_type not in config_mapping:
            raise ValueError(f"Неизвестный тип файла: {file_type}")

        filename, extension, file_type_name = config_mapping[file_type]
        path_info = self._generate_file_path(file_type, filename, extension, parent_widget)

        try:
            success = cv.imwrite(path_info["full_path"], image)
            if not success:
                raise Exception("Ошибка записи изображения")
            self._show_success_message(parent_widget, path_info["full_path"], file_type_name)
            return path_info
        except Exception as e:
            print(f"[!] Ошибка сохранения изображения: {e}")
            raise

    def save_settings(self, settings_data: str, parent_widget=None) -> Dict[str, str]:
        """Сохранение настроек"""
        path_info = self._generate_file_path(
            "settings",
            self.config.settings_filename,
            self.config.settings_extension,
            parent_widget
        )

        try:
            with open(path_info["full_path"], 'w', encoding='utf-8') as f:
                f.write(settings_data)
            self._show_success_message(parent_widget, path_info["full_path"], "файл настроек")
            return path_info
        except Exception as e:
            print(f"[!] Ошибка сохранения настроек: {e}")
            raise

    def save_raster_with_settings(self, raster, settings_data: str, parent_widget=None) -> Dict[str, str]:
        """Сохранение растра и настроек вместе"""
        # Для связанных файлов предлагаем выбрать папку один раз
        selected_dir = self._select_directory_dialog(parent_widget)
        if not selected_dir:
            raise Exception("Папка для сохранения не выбрана")

        timestamp = datetime.now().strftime(self.SAVE_DATE_FORMAT)

        # Сохраняем изображение
        raster_filename = f"{self.config.raster_filename}-{timestamp}.{self.config.raster_extension}"
        raster_path = Path(selected_dir) / raster_filename
        success = cv.imwrite(str(raster_path), raster)
        if not success:
            raise Exception("Ошибка записи изображения")

        # Сохраняем настройки
        settings_filename = f"{self.config.settings_filename}-{timestamp}.{self.config.settings_extension}"
        settings_path = Path(selected_dir) / settings_filename
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(settings_data)

        return {
            "raster": {
                "full_path": str(raster_path),
                "filename": raster_filename,
                "relative_path": raster_filename
            },
            "settings": {
                "full_path": str(settings_path),
                "filename": settings_filename,
                "relative_path": settings_filename
            }
        }