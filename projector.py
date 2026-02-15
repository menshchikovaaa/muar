import cv2
import numpy as np
import screeninfo
from PyQt6.QtCore import QThread, pyqtSignal


class ProjectorThread(QThread):
    """Поток для отображения изображения на проекторе"""
    finished_signal = pyqtSignal()

    def __init__(self, image, screen_number=1):
        super().__init__()
        self.image = image
        self.screen_number = screen_number
        self.is_running = True

    def run(self):
        try:
            # Получаем информацию о втором экране (проекторе)
            screens = screeninfo.get_monitors()
            if len(screens) > self.screen_number:
                screen = screens[self.screen_number]

                # Создаем полноэкранное окно на проекторе
                window_name = "Projector - Working Raster"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.moveWindow(window_name, screen.x, screen.y)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                # Отображаем изображение
                while self.is_running:
                    cv2.imshow(window_name, self.image)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
                        break

                cv2.destroyWindow(window_name)

            else:
                print(f"Проектор (экран #{self.screen_number}) не найден")

        except Exception as e:
            print(f"Ошибка проецирования: {e}")
        finally:
            self.finished_signal.emit()

    def stop(self):
        self.is_running = False


class ProjectorController:
    """Контроллер для управления проектором"""

    def __init__(self):
        self.projector_thread = None

    def project_image(self, image_array, screen_number=1):
        """Проецировать изображение на указанный экран"""
        if image_array is None:
            raise ValueError("Изображение для проецирования отсутствует")

        # Останавливаем предыдущее проецирование
        self.stop_projection()

        # Конвертируем изображение если нужно
        if len(image_array.shape) == 2:  # Grayscale to BGR
            image_display = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        else:
            image_display = image_array.copy()

        # Запускаем поток проецирования
        self.projector_thread = ProjectorThread(image_display, screen_number)
        self.projector_thread.finished_signal.connect(self._on_projection_finished)
        self.projector_thread.start()

        return True

    def stop_projection(self):
        """Остановить проецирование"""
        if self.projector_thread and self.projector_thread.isRunning():
            self.projector_thread.stop()
            self.projector_thread.wait(1000)  # Ждем до 1 секунды

    def _on_projection_finished(self):
        """Слот для обработки завершения проецирования"""
        self.projector_thread = None

    def is_projecting(self):
        """Проверяет, активно ли проецирование"""
        return self.projector_thread is not None and self.projector_thread.isRunning()