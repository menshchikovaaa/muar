# import cv2 as cv
# import numpy as np
# import threading
# from typing import Optional
# from settings import CameraSettings
#
#
# class AsyncCamera:
#     def __init__(self, src=0, camera_settings: Optional[CameraSettings] = None):
#         settings = camera_settings or CameraSettings()
#         self.src = src
#         self.cap = cv.VideoCapture(src)
#         self.cap.set(cv.CAP_PROP_FRAME_WIDTH, settings.width)
#         self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, settings.height)
#         self.grabbed, self.frame = self.cap.read()
#         self.processing = False
#
#         self.thread = None
#         self.read_lock = threading.Lock()
#
#     def update(self):
#         while self.processing:
#             grabbed, frame = self.cap.read()
#             with self.read_lock:
#                 self.grabbed = grabbed
#                 self.frame = frame
#
#     def start(self):
#         if self.processing:
#             print("[!] Камера уже начала съемку.")
#             return
#         self.processing = True
#         self.thread = threading.Thread(target=self.update, args=())
#         self.thread.start()
#         return self
#
#     def read(self):
#         with self.read_lock:
#             grabbed = self.grabbed
#             self.frame: np.ndarray = self.frame
#             frame = self.frame.copy()
#         return grabbed, frame
#
#     def stop(self):
#         self.processing = False
#         self.thread: threading.Thread = self.thread
#         self.thread.join()
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.cap.release()
import cv2 as cv
import numpy as np
import threading
import time
from typing import Optional, Callable
from settings import CameraSettings


class AsyncCamera:
    def __init__(self, src=0, camera_settings: Optional[CameraSettings] = None):
        settings = camera_settings or CameraSettings()
        self.src = src
        self.cap = cv.VideoCapture(src)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, settings.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, settings.height)
        self.cap.set(cv.CAP_PROP_FPS, 30)  # Устанавливаем FPS для плавности

        self.grabbed, self.frame = self.cap.read()
        self.processing = False
        self.latest_frame = None
        self.frame_callback = None

        self.thread = None
        self.read_lock = threading.Lock()

    def set_frame_callback(self, callback: Callable):
        """Установка callback-функции для получения кадров"""
        self.frame_callback = callback

    def update(self):
        try:
            while self.processing:
                grabbed, frame = self.cap.read()
                if grabbed:

                    # --- ИСПРАВЛЕНИЕ ---
                    # Проверяем, B&W (2D) или Color (3D) кадр
                    if len(frame.shape) == 2:
                        # Это B&W камера, конвертируем в RGB, чтобы UI его понял
                        frame_rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
                    elif frame.shape[2] == 3:
                        # Это цветная камера, конвертируем BGR в RGB
                        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    else:
                        # Пропускаем странные форматы (например, 4-канальные)
                        continue
                        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                    with self.read_lock:
                        self.grabbed = grabbed
                        self.frame = frame_rgb
                        self.latest_frame = frame_rgb.copy()

                    # Вызываем callback если установлен
                    if self.frame_callback:
                        self.frame_callback(frame_rgb)

                time.sleep(0.03)  # ~30 FPS

        except Exception as e:
            print(f"[!] Ошибка в потоке камеры (возможно, неверный формат кадра?): {e}")
            # Важно: сообщаем, что поток больше не работает
            self.processing = False

    def start(self):
        if self.processing:
            print("[!] Камера уже начала съемку.")
            return self
        self.processing = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def read(self):
        with self.read_lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def capture_frame(self):
        """Захват текущего кадра (возвращает в RGB)"""
        with self.read_lock:
            if self.latest_frame is not None:
                # [ИСПРАВЛЕНИЕ]
                # Убираем конвертацию в BGR.
                # UI (set_image) ожидает RGB.
                # frame_bgr = cv.cvtColor(self.latest_frame, cv.COLOR_RGB2BGR)
                return self.latest_frame.copy()
        return None

    def stop(self):
        self.processing = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.thread = None

    def release(self):
        self.stop()
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()