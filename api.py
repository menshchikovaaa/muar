from scipy import ndimage
import cv2 as cv
import numpy as np
import dataclasses
from typing import Optional, List, Dict
from model import Color, Point
from image_data import ImageData, SourceType
from paths import get_config_path_data, save_config_path_data, save_raster, save_camera
from settings import WindowSettings, RasterSettings, CameraSettings
from camera import AsyncCamera
from factory import RasterFactory
from processor import ImageProcessor
from file_service import FileService
from projector import ProjectorController


_camera = None
_file_service = FileService()
_projector_controller = ProjectorController()

def gaussian_blur_numpy(image, ksize=(9, 9), sigma=0):
    """
    Applies Gaussian blur to an image using NumPy/SciPy

    Parameters:
    image: ndarray - Input image
    ksize: tuple - Kernel size (width, height)
    sigma: float - Standard deviation. If 0, calculated from kernel size

    Returns:
    ndarray - Blurred image
    """
    # Calculate sigma if not provided
    if sigma == 0:
        sigma = 0.3*((ksize[0]-1)*0.5 - 1) + 0.8

    # Create Gaussian kernel
    x = np.linspace(-(ksize[0]//2), ksize[0]//2, ksize[0])
    y = np.linspace(-(ksize[1]//2), ksize[1]//2, ksize[1])
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
    kernel = kernel / kernel.sum()  # Normalize

    # Apply convolution
    if len(image.shape) == 3:  # Color image
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = ndimage.convolve(image[:, :, i], kernel)
        return result
    else:  # Grayscale image
        return ndimage.convolve(image, kernel)


def get_paths_config_dict() -> dict:
    return get_config_path_data()


def save_path_config_dict(**kwargs) -> None:
    return save_config_path_data(**kwargs)


def camera_settings(width: int = Optional[None], height: Optional[int] = None) -> CameraSettings:
    """ Настройки камеры """
    if width and height:
        return CameraSettings(width=width, height=height)
    return CameraSettings()


def window_settings(width: int = Optional[None], height: Optional[int] = None) -> WindowSettings:
    """ Настройки окна """
    if width and height:
        return WindowSettings(width=width, height=height)
    return WindowSettings()


def raster_settings(angle: int, distance: int, thickness: int, offset: Optional[int] = 0) -> RasterSettings:
    """ Настройки растра """
    while angle < 0:
        angle += 360
    angle %= 360
    return RasterSettings(angle=angle, distance=distance, thickness=thickness, offset=offset)


def raster_settings_double(settings: RasterSettings, add_angle: int = 0, add_offset: int = 0) -> RasterSettings:
    """ Настройки дубля растра, используемого при наложении """
    differenced = dataclasses.replace(settings)
    differenced.angle += add_angle
    while differenced.angle < 0:
        differenced.angle += 360
    differenced.angle %= 360
    differenced.offset += add_offset
    return differenced


def load_raster_settings(path: str) -> RasterSettings:
    """ Загрузка настроек растра из указанной директории """
    return RasterSettings.load(path)


def load_image_by_tag(path: str, tag: str) -> Optional[ImageData]:
    if tag.lower().strip() == "raw":
        return load_camera_image(path)
    if tag.lower().strip() == "raster":
        return load_raster_image(path)
    if tag.lower().strip() == "process":
        return load_processed_camera_image(path)
    raise AttributeError(f"[!] Неизвестный тег {tag}")


def load_raster_image(path: str) -> Optional[ImageData]:
    """ Загрузка изображения растра из указанной директории """
    try:
        img = cv.imread(path, cv.COLOR_BGR2GRAY)
    except FileNotFoundError or FileExistsError as e:
        print(f"[!] Ошибка загрузки файла -> {e}")
        return None
    return ImageData(img, SourceType.RASTER)


def load_camera_image(path: str) -> Optional[ImageData]:
    """ Загрузка изображения муара из указанной директории """
    try:
        # img = cv.imread(path)
        img_bgr = cv.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Файл не найден или не может быть прочитан: {path}")

        # [ИСПРАВЛЕНИЕ] Конвертируем BGR (с диска) в RGB (для UI)
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    except Exception as e:
        print(f"[!] Ошибка загрузки файла -> {e}")
        return None

    # Возвращаем RGB
    return ImageData(img_rgb, SourceType.RAW)


def load_processed_camera_image(path: str):
    """ Загрузка изображения муара из указанной директории """
    try:
        img = cv.imread(path, cv.COLOR_BGR2GRAY)
    except FileNotFoundError or FileExistsError as e:
        print(f"[!] Ошибка загрузки файла -> {e}")
        return None
    return ImageData(img, SourceType.PROCESSED)


def save_raster_image(image: np.ndarray, parent_widget=None) -> Dict[str, str]:
    try:
        return _file_service.save_image(image, "raster", parent_widget)
    except Exception as e:
        print(f"[!] Ошибка сохранения растра: {e}")
        return {}

def save_camera_image(image: np.ndarray, parent_widget=None) -> Dict[str, str]:
    try:
        # [ИСПРАВЛЕНИЕ] Конвертируем RGB (от камеры) в BGR (для cv.imwrite)
        image_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return _file_service.save_image(image_bgr, "camera", parent_widget)
    except Exception as e:
        print(f"[!] Ошибка сохранения изображения камеры: {e}")
        return {}

def save_raster_with_settings(raster: np.ndarray, settings_data: str, parent_widget=None) -> Dict[str, str]:
    try:
        return _file_service.save_raster_with_settings(raster, settings_data, parent_widget)
    except Exception as e:
        print(f"[!] Ошибка сохранения растра с настройками: {e}")
        return {}

def create_raster(window_settings_: WindowSettings, raster_settings_: RasterSettings) -> ImageData:
    """ Выдача растра с указанными настройками """
    factory = RasterFactory(window_settings_, raster_settings_)
    raster = factory.process()
    result = raster.copy()

    return ImageData(result, SourceType.RASTER)


def camera(camera_settings_: Optional[CameraSettings] = None) -> Optional[AsyncCamera]:
    """ Включение, выключение камеры """
    global _camera
    if not _camera:
        camera_settings_ = camera_settings_ or CameraSettings()
        _camera = AsyncCamera(camera_settings=camera_settings_)
        _camera.start()
        return _camera
    _camera.stop()
    _camera = None
    return None


def is_camera_on() -> bool:
    global _camera
    return bool(camera)


def get_picture() -> ImageData:
    """ Получение изображения со включенной камеры """
    global _camera
    if not _camera:
        raise AttributeError("[!] Нет объекта камеры")
    if not _camera.processing:
        raise ValueError("[!] Камера выключена")
    try:
        _, img = _camera.read()
    except Exception as e:
        raise Exception(f"[!] Ошибка чтения фотографии -> {e}")
    return ImageData(img, SourceType.RAW)


def processor_pipeline(image_data: ImageData, threshold_value: int, win_settings: WindowSettings) -> ImageData:
    """ Основной шаблон обработки фото растра """
    if image_data.source is not SourceType.RAW:
        raise AttributeError(
            f"[!] Передан неправильный тип изображения {image_data.source}")
    image = image_data.image
    # if image.ndim == 2:
    #     return image
    # image = ImageProcessor.threshold(image, on_value=threshold_value)
    # image = ImageProcessor.crop(image, top_crop=top_offset)
    # image = ImageProcessor.resize(
    #     image, width=win_settings.width, height=win_settings.height)
    # return ImageData(image, SourceType.PROCESSED)

    # === НАЧАЛО ИЗМЕНЕНИЯ ===
    # Раньше было: image = ImageProcessor.threshold(image) [cite: 274]
    # Теперь передаем 'top_offset' (который является нашим threshold_value)
    image = ImageProcessor.threshold(image, on_value=threshold_value)
    # === КОНЕЦ ИЗМЕНЕНИЯ ===

    # image = ImageProcessor.crop(image, top_crop=top_offset)
    image = ImageProcessor.resize(
        image, width=win_settings.width, height=win_settings.height)
    return ImageData(image, SourceType.PROCESSED)

def processor_resize(image_data: ImageData, win_settings: WindowSettings) -> ImageData:
    image = ImageProcessor.resize(image_data.image, win_settings.width, win_settings.height,
                                  interpolation=cv.INTER_NEAREST)
    return ImageData(image, SourceType.RAW)


def repeate_image(image: ImageData, axis: int, amount: int = 2) -> ImageData:
    """ Повторить изображение несколько раз вниз или вправо """
    if image.source is not SourceType.PROCESSED:
        raise AttributeError("[!] Выдано необработанное изображение")
    processed_image = ImageProcessor.repeate(image.image, axis, amount)
    return ImageData(processed_image, SourceType.PROCESSED)


def concat_images(images: List[ImageData], axis: int) -> ImageData:
    """ Склеить данные изображения в указанном направлении """
    images_amount = len(images)
    if images_amount == 0:
        raise AttributeError("[!] Передан пустой список изображений")
    elif images_amount == 1:
        return images[0]
    concated = images[0].image.copy()
    for index in range(1, images_amount):
        concated = ImageProcessor.concat(concated, images[index].image, axis)
    return ImageData(concated, SourceType.PROCESSED)


def masking(base: ImageData, mask: ImageData) -> ImageData:
    """ Покрытие изображения черно-белой маской """
    if (
            base.source not in [SourceType.RASTER, SourceType.PROCESSED]
            or mask.source not in [SourceType.RASTER, SourceType.PROCESSED]
    ):
        raise AttributeError(
            f"[!] Переданы неправильные источники изображения {base.source} {mask.source}")
    masked = ImageProcessor.masking(base.image, mask.image)
    return ImageData(masked, SourceType.PROCESSED)


def smooth(image: ImageData):
    smoothed = gaussian_blur_numpy(image.image, ksize=(9, 9))
    return ImageData(smoothed, SourceType.PROCESSED)


def poster_points(image_data: ImageData, poster_data: Optional[ImageData],
                  edges: bool = False, radius: int = 2, color: Color = Color.Red) -> ImageData:
    """ Выделить группы и отметить их центры """
    image = image_data.image
    poster_shape = (image.shape[0], image.shape[1], 3)
    poster = poster_data.image if poster_data and poster_data.image is not None else None
    if poster is None:
        poster = np.zeros(poster_shape, dtype='uint8')
    hull_group = ImageProcessor.hull_points(image)
    if edges:
        points = hull_group.hulls + hull_group.centers
    else:
        points = hull_group.centers
    points: List[Point] = points
    for point in points:
        cv.circle(poster, point.to_tuple(), radius, color, -1)
    return ImageData(poster, SourceType.NONE)


def imshow(img: np.ndarray, winname=None) -> None:
    """ Синхронный вывод изображения с названием окна winname """
    if img is None:
        return
    winname = winname or "test"
    cv.imshow(winname, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def project_image(image: np.ndarray, screen_number: int = 1) -> bool:
    """Проецировать изображение на проектор"""
    global _projector_controller
    try:
        return _projector_controller.project_image(image, screen_number)
    except Exception as e:
        print(f"[!] Ошибка проецирования: {e}")
        return False

def stop_projection() -> None:
    """Остановить проецирование"""
    global _projector_controller
    _projector_controller.stop_projection()

def is_projecting() -> bool:
    """Проверяет, активно ли проецирование"""
    global _projector_controller
    return _projector_controller.is_projecting()


#new camera metods
#
def start_camera_preview(src: int = 0, camera_settings_: Optional[CameraSettings] = None) -> Optional[AsyncCamera]: # СТАЛО
    """Запуск предпросмотра камеры"""
    global _camera
    if _camera and _camera.processing:
        return _camera

    # Останавливаем предыдущую камеру если была
    if _camera:
        _camera.release()

    camera_settings_ = camera_settings_ or CameraSettings()
    _camera = AsyncCamera(src=src, camera_settings=camera_settings_) # Используем src
    _camera.start()
    return _camera

def is_camera_processing() -> bool:
    """Проверяет, что камера существует и находится в режиме обработки"""
    global _camera
    return bool(_camera) and _camera.processing

def stop_camera_preview():
    """Остановка предпросмотра камеры"""
    global _camera
    if _camera:
        _camera.stop()
    return None


def capture_from_camera() -> Optional[ImageData]:
    """Захват кадра с камеры и остановка предпросмотра"""
    global _camera
    if not _camera or not _camera.processing:
        raise AttributeError("[!] Камера не запущена")

    try:
        captured_frame = _camera.capture_frame()
        if captured_frame is not None:
            # Останавливаем предпросмотр после захвата
            _camera.stop()
            return ImageData(captured_frame, SourceType.RAW)
        else:
            raise Exception("Не удалось захватить кадр")
    except Exception as e:
        _camera.stop()
        raise Exception(f"[!] Ошибка захвата фотографии -> {e}")


def get_camera_frame() -> Optional[np.ndarray]:
    """Получение текущего кадра для предпросмотра"""
    global _camera
    # Проверяем в `is_camera_processing`, а не здесь
    if not _camera:
        return None
    try:
        grabbed, frame = _camera.read()
        # frame будет либо кадром, либо None (если latest_frame еще None)
        return frame
    except:
        return None