import cv2 as cv
import numpy as np
import configparser
from typing import Dict
from pathlib import Path
from datetime import datetime

SAVE_DATE_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"


def get_config_path_data() -> dict:
    config_path_key = "Paths"
    path = "settings.ini"
    # path = Path(dirname(__file__)).parent / Path(path_to_ini)
    config = configparser.ConfigParser()
    config.read(path)
    # print(config.sections())
    return dict(config[config_path_key])


def save_config_path_data(**kwargs) -> None:
    config_path_key = "Paths"
    path = "settings.ini"
    # path = Path(dirname(__file__)).parent / Path(path_to_ini)
    config = configparser.ConfigParser()
    config.read(path)

    if kwargs.get("root"):
        config.set(config_path_key, "root", kwargs["root"])

    if kwargs.get("directory"):
        config.set(config_path_key, "directory", kwargs["directory"])

    if kwargs.get("folder_raster"):
        config.set(config_path_key, "folder_raster", kwargs["folder_raster"])

    if kwargs.get("folder_settings"):
        config.set(config_path_key, "folder_settings",
                   kwargs["folder_settings"])

    if kwargs.get("folder_camera"):
        config.set(config_path_key, "folder_camera", kwargs["folder_camera"])

    if kwargs.get("raster_filename"):
        config.set(config_path_key, "raster_filename",
                   kwargs["raster_filename"])

    if kwargs.get("raster_extension"):
        config.set(config_path_key, "raster_extension",
                   kwargs["raster_extension"])

    if kwargs.get("settings_filename"):
        config.set(config_path_key, "settings_filename",
                   kwargs["settings_filename"])

    if kwargs.get("settings_extension"):
        config.set(config_path_key, "settings_extension",
                   kwargs["settings_extension"])

    if kwargs.get("camera_filename"):
        config.set(config_path_key, "camera_filename",
                   kwargs["camera_filename"])

    if kwargs.get("raster_extension"):
        config.set(config_path_key, "raster_extension",
                   kwargs["raster_extension"])

    with open(path, 'wb') as configfile:
        config.write(configfile)


def _path_to_save_files(raster: bool, settings: bool, camera: bool) -> dict:
    params_path = get_config_path_data()
    name = datetime.now().strftime(SAVE_DATE_FORMAT)
    paths = {}
    if raster:
        raster_path = Path(
            params_path["root"]) / params_path["directory"] / params_path["folder_raster"]
        to_raster = f'{raster_path}\\{params_path["raster_filename"]}-{name}.{params_path["raster_extension"]}'
        paths["to_raster"] = to_raster
        paths["to_raster_filename"] = f'{params_path["raster_filename"]}-{name}.{params_path["raster_extension"]}'
    if settings:
        settings_path = Path(
            params_path["root"]) / params_path["directory"] / params_path["folder_settings"]
        to_settings = f'{settings_path}\\{params_path["settings_filename"]}-{name}.{params_path["settings_extension"]}'
        paths["to_settings"] = to_settings
        paths["to_settings_filename"] = f'{params_path["settings_filename"]}-{name}.{params_path["settings_extension"]}'
    if camera:
        camera_path = Path(
            params_path["root"]) / params_path["directory"] / params_path["folder_camera"]
        to_camera = f'{camera_path}\\{params_path["camera_filename"]}-{name}.{params_path["raster_extension"]}'
        paths["to_camera"] = to_camera
        paths["to_camera_filename"] = f'{params_path["camera_filename"]}-{name}.{params_path["raster_extension"]}'

    return paths


def save_image(image: np.ndarray, path: str):
    try:
        cv.imwrite(path, image)
    except FileNotFoundError or FileExistsError:
        print("[!] Сохранение файлов не удалось")
    except Exception as e:
        print("[!] Ошибка ", e)
    return


def save_raster(raster: np.ndarray) -> Dict[str, str]:
    paths = _path_to_save_files(True, False, False)
    raster_path = paths["to_raster"]
    save_image(raster, raster_path)
    return paths


def save_data(raster: np.ndarray, settings: str) -> Dict[str, str]:
    paths = _path_to_save_files(True, True, False)
    raster_path, settings_path = paths["to_raster"], paths["to_settings"]

    try:
        with open(settings_path, mode='w') as file:
            file.write(settings)
    except FileNotFoundError or FileExistsError:
        print("[!] Сохранение настроек не удалось")
        return
    save_image(raster, raster_path)
    return paths


def save_camera(camera: np.ndarray) -> Dict[str, str]:
    paths = _path_to_save_files(False, False, True)
    camera_path = paths["to_camera"]
    save_image(camera, camera_path)
    return paths
