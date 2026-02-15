from typing import Tuple
from dataclasses import dataclass
from model import Color, Point


@dataclass
class CameraSettings:
    width: int = 1280
    height: int = 720


@dataclass
class WindowSettings:
    width: int = 1024
    height: int = 768

    @property
    def center(self) -> Point:
        return Point(self.width // 2, self.height // 2)


@dataclass
class RasterSettings:
    angle: int
    distance: int
    thickness: int
    offset: int = 0
    color: Tuple[int, int, int] = Color.Black

    @classmethod
    def load(cls, path: str):
        try:
            with open(path, "r") as file:
                settings = file.readline()
        except FileNotFoundError or FileExistsError as e:
            print("[!] Ошибка чтения файла -> ", e)
            return None
        except ...:
            print("[!] Неизвестная ошибка")
            return None

        temp = cls(0, 0, 0)
        try:
            for part in settings.split(";"):
                field, value = part.split("=")
                if field == "color":
                    value = tuple(map(int, value.strip("() \n\t").split(',')))
                    temp.__setattr__(field, value)
                else:
                    temp.__setattr__(field, int(value))
        except ...:
            print("[!] Загрузка настроек не удалась")
            return None
        return temp

    def stringify(self):
        result = []
        for field, value in self.__dict__.items():
            if (value or value == 0) and field in ["angle", "distance", "thickness", "offset", "color"]:
                result.append(f"{field}={value}")
        return ";".join(result)

    def __repr__(self):
        return self.stringify().replace(";", "\n").title()
