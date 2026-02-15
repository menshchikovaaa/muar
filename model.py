import math
from enum import IntEnum, Enum
from typing import List, Optional
from functools import total_ordering


def _center_by_coords_2d(*points):
    x_vertexes = [point.cox for point in points]
    y_vertexes = [point.coy for point in points]
    x_center = int(sum(x_vertexes) / len(x_vertexes))
    y_center = int(sum(y_vertexes) / len(y_vertexes))
    return x_center, y_center


class Direction(IntEnum):
    Right = 0
    Left = 1


class Color:
    Black = (0, 0, 0)
    White = (255, 255, 255)
    Red = (0, 0, 255)
    Green = (0, 255, 0)
    Blue = (255, 0, 0)
    Yellow = (0, 255, 255)


@total_ordering
class Point:
    def __init__(self, cox: int, coy: int):
        self.cox = int(cox)
        self.coy = int(coy)

    def _radius(self):
        return int(math.sqrt(self.cox ** 2 + self.coy ** 2))

    def _angle(self):
        return math.atan(self.coy / self.cox)

    @staticmethod
    def distance(p1, p2):
        diffx = p1.cox - p2.cox
        diffy = p1.coy - p2.coy
        return int(math.sqrt(diffx * diffx + diffy * diffy))

    def to_tuple(self):
        return self.cox, self.coy

    def __str__(self):
        return f"Point({self.cox}, {self.coy})"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        other: Point = other
        return (self._angle(), self._radius()) < (other._angle(), other._radius())

    def __gt__(self, other):
        other: Point = other
        return (self._angle(), self._radius()) > (other._angle(), other._radius())


class Section:
    def __init__(self, pta: Point, ptb: Point):
        self.pta = pta
        self.ptb = ptb

    def shift(self, dx, dy):
        self.pta.cox -= dx
        self.pta.coy -= dy
        self.ptb.cox -= dx
        self.ptb.coy -= dy

    @property
    def length(self):
        diffx = self.ptb.cox - self.pta.cox
        diffy = self.ptb.coy - self.pta.coy
        return math.sqrt(diffx * diffx + diffy * diffy)

    @staticmethod
    def perp(section, length):
        diffx = section.ptb.cox - section.pta.cox
        diffy = section.ptb.coy - section.pta.coy
        if diffx == 0:
            new_pta = Point(section.ptb.cox + length, section.ptb.coy)
            new_ptb = Point(section.ptb.cox - length, section.ptb.coy)
            return Section(new_pta, new_ptb)
        if diffy == 0:
            new_pta = Point(section.ptb.cox, section.ptb.coy + length)
            new_ptb = Point(section.ptb.cox, section.ptb.coy - length)
            return Section(new_pta, new_ptb)

        diffx /= section.length
        diffy /= section.length
        diffx, diffy = -diffy, diffx
        new_pta = Point(section.ptb.cox + diffx * length,
                        section.ptb.coy + diffy * length)
        new_ptb = Point(section.ptb.cox - diffx * length,
                        section.ptb.coy - diffy * length)
        return Section(new_pta, new_ptb)

    def __str__(self):
        return f"{self.pta} -> {self.ptb}"

    def __repr__(self):
        return self.__str__()


class DeformType(Enum):
    inDeform = "inDeform"
    outDeform = "outDeform"
    noneDeform = "noneDeform"


@total_ordering
class Group:
    ido_generator = (i for i in range(int(1e6)))

    def __init__(self, hull_points: list):
        self.ido = next(self.ido_generator)
        self.points = [Point(*vertex) for vertex in hull_points]

    def center(self) -> Point:
        x_center, y_center = _center_by_coords_2d(*self.points)
        x_center += 1
        y_center += 1
        return Point(x_center, y_center)

    def __str__(self):
        return ", ".join([f"{prop}: {value}" for prop, value in self.__dict__.items()])

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self.ido < other.ido

    def __gt__(self, other):
        return self.ido > other.ido


class GroupPack:
    ido_generator = (i for i in range(int(1e3)))

    def __init__(self, groups: List[Group]):
        self.ido = next(self.ido_generator)
        self._groups = groups

    def pick_group(self, ido: int) -> Optional[Group]:
        try:
            return sorted(self._groups, key=lambda x: x.ido)[ido]
        except IndexError:
            return None

    @property
    def groups(self) -> List[Group]:
        return sorted(self._groups)

    @property
    def centers(self) -> List[Point]:
        return [group.center() for group in self._groups]

    @property
    def hulls(self) -> List[Point]:
        hull_points = []
        for group in self._groups:
            hull_points.extend(group.points)
        return hull_points

    def __str__(self):
        return "\n".join(self._groups)

    def __repr__(self):
        return self.__str__()
