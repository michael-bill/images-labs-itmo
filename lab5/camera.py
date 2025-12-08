"""
Камера для рендеринга.
"""

import numpy as np
from numba import njit
from math_utils import normalize, cross


class Camera:
    """Точечная (pinhole) камера."""

    def __init__(self, position, look_at, up, fov: float, width: int, height: int):
        self.position = np.array(position, dtype=np.float64)
        self.width = width
        self.height = height

        # Базис камеры
        forward = normalize(np.array(look_at, dtype=np.float64) - self.position)
        right = normalize(cross(forward, np.array(up, dtype=np.float64)))
        up_vec = cross(right, forward)

        self.forward = forward
        self.right = right
        self.up = up_vec

        # Размер экрана в мировых координатах
        aspect = width / height
        self.half_height = np.tan(fov * np.pi / 360.0)
        self.half_width = self.half_height * aspect


@njit(cache=True)
def get_ray(x: int, y: int, width: int, height: int,
            position: np.ndarray, forward: np.ndarray,
            right: np.ndarray, up: np.ndarray,
            half_width: float, half_height: float,
            jitter: bool = True):
    """
    Генерирует направление луча для пикселя (x, y).
    Возвращает (origin, direction).
    """
    # Смещение внутри пикселя для антиалиасинга
    if jitter:
        px = x + np.random.random()
        py = y + np.random.random()
    else:
        px = x + 0.5
        py = y + 0.5

    # Нормализованные координаты [-1, 1]
    u = (2.0 * px / width - 1.0) * half_width
    v = (1.0 - 2.0 * py / height) * half_height

    direction = forward + right * u + up * v
    direction = direction / np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)

    return position.copy(), direction
