"""
Геометрические примитивы: лучи и треугольники.
"""

import numpy as np
from numba import njit
from math_utils import dot, cross, normalize


@njit(cache=True)
def ray_triangle_intersect(ray_origin: np.ndarray, ray_dir: np.ndarray,
                           v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Пересечение луча с треугольником (алгоритм Моллера-Трумбора).
    Возвращает t (параметр луча) или -1.0 если пересечения нет.
    """
    EPSILON = 1e-7

    edge1 = v1 - v0
    edge2 = v2 - v0

    h = cross(ray_dir, edge2)
    a = dot(edge1, h)

    # Луч параллелен треугольнику
    if abs(a) < EPSILON:
        return -1.0

    f = 1.0 / a
    s = ray_origin - v0
    u = f * dot(s, h)

    if u < 0.0 or u > 1.0:
        return -1.0

    q = cross(s, edge1)
    v = f * dot(ray_dir, q)

    if v < 0.0 or u + v > 1.0:
        return -1.0

    t = f * dot(edge2, q)

    if t < EPSILON:
        return -1.0

    return t


@njit(cache=True)
def compute_triangle_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Вычисляет нормаль треугольника."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    return normalize(cross(edge1, edge2))


@njit(cache=True)
def triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Вычисляет площадь треугольника."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_prod = cross(edge1, edge2)
    return 0.5 * np.sqrt(cross_prod[0]**2 + cross_prod[1]**2 + cross_prod[2]**2)


@njit(cache=True)
def sample_triangle_point(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Случайная точка на треугольнике (равномерное распределение)."""
    r1 = np.random.random()
    r2 = np.random.random()

    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = r2 * sqrt_r1

    return v0 + (v1 - v0) * u + (v2 - v0) * v

