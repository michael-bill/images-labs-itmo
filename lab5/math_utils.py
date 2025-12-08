"""
Математические утилиты для работы с векторами.
Использует numba для JIT-компиляции критических функций.
"""

import numpy as np
from numba import njit

# Тип для 3D вектора: numpy array shape (3,)
Vec3 = np.ndarray


@njit(cache=True)
def normalize(v: Vec3) -> Vec3:
    """Нормализация вектора."""
    length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length < 1e-10:
        return np.zeros(3)
    return v / length


@njit(cache=True)
def dot(a: Vec3, b: Vec3) -> float:
    """Скалярное произведение."""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@njit(cache=True)
def cross(a: Vec3, b: Vec3) -> Vec3:
    """Векторное произведение."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])


@njit(cache=True)
def length(v: Vec3) -> float:
    """Длина вектора."""
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


@njit(cache=True)
def reflect(direction: Vec3, normal: Vec3) -> Vec3:
    """Зеркальное отражение вектора относительно нормали."""
    return direction - normal * (2 * dot(direction, normal))


@njit(cache=True)
def random_cosine_hemisphere(normal: Vec3) -> Vec3:
    """
    Случайное направление в полусфере с косинус-взвешенным распределением.
    Используется для выборки по Ламберту.
    """
    r1 = np.random.random()
    r2 = np.random.random()

    phi = 2 * np.pi * r1
    cos_theta = np.sqrt(r2)
    sin_theta = np.sqrt(1 - r2)

    # Локальные координаты
    x = np.cos(phi) * sin_theta
    y = np.sin(phi) * sin_theta
    z = cos_theta

    # Строим ортонормированный базис от нормали
    if abs(normal[0]) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    else:
        up = np.array([1.0, 0.0, 0.0])

    tangent = normalize(cross(up, normal))
    bitangent = cross(normal, tangent)

    # Преобразуем в мировые координаты
    return normalize(tangent * x + bitangent * y + normal * z)

