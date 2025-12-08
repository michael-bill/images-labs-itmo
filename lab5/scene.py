"""
Сцена: хранение геометрии, материалов и источников света.
"""

import numpy as np
from numba import njit
from numba.typed import List as NumbaList
from geometry import ray_triangle_intersect, compute_triangle_normal, triangle_area, sample_triangle_point
from math_utils import dot, length, normalize


class Scene:
    """
    Контейнер сцены с треугольниками и материалами.
    Данные хранятся в numpy массивах для быстрого доступа из numba.
    """
    
    def __init__(self):
        # Списки для накопления данных при построении сцены
        self._vertices = []      # вершины треугольников (v0, v1, v2)
        self._material_ids = []  # индекс материала для каждого треугольника
        self._materials = []     # список материалов
        self._light_indices = [] # индексы светящихся треугольников
        
        # Финальные numpy массивы (создаются при компиляции)
        self.vertices = None     # shape: (n_triangles, 3, 3)
        self.material_ids = None # shape: (n_triangles,)
        self.normals = None      # shape: (n_triangles, 3)
        self.areas = None        # shape: (n_triangles,)
        
        # Материалы: (diffuse_color, specular_color, emission)
        self.diffuse = None      # shape: (n_materials, 3)
        self.specular = None     # shape: (n_materials, 3)
        self.emission = None     # shape: (n_materials, 3)
        
        self.light_indices = None
        self.light_areas = None
        self.total_light_area = 0.0
    
    def add_material(self, diffuse=(0.5, 0.5, 0.5), specular=(0.0, 0.0, 0.0),
                     emission=(0.0, 0.0, 0.0)) -> int:
        """Добавляет материал и возвращает его индекс."""
        self._materials.append((
            np.array(diffuse, dtype=np.float64),
            np.array(specular, dtype=np.float64),
            np.array(emission, dtype=np.float64)
        ))
        return len(self._materials) - 1
    
    def add_triangle(self, v0, v1, v2, material_id: int):
        """Добавляет треугольник."""
        idx = len(self._vertices)
        self._vertices.append((
            np.array(v0, dtype=np.float64),
            np.array(v1, dtype=np.float64),
            np.array(v2, dtype=np.float64)
        ))
        self._material_ids.append(material_id)
        
        # Проверяем, светится ли материал
        if np.max(self._materials[material_id][2]) > 0:
            self._light_indices.append(idx)
    
    def add_quad(self, v0, v1, v2, v3, material_id: int):
        """Добавляет четырёхугольник как два треугольника."""
        self.add_triangle(v0, v1, v2, material_id)
        self.add_triangle(v0, v2, v3, material_id)
    
    def compile(self):
        """
        Компилирует сцену в numpy массивы для быстрого доступа.
        Вызывать после добавления всей геометрии.
        """
        n_tris = len(self._vertices)
        n_mats = len(self._materials)
        
        # Вершины
        self.vertices = np.zeros((n_tris, 3, 3), dtype=np.float64)
        for i, (v0, v1, v2) in enumerate(self._vertices):
            self.vertices[i, 0] = v0
            self.vertices[i, 1] = v1
            self.vertices[i, 2] = v2
        
        # Индексы материалов
        self.material_ids = np.array(self._material_ids, dtype=np.int32)
        
        # Предвычисляем нормали и площади
        self.normals = np.zeros((n_tris, 3), dtype=np.float64)
        self.areas = np.zeros(n_tris, dtype=np.float64)
        for i in range(n_tris):
            self.normals[i] = compute_triangle_normal(
                self.vertices[i, 0], self.vertices[i, 1], self.vertices[i, 2]
            )
            self.areas[i] = triangle_area(
                self.vertices[i, 0], self.vertices[i, 1], self.vertices[i, 2]
            )
        
        # Материалы
        self.diffuse = np.zeros((n_mats, 3), dtype=np.float64)
        self.specular = np.zeros((n_mats, 3), dtype=np.float64)
        self.emission = np.zeros((n_mats, 3), dtype=np.float64)
        for i, (d, s, e) in enumerate(self._materials):
            self.diffuse[i] = d
            self.specular[i] = s
            self.emission[i] = e
        
        # Источники света
        self.light_indices = np.array(self._light_indices, dtype=np.int32)
        if len(self._light_indices) > 0:
            self.light_areas = self.areas[self.light_indices]
            self.total_light_area = np.sum(self.light_areas)
        else:
            self.light_areas = np.array([], dtype=np.float64)
            self.total_light_area = 0.0
        
        print(f"Сцена скомпилирована: {n_tris} треугольников, {n_mats} материалов, "
              f"{len(self._light_indices)} источников света")


@njit(cache=True)
def intersect_scene(ray_origin: np.ndarray, ray_dir: np.ndarray,
                    vertices: np.ndarray, normals: np.ndarray,
                    material_ids: np.ndarray):
    """
    Поиск ближайшего пересечения луча со сценой.
    Возвращает: (t, hit_point, normal, material_id) или None-значения если нет пересечения.
    """
    closest_t = np.inf
    hit_idx = -1
    
    n_triangles = vertices.shape[0]
    for i in range(n_triangles):
        t = ray_triangle_intersect(
            ray_origin, ray_dir,
            vertices[i, 0], vertices[i, 1], vertices[i, 2]
        )
        if t > 0 and t < closest_t:
            closest_t = t
            hit_idx = i
    
    if hit_idx < 0:
        return -1.0, np.zeros(3), np.zeros(3), -1
    
    hit_point = ray_origin + ray_dir * closest_t
    normal = normals[hit_idx].copy()
    
    # Ориентируем нормаль к источнику луча
    if dot(normal, ray_dir) > 0:
        normal = -normal
    
    return closest_t, hit_point, normal, material_ids[hit_idx]


@njit(cache=True)
def sample_light_point(vertices: np.ndarray, normals: np.ndarray,
                       emission: np.ndarray, material_ids: np.ndarray,
                       light_indices: np.ndarray, light_areas: np.ndarray,
                       total_light_area: float):
    """
    Выборка случайной точки на источнике света.
    Возвращает: (point, normal, emission, pdf).
    """
    n_lights = len(light_indices)
    if n_lights == 0:
        return np.zeros(3), np.zeros(3), np.zeros(3), 0.0
    
    # Выбираем источник пропорционально площади
    r = np.random.random() * total_light_area
    cumsum = 0.0
    light_idx = 0
    for i in range(n_lights):
        cumsum += light_areas[i]
        if r <= cumsum:
            light_idx = i
            break
    
    tri_idx = light_indices[light_idx]
    point = sample_triangle_point(
        vertices[tri_idx, 0], vertices[tri_idx, 1], vertices[tri_idx, 2]
    )
    normal = normals[tri_idx]
    mat_id = material_ids[tri_idx]
    emiss = emission[mat_id]
    
    # PDF = 1 / площадь выбранного источника
    pdf = 1.0 / light_areas[light_idx]
    
    return point, normal, emiss, pdf

