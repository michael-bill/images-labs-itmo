"""
Path Tracer - ядро рендеринга.
Все критические функции оптимизированы с помощью numba.
"""

import numpy as np
from numba import njit, prange
from math_utils import dot, normalize, reflect, random_cosine_hemisphere, length
from scene import intersect_scene, sample_light_point
from camera import get_ray
    

@njit(cache=True)
def trace_ray(ray_origin: np.ndarray, ray_dir: np.ndarray,
              vertices: np.ndarray, normals: np.ndarray, areas: np.ndarray,
              material_ids: np.ndarray,
              diffuse: np.ndarray, specular: np.ndarray, emission: np.ndarray,
              light_indices: np.ndarray, light_areas: np.ndarray,
              total_light_area: float,
              max_depth: int, rr_depth: int) -> np.ndarray:
    """
    Трассировка одного пути методом path tracing.
    Использует русскую рулетку и выборку по значимости.
    """
    color = np.zeros(3)
    throughput = np.ones(3)  # накопленный коэффициент пропускания
    
    current_origin = ray_origin.copy()
    current_dir = ray_dir.copy()
    
    for depth in range(max_depth):
        # Поиск пересечения
        t, hit_point, normal, mat_id = intersect_scene(
            current_origin, current_dir, vertices, normals, material_ids
        )
        
        if mat_id < 0:
            break  # луч ушёл в пустоту
        
        mat_emission = emission[mat_id]
        mat_diffuse = diffuse[mat_id]
        mat_specular = specular[mat_id]
        
        # Если попали в источник света
        if mat_emission[0] > 0 or mat_emission[1] > 0 or mat_emission[2] > 0:
            color = color + throughput * mat_emission
            break
        
        # Русская рулетка после определённой глубины
        total_refl = mat_diffuse + mat_specular
        p_continue = min(max(total_refl[0], max(total_refl[1], total_refl[2])), 0.95)
        
        if depth >= rr_depth:
            if np.random.random() > p_continue:
                break
            throughput = throughput / p_continue
        
        # Выбор типа отражения (выборка по значимости)
        diff_weight = max(mat_diffuse[0], max(mat_diffuse[1], mat_diffuse[2]))
        spec_weight = max(mat_specular[0], max(mat_specular[1], mat_specular[2]))
        total_weight = diff_weight + spec_weight
        
        if total_weight < 1e-6:
            break
        
        p_diffuse = diff_weight / total_weight
        
        # Смещение от поверхности для избежания self-intersection
        EPSILON = 0.0001
        
        if np.random.random() < p_diffuse:
            # Диффузное отражение (Ламберт)
            offset = normal * EPSILON
            
            # Прямое освещение (Next Event Estimation)
            if len(light_indices) > 0:
                light_point, light_normal, light_emission, pdf = sample_light_point(
                    vertices, normals, emission, material_ids,
                    light_indices, light_areas, total_light_area
                )
                
                to_light = light_point - hit_point
                dist = length(to_light)
                light_dir = to_light / dist
                
                cos_theta = dot(normal, light_dir)
                cos_theta_light = dot(-light_dir, light_normal)
                
                if cos_theta > 0 and cos_theta_light > 0:
                    # Теневой луч
                    shadow_t, _, _, shadow_mat = intersect_scene(
                        hit_point + offset, light_dir, vertices, normals, material_ids
                    )
                    
                    if shadow_t < 0 or shadow_t > dist - 0.001:
                        # Свет виден
                        geometry = cos_theta * cos_theta_light / (dist * dist)
                        brdf = mat_diffuse / np.pi
                        direct = light_emission * brdf * geometry * total_light_area
                        color = color + throughput * direct
            
            # Обновляем направление для непрямого освещения
            new_dir = random_cosine_hemisphere(normal)
            throughput = throughput * mat_diffuse
            
            current_origin = hit_point + offset
            current_dir = new_dir
        else:
            # Зеркальное отражение
            new_dir = reflect(current_dir, normal)
            
            # Для зеркала смещаемся в направлении отражённого луча
            offset = new_dir * EPSILON
            throughput = throughput * mat_specular
            
            current_origin = hit_point + offset
            current_dir = new_dir
    
    return color


@njit(parallel=True, cache=True)
def render_image(width: int, height: int, samples_per_pixel: int,
                 position: np.ndarray, forward: np.ndarray,
                 right: np.ndarray, up: np.ndarray,
                 half_width: float, half_height: float,
                 vertices: np.ndarray, normals: np.ndarray, areas: np.ndarray,
                 material_ids: np.ndarray,
                 diffuse: np.ndarray, specular: np.ndarray, emission: np.ndarray,
                 light_indices: np.ndarray, light_areas: np.ndarray,
                 total_light_area: float,
                 max_depth: int, rr_depth: int) -> np.ndarray:
    """
    Параллельный рендеринг изображения.
    Использует numba parallel для многопоточности.
    """
    image = np.zeros((height, width, 3))
    
    for y in prange(height):
        for x in range(width):
            pixel_color = np.zeros(3)
            
            for _ in range(samples_per_pixel):
                origin, direction = get_ray(
                    x, y, width, height,
                    position, forward, right, up,
                    half_width, half_height, True
                )
                
                sample_color = trace_ray(
                    origin, direction,
                    vertices, normals, areas, material_ids,
                    diffuse, specular, emission,
                    light_indices, light_areas, total_light_area,
                    max_depth, rr_depth
                )
                
                pixel_color = pixel_color + sample_color
            
            image[y, x] = pixel_color / samples_per_pixel
    
    return image

