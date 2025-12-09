"""
Path Tracer - Синтез изображений методом трассировки путей.

Лабораторная работа №5: Методы обработки изображений.
Реализация глобального освещения методом path tracing.

Запуск: python main.py
"""

import time
import numpy as np
from camera import Camera
from cornell_box import create_cornell_box
from renderer import render_image
from postprocess import tonemap_and_gamma, save_ppm, save_png


# ==================== КОНФИГУРАЦИЯ ====================
# Параметры можно изменить для получения разных результатов

CONFIG = {
    # --- Параметры рендеринга ---
    'width': 600,               # ширина изображения
    'height': 600,              # высота изображения
    'samples_per_pixel': 512,    # сэмплов на пиксель (больше = меньше шума)
    'max_depth': 6,             # максимальная глубина трассировки
    'russian_roulette_depth': 2,# глубина начала русской рулетки

    # --- Камера ---
    'camera_position': [0, 2.5, -8],  # позиция камеры
    'camera_look_at': [0, 2.5, 0],    # точка, куда смотрит камера
    'camera_fov': 45,                  # угол обзора (градусы)

    # --- Сцена (Cornell Box) ---
    'room_size': 5.0,                         # размер комнаты
    'left_wall_color': [0.75, 0.25, 0.25],    # красная стена
    'right_wall_color': [0.25, 0.25, 0.75],   # синяя стена
    'wall_color': [0.75, 0.75, 0.75],         # белые стены

    # --- Источник света ---
    'light_intensity': 15.0,           # яркость
    'light_color': [1.0, 0.95, 0.9],   # цвет (тёплый белый)
    'light_size': 1.8,                 # размер

    # --- Левый куб (диффузный) ---
    'box1_position': [-1.0, 0, 0.5],
    'box1_size': 1.3,
    'box1_rotation': -18,
    'box1_diffuse': [0.75, 0.75, 0.75],  # диффузный цвет
    'box1_specular': [0.0, 0.0, 0.0],    # без зеркальности

    # --- Правый куб (зеркальный) ---
    'box2_position': [1.0, 0, -0.5],
    'box2_size': 1.3,
    'box2_rotation': 15,
    'box2_diffuse': [0.05, 0.05, 0.05],  # почти чёрный
    'box2_specular': [0.9, 0.9, 0.9],    # высокая зеркальность

    # --- Постобработка ---
    'gamma': 2.2,      # гамма-коррекция
    'exposure': 1.5,   # экспозиция
}


def main():
    """Основная функция рендеринга."""

    print("=" * 60)
    print("Path Tracer - Трассировка путей")
    print("=" * 60)

    # 1. Создаём сцену Cornell Box
    print("\n[1/4] Создание сцены...")
    scene = create_cornell_box(CONFIG)

    # 2. Создаём камеру
    print("[2/4] Настройка камеры...")
    camera = Camera(
        position=CONFIG['camera_position'],
        look_at=CONFIG['camera_look_at'],
        up=[0, 1, 0],
        fov=CONFIG['camera_fov'],
        width=CONFIG['width'],
        height=CONFIG['height']
    )

    # 3. Рендеринг
    print(f"[3/4] Рендеринг {CONFIG['width']}x{CONFIG['height']}, "
          f"{CONFIG['samples_per_pixel']} сэмплов/пиксель...")

    start_time = time.time()

    image = render_image(
        CONFIG['width'], CONFIG['height'], CONFIG['samples_per_pixel'],
        camera.position, camera.forward, camera.right, camera.up,
        camera.half_width, camera.half_height,
        scene.vertices, scene.normals, scene.material_ids,
        scene.diffuse, scene.specular, scene.emission,
        scene.light_indices, scene.light_areas, scene.total_light_area,
        CONFIG['max_depth'], CONFIG['russian_roulette_depth']
    )

    elapsed = time.time() - start_time
    print(f"      Завершено за {elapsed:.1f} секунд")

    # 4. Постобработка и сохранение
    print("[4/4] Постобработка и сохранение...")
    image = tonemap_and_gamma(image, gamma=CONFIG['gamma'], exposure=CONFIG['exposure'])

    save_ppm("result.ppm", image)
    save_png("result.png", image)

    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)


if __name__ == "__main__":
    main()
