"""
Постобработка: тонемаппинг, гамма-коррекция, сохранение изображений.
"""

import numpy as np


def reinhard_tonemap(image: np.ndarray) -> np.ndarray:
    """Тонемаппинг Рейнхарда: L / (1 + L)"""
    return image / (1.0 + image)


def exposure_tonemap(image: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Экспозиционный тонемаппинг: 1 - exp(-exposure * L)"""
    return 1.0 - np.exp(-exposure * image)


def tonemap_and_gamma(image: np.ndarray, gamma: float = 2.2,
                      normalize_method: str = 'reinhard',
                      exposure: float = 1.5) -> np.ndarray:
    """
    Тонемаппинг и гамма-коррекция.

    normalize_method: 
        'max' - нормализация по максимальной яркости
        'mean' - по средней яркости (с множителем)
        'reinhard' - тонемаппинг Рейнхарда
        'exposure' - экспозиционный тонемаппинг
        число - по заданному значению
    """
    image = image.copy()

    # Тонемаппинг / нормализация
    if normalize_method == 'max':
        max_val = np.max(image)
        if max_val > 0:
            image = image / max_val
    elif normalize_method == 'mean':
        # Используем percentile для робастности к выбросам
        target = np.percentile(image, 90)
        if target > 0:
            image = image / (target * 2)
    elif normalize_method == 'reinhard':
        image = reinhard_tonemap(image * exposure)
    elif normalize_method == 'exposure':
        image = exposure_tonemap(image, exposure)
    elif isinstance(normalize_method, (int, float)):
        image = image / normalize_method

    # Отсечение значений выше 1
    image = np.clip(image, 0, 1)

    # Гамма-коррекция: V_corr = V^(1/γ)
    image = np.power(image, 1.0 / gamma)

    return image


def save_ppm(filename: str, image: np.ndarray):
    """Сохранение в формате PPM (P3 - текстовый)."""
    height, width = image.shape[:2]
    image_8bit = (image * 255).astype(np.uint8)

    with open(filename, 'w') as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for y in range(height):
            row = []
            for x in range(width):
                r, g, b = image_8bit[y, x]
                row.append(f"{r} {g} {b}")
            f.write(" ".join(row) + "\n")

    print(f"Сохранено: {filename}")


def save_png(filename: str, image: np.ndarray):
    """Сохранение в формате PNG."""
    try:
        from PIL import Image
        image_8bit = (image * 255).astype(np.uint8)
        img = Image.fromarray(image_8bit, 'RGB')
        img.save(filename)
        print(f"Сохранено: {filename}")
    except ImportError:
        print("PIL не найден, сохраняю в PPM")
        save_ppm(filename.replace('.png', '.ppm'), image)

