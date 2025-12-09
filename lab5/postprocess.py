"""
Постобработка: тонемаппинг, гамма-коррекция, сохранение изображений.
"""

import numpy as np


def reinhard_tonemap(image):
    """
    Тонемаппинг Рейнхарда: L_out = L / (1 + L)
    Преобразует HDR изображение в диапазон [0, 1].
    """
    return image / (1.0 + image)


def tonemap_and_gamma(image, gamma=2.2, exposure=1.5):
    """
    Постобработка изображения:
    1. Тонемаппинг Рейнхарда (HDR -> LDR)
    2. Отсечение значений выше 1
    3. Гамма-коррекция
    
    Параметры:
        gamma: показатель гамма-коррекции (обычно 2.2)
        exposure: коэффициент экспозиции (яркость)
    """
    image = image.copy()

    # Тонемаппинг Рейнхарда
    image = reinhard_tonemap(image * exposure)

    # Отсечение значений выше 1
    image = np.clip(image, 0, 1)

    # Гамма-коррекция: V_corr = V^(1/γ)
    image = np.power(image, 1.0 / gamma)

    return image


def save_ppm(filename, image):
    """
    Сохранение в формате PPM (P3 - текстовый).
    
    Формат PPM:
    - P3 - магическое число (текстовый RGB)
    - ширина высота
    - максимальное значение (255)
    - RGB значения пикселей
    """
    height, width = image.shape[:2]
    
    # Преобразуем в 8-бит
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


def save_png(filename, image):
    """Сохранение в формате PNG (требует PIL/Pillow)."""
    try:
        from PIL import Image
        image_8bit = (image * 255).astype(np.uint8)
        img = Image.fromarray(image_8bit, 'RGB')
        img.save(filename)
        print(f"Сохранено: {filename}")
    except ImportError:
        print("PIL не найден, сохраняю в PPM")
        save_ppm(filename.replace('.png', '.ppm'), image)
