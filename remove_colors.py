import cv2
import numpy as np

# Шаг 1: Прочитать изображение
image = cv2.imread('image.jpeg')

# Шаг 2: Преобразовать изображение в HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Шаг 3: Определить диапазоны цветов в формате HSV

# Желтый
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Зелёный
lower_green = np.array([35, 100, 50])
upper_green = np.array([85, 255, 255])

# Коричневый (широкий диапазон оттенков)
lower_brown = np.array([10, 100, 20])
upper_brown = np.array([20, 255, 200])

# Красный (два диапазона для охвата всех оттенков красного)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Розовый
lower_pink = np.array([145, 100, 100])
upper_pink = np.array([165, 255, 255])

# Оранжевый
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])

# Шаг 4: Создать маски для каждого цвета
mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)
mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
mask_pink = cv2.inRange(hsv_image, lower_pink, upper_pink)
mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)

# Объединение масок
mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # объединяем две маски для красного цвета
mask_combined = cv2.bitwise_or(mask_yellow, mask_green)
mask_combined = cv2.bitwise_or(mask_combined, mask_brown)
mask_combined = cv2.bitwise_or(mask_combined, mask_red)
mask_combined = cv2.bitwise_or(mask_combined, mask_pink)
mask_combined = cv2.bitwise_or(mask_combined, mask_orange)

# Шаг 5: Применить объединённую маску к изображению
result = cv2.bitwise_and(image, image, mask=mask_combined)

# Показать оригинальное и итоговое изображение
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image)
cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)
cv2.imshow('Filtered Image', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
