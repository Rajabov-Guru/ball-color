import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from settings import settings
from utils import which_color_is_it


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_image(img, size=(settings.crop_size, settings.crop_size)):
    resized = cv2.resize(img, size)
    height, width, _ = resized.shape
    square_size = settings.mini_crop_size
    # Calculate the top-left corner of the square (center the square)
    start_x = (width - square_size) // 2
    start_y = (height - square_size) // 2

    # Crop the square from the image
    cropped_image = resized[start_y:start_y + square_size, start_x:start_x + square_size]

    if settings.show_mini_crop:
        # Save or display the cropped image
        cv2.imwrite('cropped_billiard_ball.jpg', cropped_image)
        cv2.namedWindow('Cropped Image', cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('Cropped Image', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_image


def find_dominant_color(image, k=settings.cluster_amount):
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_
    return colors


def plot_colors(colors):
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    step = 300 // len(colors)

    for i, color in enumerate(colors):
        rect[:, i * step:(i + 1) * step] = color.astype(int)

    plt.axis('off')
    plt.imshow(rect)
    plt.show()


def map_color_to_ball(dominant_color):
    closest_ball = None
    min_distance = float('inf')

    for ball, color in settings.COLOR_MAP.items():
        distance = np.linalg.norm(np.array(color) - dominant_color)
        if distance < min_distance:
            min_distance = distance
            closest_ball = ball

    return closest_ball


def identify_ball_color(image_path: str):
    img = load_image(image_path)
    preprocessed_img = preprocess_image(img)

    dominant_colors = find_dominant_color(preprocessed_img)
    filtered_dominants = []

    for dominant in dominant_colors:
        color = which_color_is_it(dominant)
        if color != "white" and color is not None:
            filtered_dominants.append(dominant)

    if settings.show_plot:
        plot_colors(dominant_colors)

    if len(filtered_dominants) > 0:
        dominant_color = filtered_dominants[0]
    else:
        dominant_color = dominant_colors[0]

    new_color_array = [
        int(round(dominant_color[0])),
        int(round(dominant_color[1])),
        int(round(dominant_color[2]))
    ]

    identified_ball = map_color_to_ball(dominant_color)
    return identified_ball, new_color_array, dominant_colors, filtered_dominants
