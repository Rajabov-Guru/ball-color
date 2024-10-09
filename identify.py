import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from settings import settings
from utils import which_color_is_it


def get_most_spreaded_color(image_path, n):
    img = load_image(image_path)
    image = preprocess_image(img)

    # Reshape the image to a 2D array of pixels (height*width, 3)
    pixels = image.reshape(-1, 3)

    # Use K-means clustering to cluster the pixel intensities
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pixels)

    # Get the cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_

    # Get the number of pixels in each cluster
    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    # Sort counts in descending order and get the indices of the sorted order
    sorted_indices = np.argsort(-counts)

    # Select the top 3 most spread colors
    top_colors = dominant_colors[sorted_indices[:n]]

    # Convert to int and return the top 3 most spread colors
    top_colors = top_colors.astype(int)

    # TODO: filter top colors

    return top_colors


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(img, size=(settings.crop_size, settings.crop_size)):
    resized = cv2.resize(img, size)
    return resized


def mini_crop_image(image):
    height, width, _ = image.shape
    square_size = settings.mini_crop_size
    # Calculate the top-left corner of the square (center the square)
    start_x = (width - square_size) // 2
    start_y = (height - square_size) // 2

    # Crop the square from the image
    cropped_image = image[start_y:start_y + square_size, start_x:start_x + square_size]
    return cropped_image


def get_mask(crop):
    hh, ww = crop.shape[:2]
    yc = hh // 2
    xc = ww // 2
    radius = round(ww / 2) - 10
    mask = np.zeros_like(crop)
    mask = cv2.circle(mask, (xc, yc), radius, (255, 255, 255), -1)
    result = crop
    result[mask == 0] = 0
    return result


def preprocess_image(img):
    resized = resize_image(img)
    masked = get_mask(resized)

    result_image = masked

    if settings.show_mini_crop:
        # Save or display the cropped image
        cv2.imwrite('cropped_billiard_ball.jpg', result_image)
        cv2.namedWindow('Cropped Image', cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('Cropped Image', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_image


def find_dominant_color(image, k=settings.cluster_amount):
    pixels = np.float32(image.reshape(-1, 3))
    pixels = pixels[np.all(pixels != 0, axis=1)]

    n_colors = k
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    palette_and_counts = list(zip(palette, counts))

    palette_and_counts.sort(key=lambda x: x[1], reverse=True)

    sorted_palette, sorted_counts = zip(*palette_and_counts)
    dominant = sorted_palette[:k]
    return dominant

    # kmeans = KMeans(n_clusters=k)
    # kmeans.fit(pixels)
    #
    # dominant_colors = kmeans.cluster_centers_
    #
    # labels, counts = np.unique(kmeans.labels_, return_counts=True)
    #
    # sorted_indices = np.argsort(-counts)
    #
    # top_colors = dominant_colors[sorted_indices[:k]]
    #
    # top_colors = top_colors.astype(int)
    #
    # return top_colors


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


def identify_ball_color(image_path: str = None, image_crop = None):
    if image_crop is None and image_path:
        img = load_image(image_path)
    else:
        img = image_crop

    img = preprocess_image(img)

    dominant_colors = find_dominant_color(img)
    filtered_dominants = []
    white_dominants = []
    color_set = set()

    for dominant in dominant_colors:
        color, _ = which_color_is_it(dominant)
        if color == 'white':
            white_dominants.append(dominant)
            continue
        if color not in color_set and color is not None:
            filtered_dominants.append(dominant)
            color_set.add(color)

    if len(white_dominants) > 0 and len(filtered_dominants) == 0:
        min_distance = 1000
        true_white = white_dominants[0]
        for dominant in white_dominants:
            color, distance = which_color_is_it(dominant)
            if distance < min_distance:
                true_white = dominant
                min_distance = distance

        filtered_dominants.append(true_white)

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
