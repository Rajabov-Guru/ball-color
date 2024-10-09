import cv2
from scipy.spatial import KDTree
import numpy as np
import yaml

with open('colors_config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

ball_colors_dict_all_rgb = {tuple(color[:3]): color[3] for color in config_data['ball_colors_dict_all_rgb']}
glove_colors_dict_all_rgb = {tuple(color[:3]): color[3] for color in config_data['glove_colors_dict_all_rgb']}


class Color():
    def __init__(self, colors_dict):
        self.colors_dict = colors_dict
        self.names = []
        self.rgb_values = []
        for rgb, color_name in self.colors_dict.items():
            self.names.append(color_name)
            self.rgb_values.append(rgb)
        self.kdt_db = KDTree(self.rgb_values)

    def get_mask(self, crop):
        hh, ww = crop.shape[:2]
        yc = hh // 2
        xc = ww // 2
        radius = 13
        mask = np.zeros_like(crop)
        mask = cv2.circle(mask, (xc, yc), radius, (255, 255, 255), -1)
        result = crop
        result[mask == 0] = 0
        return result

    def convert_rgb_to_names(self, rgb_tuple):
        distance, index = self.kdt_db.query(rgb_tuple)
        return self.names[index]

    def get_main_color(self, crop):
        crop_prep = crop.copy()
        mask = self.get_mask(crop_prep)
        pixels = np.float32(mask.reshape(-1, 3))
        pixels = pixels[np.all(pixels != 0, axis=1)]
        n_colors = 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        return self.convert_rgb_to_names((dominant))


ball_color = Color(ball_colors_dict_all_rgb)
glove_color = Color(glove_colors_dict_all_rgb)