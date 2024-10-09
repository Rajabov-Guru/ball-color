class Settings:
    show_mini_crop = False
    show_plot = False
    output_dominant_colors = True

    test_crops_dir = "crops"
    test_color = "brown"
    custom_test = None
    # custom_test = {
    #      "crop_path": "crops/brown/crop_678_10.jpeg",
    #      "expected": "brown"
    # }

    crop_size = 100
    mini_crop_size = 60
    cluster_amount = 3

    COLOR_MAP = {
        'red': (201, 10, 60),  # crops/red/crop_783_8.jpeg
        'pink': (230, 122, 126),
        'yellow': (234, 173, 78),  # needs tuning
        'blue': (8, 60, 136),
        'orange': (229, 117, 66),  # needs tuning
        'green': (71, 107, 93),  # needs tuning
        'brown': (162, 117, 67),  # crops/brown/crop_678_10.jpeg
        'black': (40, 57, 54),
        'white': (221, 206, 153)  # crops/white/crop_633_10.jpeg
    }


settings = Settings()