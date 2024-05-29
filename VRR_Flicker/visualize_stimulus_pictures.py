import os
import numpy as np
from PIL import Image, ImageDraw
import glfw
import sys
sys.path.append(r'E:\Py_codes\VRR_Real')
from G1_Calibration.compute_size_real import compute_scale_from_degree
from G1_Calibration.compute_x_y_location import compute_x_y_from_eccentricity

def generate_image_disk(x_center, y_center, x_scale, y_scale, screen_width, screen_height, color_value):
    image = Image.new("RGB", (screen_width, screen_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    x_center = x_center + screen_width/2
    y_center = y_center + screen_height / 2

    # Draw an anti-aliased white circle
    draw.ellipse(
        (
            x_center - x_scale*screen_width,
            y_center - y_scale*screen_height,
            x_center + x_scale*screen_width,
            y_center + y_scale*screen_height,
        ),
        fill=(color_value, color_value, color_value),
        outline=(color_value, color_value, color_value),  # Use the same color as outline to achieve anti-aliasing
    )

    return np.array(image)

def generate_image_square(x_center, y_center, x_scale, y_scale, screen_width, screen_height, color_value):
    image = Image.new("RGB", (screen_width, screen_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    x_center = x_center + screen_width/2
    y_center = y_center + screen_height / 2

    # Draw an anti-aliased rectangle
    draw.rectangle(
        (
            x_center - x_scale*screen_width,
            y_center - y_scale*screen_height,
            x_center + x_scale*screen_width,
            y_center + y_scale*screen_height,
        ),
        fill=(color_value, color_value, color_value),
        outline=(color_value, color_value, color_value),  # Use the same color as outline to achieve anti-aliasing
    )

    return np.array(image)

def generate_image_for_sizes(size_list, color_list):
    glfw.init()
    second_monitor = glfw.get_monitors()[1]
    screen_width, screen_height = glfw.get_video_mode(second_monitor).size
    images = []
    for size_value in size_list:
        for color_value in color_list:
            x_center, y_center = compute_x_y_from_eccentricity(eccentricity=0)
            x_scale, y_scale = compute_scale_from_degree(visual_degree=size_value)
            x_scale = x_scale / 2
            y_scale = y_scale / 2
            if size_value == "full":
                img = generate_image_square(x_center, y_center, x_scale, y_scale, screen_width, screen_height, color_value)
            else:
                img = generate_image_disk(x_center, y_center, x_scale, y_scale, screen_width, screen_height, color_value)
            images.append(img)
    return images

if __name__ == "__main__":
    size_list = [0.5, 1, 16, 'full']
    color_list = [25, 26, 35, 50, 178, 255] # Max is 255
    images = generate_image_for_sizes(size_list=size_list, color_list=color_list)
    for idx, img_array in enumerate(images):
        size_value = size_list[idx // len(color_list)]
        color_value = color_list[idx % len(color_list)]
        save_path = f'Disk_Sizes_Colors/Size_{size_value}_Color_{color_value}_stimulus.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(img_array).save(save_path)
