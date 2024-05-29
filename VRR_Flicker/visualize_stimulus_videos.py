import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import glfw
import sys
sys.path.append(r'E:\Py_codes\VRR_Real')
from G1_Calibration.compute_size_real import compute_scale_from_degree
from G1_Calibration.compute_x_y_location import compute_x_y_from_eccentricity

def generate_frame(x_center, y_center, x_scale, y_scale, size_value, screen_width, screen_height, time, freq, base_color, delta_color):
    t = time * freq
    if int(t) % 2 == 0:
        color = base_color + delta_color
    else:
        color = base_color - delta_color

    image = Image.new("RGB", (screen_width, screen_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    x_center = x_center + screen_width/2
    y_center = y_center + screen_height / 2

    # Draw an anti-aliased circle or rectangle
    if size_value == 'full':
        draw.rectangle(
            (
                x_center - x_scale*screen_width,
                y_center - y_scale*screen_height,
                x_center + x_scale*screen_width,
                y_center + y_scale*screen_height,
            ),
            fill=(color, color, color),
            outline=(color, color, color),  # Use the same color as outline to achieve anti-aliasing
        )
    else:
        draw.ellipse(
            (
                x_center - x_scale*screen_width,
                y_center - y_scale*screen_height,
                x_center + x_scale*screen_width,
                y_center + y_scale*screen_height,
            ),
            fill=(color, color, color),
            outline=(color, color, color),  # Use the same color as outline to achieve anti-aliasing
        )

    return np.array(image)

def generate_video(output_dir, size_list, vrr_f_list, duration, fps, base_color, delta_color):
    glfw.init()
    second_monitor = glfw.get_monitors()[1]
    screen_width, screen_height = glfw.get_video_mode(second_monitor).size
    total_frames = duration * fps
    for size_value in tqdm(size_list):
        for vrr_f_value in vrr_f_list:
            x_center, y_center = compute_x_y_from_eccentricity(eccentricity=0)
            x_scale, y_scale = compute_scale_from_degree(visual_degree=size_value)
            x_scale = x_scale / 2
            y_scale = y_scale / 2
            images = []
            for frame_num in range(total_frames):
                time = frame_num / fps
                frame = generate_frame(x_center, y_center, x_scale, y_scale, size_value, screen_width, screen_height, time, vrr_f_value, base_color, delta_color)
                images.append(frame)
            save_path = os.path.join(output_dir, f'Size_{size_value}_FRR_{vrr_f_value}_stimulus.mp4')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.mimwrite(save_path, images, fps=fps)

if __name__ == "__main__":
    import imageio
    size_list = ['full']#[0.5, 1, 16, 'full']
    vrr_f_list = [0.5, 2, 4, 8, 10, 12, 14, 16]
    duration = 2  # seconds
    fps = 120  # frames per second assume
    base_color = 100  # base color value
    delta_color = 20  # amplitude of color change

    generate_video('Videos_visualization', size_list, vrr_f_list, duration, fps, base_color, delta_color)
