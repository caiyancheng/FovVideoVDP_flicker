import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import glfw
import sys
import cv2
import math

from LuminanceVRR2Sensitivity import LuminanceVRR2Sensitivity
import pandas as pd
luminance_vrr_2_sensitivity = LuminanceVRR2Sensitivity()

def compute_scale_from_degree(visual_degree, distance = 1):
    if visual_degree == 'full':
        return 1, 1
    screen_width_resolution = 3840
    screen_height_resolution = 2160
    # screen_width = 1.2176
    # screen_height = 0.6849
    screen_width = 1.225
    screen_height = 0.706
    W = math.tan(visual_degree/2 * math.pi / 180) * 2 * distance
    W_pixels = W / screen_width * screen_width_resolution
    W_scale = W_pixels / screen_width_resolution
    H_scale = W_pixels / screen_height_resolution

    return W_scale, H_scale

def compute_x_y_from_eccentricity(eccentricity, distance = 1):
    screen_width_resolution = 3840
    screen_height_resolution = 2160
    screen_width = 1.2176
    screen_height = 0.6849
    Y = 0.
    X = (distance * math.tan(eccentricity * math.pi / 180))/screen_width * 2
    return X, Y

def pre_gernerate_frame_unit(radius_value, screen_width, screen_height):
    x_center, y_center = compute_x_y_from_eccentricity(eccentricity=0)
    x_scale, y_scale = compute_scale_from_degree(visual_degree=radius_value)
    image = Image.new("RGB", (screen_width, screen_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    x_center = x_center + screen_width / 2
    y_center = y_center + screen_height / 2

    # Draw an anti-aliased circle or rectangle
    if abs(radius_value - 27.46) < 0.01: #full screen
        draw.rectangle(
            (
                x_center - x_scale * screen_width,
                y_center - y_scale * screen_height,
                x_center + x_scale * screen_width,
                y_center + y_scale * screen_height,
            ),
            fill=(1, 1, 1),
            outline=(1, 1, 1),  # Use the same color as outline to achieve anti-aliasing
        )
    else:
        draw.ellipse(
            (
                x_center - x_scale * screen_width,
                y_center - y_scale * screen_height,
                x_center + x_scale * screen_width,
                y_center + y_scale * screen_height,
            ),
            fill=(1, 1, 1),
            outline=(1, 1, 1),  # Use the same color as outline to achieve anti-aliasing
        )
    frame_unit = np.array(image)
    return frame_unit


def generate_frame(frame_unit, time, vrr_f_value, Luminance, delta_Luminance):
    t = time * vrr_f_value
    if int(t) % 2 == 0:
        test_L = Luminance + delta_Luminance
    else:
        test_L = Luminance - delta_Luminance
    reference_L = Luminance
    return frame_unit * test_L, frame_unit * reference_L


def generate_quest_luminance_one_video(radius_value, frr_value, fps, screen_width, screen_height, Luminance,
                                       delta_Luminance, total_frames):
    images_test = []
    images_reference = []
    frame_unit = pre_gernerate_frame_unit(radius_value, screen_width, screen_height)
    # For Debug:
    # import matplotlib.pyplot as plt
    # plt.imshow(frame_unit)

    for frame_num in range(total_frames):
        time = frame_num / fps
        frame_test, frame_reference = generate_frame(frame_unit, time, frr_value, Luminance, delta_Luminance)
        images_test.append(frame_test.astype(np.float16))
        images_reference.append(frame_reference.astype(np.float16))
    video_test = np.stack(images_test, axis=3)
    video_reference = np.stack(images_reference, axis=3)

    return video_test, video_reference


def generate_quest_luminance_video_json(output_dir, duration, fps, down_rate):
    glfw.init()
    second_monitor = glfw.get_monitors()[1]
    screen_width, screen_height = glfw.get_video_mode(second_monitor).size
    screen_width = int(screen_width / down_rate)
    screen_height = int(screen_height / down_rate)
    total_frames = duration * fps

    df = pd.read_csv(r'yancheng2024_sensitivity_average.csv') #E:\Matlab_codes\csf_datasets\raw_data\yancheng2024/
    query_radius_list = df['Radius'].sort_values()
    query_frr_list = df['FRR'].sort_values()
    os.makedirs(output_dir, exist_ok=True)

    for index in tqdm(range(len(df))):
        radius_value = query_radius_list[index]
        frr_value = query_frr_list[index]
        sub_df = df[(df['Radius'] == radius_value) & (df['FRR'] == frr_value)]
        Luminance = float(sub_df['Luminance']) # 找到mean Luminance
        contrast = 1 / luminance_vrr_2_sensitivity.LT2S(Luminance, frr_value)
        delta_Luminance = float(Luminance * contrast)

        video_test, video_reference = generate_quest_luminance_one_video(radius_value, frr_value, fps, screen_width,
                                                                         screen_height, Luminance, delta_Luminance,
                                                                         total_frames)
        stimulus_dict = {
            'video_test': video_test.tolist(),
            'video_reference': video_reference.tolist(),
        }
        with open(os.path.join(output_dir, f'Radius_{radius_value}_FRR_{frr_value}_stimulus.json'), 'w') as fp:
            json.dump(stimulus_dict, fp)



if __name__ == "__main__":
    # size_list = [0.5, 1, 16, 'full']
    # # vrr_f_list = [0.5, 2, 4, 8, 10, 12, 14, 16]
    # vrr_f_list = [0.5, 2, 4, 8, 10, 11.9, 13.3, 14.9]
    duration = 2  # seconds
    fps = 120  # frames per second assume
    down_rate = 60

    generate_quest_luminance_video_json(f'New_FVVDP_Quest_Luminance_Video_FPS_{fps}_downsampling_{down_rate}',
                                        duration, fps, down_rate)
