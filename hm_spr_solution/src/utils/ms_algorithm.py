from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import deepcopy
from pathlib import Path
import datetime
import argparse

BACKGROUND = [0, 0, 0]
LOWER = [255, 96, 55]
MIDDLE = [221, 255, 51]
RIVET = [61, 245, 61]
UPPER = [61, 61, 245]
COLOUR_NAMES = ["background", "lower", "middle", "rivet", "upper"]

def get_image_array(path):
    img_arr = None
    
    img = Image.open(path)
    if img is not None:
        img_arr = np.array(img)
    
    return img_arr

def colour_quantisation(arr_original):
    arr = deepcopy(arr_original)
    colours = [BACKGROUND, LOWER, MIDDLE, RIVET, UPPER]

    for w in range(arr.shape[0]):
        for h in range(arr.shape[1]):
            max_diff = 255 * 3
            temp_diff = 0
            current_pixel = arr[w, h]
            quantised_pixel = [255, 255, 255]
            set_channel_idx = None
            # print(current_pixel)
            
            # 있는지 확인
            matching_flag = False
            for c in colours:
                if (current_pixel == c).all():
                    matching_flag = True
            if matching_flag:
                continue       
                    
            for colour_idx, c in enumerate(colours):
                # print(f"[{COLOUR_NAMES[colour_idx]}] : {c}")
                
                for channel_idx in range(arr.shape[2]):
                    temp_diff += np.abs(current_pixel[channel_idx] - c[channel_idx])
                    # print(f"{temp_diff = }")
                
                if temp_diff == 0:
                    continue
                
                elif temp_diff < max_diff:
                    # print(f"It's smaller!, [{colour_idx}] colour, [{COLOUR_NAMES[colour_idx]}]")
                    max_diff = temp_diff
                    quantised_pixel = colours[colour_idx]
                    set_channel_idx = colour_idx
                
                temp_diff = 0
                # print(f"[{max_diff = }]")
            arr[w, h] = quantised_pixel
            # print(f"before: {current_pixel}, after: {quantised_pixel} -> [{COLOUR_NAMES[set_channel_idx]}]")

    return arr

def get_line_coords_via_upper_corners(img_arr, colour):
    condition_1 = img_arr[:, :, 0] == colour[0]
    condition_2 = img_arr[:, :, 1] == colour[1]
    condition_3 = img_arr[:, :, 2] == colour[2]
    
    channel_1_coords, channel_2_coords = np.where(condition_1 & condition_2 & condition_3)
    
    # 좌우 모서리 기준점 좌표
    # 모서리 기준점 좌표 순서는 [x축, y축]
    upper_left = [0, 0]
    upper_right = [img_arr.shape[1] - 1, 0]

    # 최대 거리 설정(이미지 대각선)
    max_diff_upper_left = img_arr.shape[0] * img_arr.shape[1]
    max_diff_upper_right = img_arr.shape[0] * img_arr.shape[1]

    # 찾아낸 좌표들 기본값 설정
    left_x = None
    left_y = None

    right_x = None
    right_y = None

    # 좌표 값마다 모서리 기준점과 거리(L1) 비교
    # --- 이미지 표현에서 첫번째 채널 값은 y축, 두번째 채널 값은 x축을 표현
    for idx, (y, x) in enumerate(zip(channel_1_coords, channel_2_coords)):
        
        left_distance = np.abs(upper_left[0] - x) + np.abs(upper_left[1] - y)
        right_distance = np.abs(upper_right[0] - x) + np.abs(upper_right[1] - y)
        
        if left_distance < max_diff_upper_left:
            left_x = x
            left_y = y
            
            max_diff_upper_left = left_distance
            print(f"left coords found [{left_y}, {left_x}]")
            
        elif right_distance < max_diff_upper_right:
            right_x = x
            right_y = y
            
            max_diff_upper_right = right_distance
            print(f"right coords found [{right_y}, {right_x}]")
            
    # 시각화
    # plt.imshow(img_arr)
    # plt.plot([left_x_rivet, right_x_rivet], [left_y_rivet, right_y_rivet], "wo", linestyle="--")
    # plt.show()
    
    return left_x, left_y, right_x, right_y

def get_pixel_per_mm_from_rivet_coords(x1, x2, y1, y2):
    standard = 7.75
    line_1_squared = (y1 - y2) ** 2
    line_2_squared = (x1 - x2) ** 2
    rivet_diameter_pixel = np.sqrt(line_1_squared + line_2_squared)
    
    pixel_per_mm = standard / rivet_diameter_pixel
    print(f"{pixel_per_mm =}")
    return pixel_per_mm

def get_matrix(x1, y1, x2, y2):
    matrix = np.array([
        [x1, x2],
        [y1, y2]
    ])
    
    return matrix


def rotate_vector_around_origin(vector, rad):
    x, y = vector
    if rad > 0:
        xx = x * math.cos(rad) + y * math.sin(rad)
        yy = -x * math.sin(rad) + y * math.cos(rad)
    elif rad < 0:
        xx = x * math.cos(rad) + -y * math.sin(rad)
        yy = x * math.sin(rad) + y * math.cos(rad)
    else:
        xx = x
        yy = y        
    
    rotated_vector = np.array([xx, yy])
    print("====== rotate ======")
    print(f"original vector: \n{vector}\nrad: {rad}\n")
    print(f"rotated vector: \n{rotated_vector}")
    print("====================")
    return rotated_vector

def rotate_vector(vector, two_y_points, rad):
    x, y = vector
    left_y, right_y = two_y_points
    c, s = np.cos(rad), np.sin(rad)
    
    # y축에서 0이 더 높이 위치한 이미지상에서 left_y 가 right_y보다 더 높은 값을 갖는다면,
    # 더 낮게 위치하고 있으므로 clockwise로 진행 필요
    if left_y > right_y:
        j = np.matrix([
            [c, -s],
            [s, c]
        ])
        
    # 반대로 counter-clockwise
    elif left_y < right_y:
        j = np.matrix([
            [c, s],
            [-s, c]
        ])
    else:
        j = np.matrix([
            [1, 0],
            [0, 1]
        ])
        
    m = np.dot(j, [x, y])
    rotated_vector = np.array([m.T[0].item(), m.T[1].item()], dtype=float)

    print("====== rotate ======")
    print(f"original vector: \n{vector}\nrad: {rad}\n")
    print(f"rotated vector: \n{rotated_vector}")
    print("====================")
    
    return rotated_vector

def get_angle_between_two_vectors_from_origin(u, v):
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    deg = np.degrees(rad)
    
    print("--- get angle between two vectors ---")
    print(f"u: \n{u}\nv: \n{v}")
    print(f"rad: {rad}, deg: {deg}")
    print("-------------------------------------")
    return rad, deg

def move_to_origin(matrix, x1, y1):
    matrix_copy = deepcopy(matrix)

    # 0,0 으로 origin(x1, y1) 옮기기
    matrix_copy[0, 0] -= x1
    matrix_copy[0, 1] -= x1
    matrix_copy[1, 0] -= y1
    matrix_copy[1, 1] -= y1
    print("--- move to origin ---")
    print(f"before: \n{matrix}\nafter: \n{matrix_copy}")
    print("----------------------")
    return matrix_copy

def move_to_x_axis_with_origin_anchored(matrix):
    # x축으로 거리 유지하며 vector 옮기기
    vector_length = np.sqrt((matrix[0, 1] - matrix[0, 0]) ** 2 + (matrix[1, 1] - matrix[1, 0]) ** 2)
    matrix_on_x_axis = [
        [0, vector_length],
        [0, 0]
    ]

    matrix_on_x_axis = np.array(matrix_on_x_axis)

    print("--- move to x axis ---")
    print(f"length: {vector_length}\nafter: \n{matrix_on_x_axis}")
    print("----------------------")
    
    return matrix_on_x_axis


def get_head_diff(path, is_gen=True):
    
    # 이미지 어레이 가져오기
    img_arr = None
    if isinstance(path, str) or isinstance(path, Path):
        img_arr = get_image_array(path)
    elif isinstance(path, np.ndarray):
        img_arr = path
    
    # 생성된 이미지일 경우 양자화
    if is_gen:
        img_arr = colour_quantisation(img_arr)
    
    # 리벳 직경 좌표 구하기
    left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet = get_line_coords_via_upper_corners(img_arr, RIVET)
    
    # 리벳 직경에 따른 mm 구하기
    pixel_per_mm = get_pixel_per_mm_from_rivet_coords(left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet)
    
    # 상판 끝부분 좌표 구하기
    left_x_upper, left_y_upper, right_x_upper, right_y_upper = get_line_coords_via_upper_corners(img_arr, UPPER)
    
    # 각각 두 점으로 이루어진 매트릭스 구하기
    rivet_matrix = get_matrix(left_x_rivet, left_y_rivet, right_x_rivet, right_y_rivet)
    upper_matrix = get_matrix(left_x_upper, left_y_upper, right_x_upper, right_y_upper)
    
    # x축으로 돌릴 때 사용할 좌표
    x_trans = upper_matrix[0, 0]
    y_trans = upper_matrix[1, 0]
    
    # 상판의 x축으로 옮긴 좌표 구하기
    upper_origin = move_to_origin(upper_matrix, x_trans, y_trans) # (0, 0)으로 옮긴 upper_matrix
    upper_on_x_axis = move_to_x_axis_with_origin_anchored(upper_origin) # (0, 0)에서 x축으로 맞춰진 matrix
    
    # x축으로 옮겨진 상판을 토대로 옮겨진 각도 구하기
    rad, deg = get_angle_between_two_vectors_from_origin(upper_origin.T[1], upper_on_x_axis.T[1]) # 첫번째는 (0, 0)이고 두번째가 (0, 0)을 기점으로 하는 벡터
    
    # 리벳 (0, 0)으로 옮기기
    rivet_origin = move_to_origin(rivet_matrix, x_trans, y_trans)
    
    # 리벳의 두 점들 rotate하기(각각 (0, 0)에서 떨어진 벡터)
    # 기준으로는 x축으로 옮겨지기 전 좌표들의 y값 2개를 전달
    rivet_rotated_1 = rotate_vector(rivet_origin.T[0], upper_matrix[1], rad)
    rivet_rotated_2 = rotate_vector(rivet_origin.T[1], upper_matrix[1], rad)
    
    rivet_matrix_rotated = np.array([
        rivet_rotated_1,
        rivet_rotated_2
    ]).T # transposed
    
    # rotated 리벳 좌표들의 y 좌표값만 구하기
    # 왜냐하면 x축에 align 되어서 그 값이 곧 거리이기 때문
    # 또한 -1을 곱하여주는데 이 이유는 이미지상 y축의 숫자 크기 증가 방향이 계산 시 가정된 사람의 y축 숫자 크기 방향과 반대이기 때문
    pixel_distances_for_two_rivet_points = -rivet_matrix_rotated[1]
    
    # # 최소값 구하기...?
    # result = min(pixel_distances_for_two_rivet_points) * pixel_per_mm
    
    # 두 값 평균?
    result = np.mean(pixel_distances_for_two_rivet_points * pixel_per_mm)
    
    print(f"~~~~~ Final result: {result} ~~~~~")
    
    fig = plt.figure()
    plt.imshow(img_arr)
    plt.plot([left_x_rivet, right_x_rivet], [left_y_rivet, right_y_rivet], "wo", linestyle="--")
    plt.plot([left_x_upper, right_x_upper], [left_y_upper, right_y_upper], "wo", linestyle="--")
    
    plt.plot(rivet_matrix_rotated[0], rivet_matrix_rotated[1], "yo", linestyle="--")
    plt.plot(upper_on_x_axis[0], upper_on_x_axis[1], "ro", linestyle="--")
    plt.plot(rivet_origin[0], rivet_origin[1], "go", linestyle="--")
    plt.plot(upper_origin[0], upper_origin[1], "bo", linestyle="--")
    
    try:
        plt.savefig(f"./result_{str(path)}.png")
    except:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"./result_{timestamp}.png")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default=None)
    args = parser.parse_args()
    
    get_head_diff(args.img_path)
    