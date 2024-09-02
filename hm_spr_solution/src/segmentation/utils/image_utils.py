import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs import Config
import traceback

def erase_coloured_text_and_lines(img_path):
    # 1. 하늘색 HSV : 186, 98%, 95%
    # 2. 녹색 HSV : 118, 98%, 95%
    #               118, 98%, 65%
    #               118, 98%, 35%
    # 3. 빨강 HSV : 0, 98%, 65%
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
        image = img.copy()
    elif isinstance(img_path, np.ndarray):
        image = img_path
    else:
        image = np.array(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 위의 기준보다 좀 더 널럴하게 Saturation 맞췄음. 아닐 경우 잘 인지 안 됨
    s_max = 255
    v_max = 255
    lower = np.array([0, s_max*0.60, v_max*0.30])
    upper = np.array([200, s_max*0.98, v_max*0.98])
    # lower = np.array([0, 128, 128])
    # upper = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((9, 9), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)  #// make dilation image

    dst = cv2.inpaint(image, dilated_mask, 5, cv2.INPAINT_NS)

    return dst

def get_transforms(is_train):
    def _lambda_contrast_fn(image, **kwargs):
        image = 255 - image
        image = image.astype(np.uint8)
        return image
    
    def _lambda_minmax_fn(image, **kwards):
        image = image.astype(np.float32)
        image = image / 255.0
        return image
    
    if is_train:
        transforms = A.Compose([
            # A.Lambda(image=_lambda_contrast_fn),
            A.RGBShift(r_shift_limit=(-50, 50), g_shift_limit=(-50, 50), b_shift_limit=(-50, 50)),
            # A.ColorJitter(brightness=(0.8, 1), contrast=(0.8, 1), saturation=(0.5, 1), hue=(-0.5, 0.5)),
            A.RandomCrop(height=Config.CROP_HEIGHT, width=Config.CROP_HEIGHT, p=0.5),
            A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH),
            A.HorizontalFlip(),
            # A.GridDistortion(),            
            # A.Blur(),
            # A.Lambda(image=_lambda_minmax_fn),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ])
    else:
        transforms = A.Compose([
            A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # A.Lambda(image=_lambda_minmax_fn),
            ToTensorV2(transpose_mask=True),
        ])
    return transforms 


def get_label_info(labeltxt_path):
    try:
        with open(labeltxt_path, "r") as f:
            label_txt = f.readlines()[1:]
    except Exception as e:
        print(e) 
        traceback.print_exc()
        label_txt = None
        
    label_to_rgb = {}
    if label_txt is None:
        label_to_rgb[0] = [0, 0, 0] # background
        label_to_rgb[1] = [255, 96, 55] # lower
        label_to_rgb[2] = [221, 255, 51] # middle
        label_to_rgb[3] = [61, 245, 61] # rivet
        label_to_rgb[4] = [61, 61, 245] # upper
        
    else:
        for txt_idx, txt in enumerate(label_txt):
            divider_1 = txt.find(":")
            divider_2 = txt.find("::")

            label_name = txt[:divider_1]
            label_value = txt[divider_1+1:divider_2]
            rgb_values = list(map(int, label_value.split(",")))

            label_to_rgb[txt_idx] = rgb_values

    return label_to_rgb